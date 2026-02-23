import hashlib
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import redis
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse
from minio.error import S3Error
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from api.auth import require_api_key
from api.rate_limit import limiter, read_limit, submit_limit
from shared.minio_client import StorageClient
from shared.models import (
    Job,
    JobDedup,
    JobStage,
    JobStatus,
    RiskResult,
    get_session,
    init_db,
)
from shared.redis_queue import JobQueue
from shared.settings import settings

logger = logging.getLogger(__name__)

_RAW_PREFIX = "contracts/raw"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Contract Risk Pipeline", lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
        headers={"Retry-After": "60"},
    )


app.add_middleware(SlowAPIMiddleware)


@app.middleware("http")
async def enforce_upload_size(request: Request, call_next):
    # Short-circuits oversized uploads before multipart body parsing starts.
    if request.method == "POST" and request.url.path == "/jobs":
        raw = request.headers.get("content-length")
        if raw is None:
            return JSONResponse(
                status_code=411, content={"detail": "Content-Length header required"}
            )
        try:
            length = int(raw)
        except ValueError:
            return JSONResponse(
                status_code=400, content={"detail": "Invalid Content-Length header"}
            )
        if length > settings.max_upload_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"Request body exceeds maximum of "
                        f"{settings.max_upload_bytes} bytes"
                    )
                },
            )
    return await call_next(request)


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str
    filename: str


class JobStatusResponse(BaseModel):
    id: str
    status: str
    stage: str
    filename: str
    retry_count: int
    error: Optional[str]
    created_at: datetime


class ReportResponse(BaseModel):
    job_id: str
    filename: str
    overall_score: int
    risk_level: str
    clause_summary: dict
    flags: list
    report_url: str


class JobListItem(BaseModel):
    id: str
    status: str
    stage: str
    filename: str
    created_at: datetime


@app.get("/health")
def health():
    db = get_session()
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    finally:
        db.close()

    try:
        queue = JobQueue()
        queue.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    healthy = db_ok and redis_ok
    return {
        "status": "ok" if healthy else "degraded",
        "services": {"postgres": db_ok, "redis": redis_ok},
    }


def _dedup_key(client_key: Optional[str], pdf_bytes: bytes) -> str:
    # Namespaced so a 64-hex client key cannot collide with a content hash.
    if client_key:
        return f"client:{client_key}"
    return f"content:{hashlib.sha256(pdf_bytes).hexdigest()}"


@app.post(
    "/jobs",
    response_model=JobCreatedResponse,
    status_code=201,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(submit_limit)
async def submit_job(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()

    # Defensive: Content-Length can be forged or absent; verify actual body size.
    if len(pdf_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Uploaded file exceeds maximum of {settings.max_upload_bytes} bytes"
            ),
        )

    dedup_key = _dedup_key(idempotency_key, pdf_bytes)

    # Fast path: existing dedup row short-circuits before any storage or queue work.
    db = get_session()
    try:
        existing = db.query(JobDedup).filter(JobDedup.key == dedup_key).first()
        if existing is not None:
            job = db.get(Job, existing.job_id)
            if job is not None:
                response.status_code = 200
                response.headers["Idempotent-Replay"] = "true"
                return JobCreatedResponse(
                    job_id=job.id, status=job.status, filename=job.filename
                )
    finally:
        db.close()

    job_id = str(uuid.uuid4())
    object_key = f"{_RAW_PREFIX}/{job_id}.pdf"

    try:
        storage = StorageClient()
        storage.upload_bytes(object_key, pdf_bytes, content_type="application/pdf")
    except S3Error as exc:
        logger.error("api: storage upload failed for job %s: %s", job_id, exc)
        raise HTTPException(status_code=503, detail="Storage service unavailable")

    db = get_session()
    try:
        job = Job(
            id=job_id,
            status=JobStatus.QUEUED,
            stage=JobStage.INGESTION,
            object_key=object_key,
            filename=file.filename,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(job)
        db.add(JobDedup(key=dedup_key, job_id=job_id))
        try:
            db.commit()
        except IntegrityError:
            # Concurrent submission won the dedup race; return the winner.
            db.rollback()
            existing = db.query(JobDedup).filter(JobDedup.key == dedup_key).first()
            if existing is not None:
                winner = db.get(Job, existing.job_id)
                if winner is not None:
                    logger.info(
                        "api: idempotency race; returning winner job_id=%s",
                        winner.id,
                    )
                    response.status_code = 200
                    response.headers["Idempotent-Replay"] = "true"
                    return JobCreatedResponse(
                        job_id=winner.id,
                        status=winner.status,
                        filename=winner.filename,
                    )
            raise
    finally:
        db.close()

    try:
        queue = JobQueue()
        queue.enqueue(job_id)
    except redis.ConnectionError as exc:
        logger.error("api: failed to enqueue job %s: %s", job_id, exc)
        raise HTTPException(status_code=503, detail="Queue service unavailable")

    logger.info("api: job submitted job_id=%s filename=%s", job_id, file.filename)
    return JobCreatedResponse(
        job_id=job_id, status=JobStatus.QUEUED, filename=file.filename
    )


@app.get(
    "/jobs",
    response_model=list[JobListItem],
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(read_limit)
def list_jobs(request: Request, status: Optional[str] = Query(default=None)):
    db = get_session()
    try:
        q = db.query(Job).order_by(Job.created_at.desc())
        if status:
            try:
                status_enum = JobStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unknown status: {status}")
            q = q.filter(Job.status == status_enum)
        jobs = q.limit(100).all()
        return [
            JobListItem(
                id=j.id,
                status=j.status,
                stage=j.stage,
                filename=j.filename,
                created_at=j.created_at,
            )
            for j in jobs
        ]
    finally:
        db.close()


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(read_limit)
def get_job(request: Request, job_id: str):
    db = get_session()
    try:
        job = db.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobStatusResponse(
            id=job.id,
            status=job.status,
            stage=job.stage,
            filename=job.filename,
            retry_count=job.retry_count,
            error=job.error,
            created_at=job.created_at,
        )
    finally:
        db.close()


@app.get(
    "/jobs/{job_id}/report",
    response_model=ReportResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(read_limit)
def get_report(request: Request, job_id: str):
    db = get_session()
    try:
        job = db.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Job is not completed yet (status={job.status}, stage={job.stage})"
                ),
            )

        result = db.query(RiskResult).filter(RiskResult.job_id == job_id).first()
        if result is None:
            raise HTTPException(status_code=404, detail="Report not found")

        storage = StorageClient()
        report_url = storage.presigned_url(result.report_key, expires_seconds=3600)

        return ReportResponse(
            job_id=job.id,
            filename=job.filename,
            overall_score=result.overall_score,
            risk_level=result.risk_level,
            clause_summary=result.clause_summary,
            flags=result.flags,
            report_url=report_url,
        )
    finally:
        db.close()
