import hashlib
import logging
import time
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
from fastapi.responses import JSONResponse, Response as FastAPIResponse
from minio.error import S3Error
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from api.auth import require_api_key
from api.llm import AnswerResponse, ask as llm_ask, grounding_error
from api.rag import embed_query, preload_embedding_model, retrieve
from api.rate_limit import limiter, read_limit, submit_limit
from shared.logging_config import configure_logging
from shared.metrics import (
    jobs_submitted_total,
    rag_generation_latency_seconds,
    rag_query_log_failures_total,
    rag_questions_total,
    rag_retrieval_latency_seconds,
    rag_tokens_total,
)
from shared.minio_client import StorageClient
from shared.models import (
    Job,
    JobDedup,
    JobStage,
    JobStatus,
    RagQuery,
    RiskResult,
    get_session,
)
from shared.redis_queue import JobQueue
from shared.settings import settings

configure_logging()
logger = logging.getLogger(__name__)

_RAW_PREFIX = "contracts/raw"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Schema bootstrap is handled out-of-band by scripts/init_db.py (run by
    # the docker-compose init-db service or a CI step). The API process
    # assumes the schema exists by the time the first request lands.
    app.state.storage = StorageClient()
    app.state.queue = JobQueue()
    # Pre-load the embedding model so the first /ask request does not block
    # on a 5-10 s sentence-transformer initialisation. Skipped when /ask is
    # going to return 503 anyway (ANTHROPIC_API_KEY unset) so dev startup
    # without the RAG backend stays fast.
    if settings.anthropic_api_key:
        preload_embedding_model()
    else:
        logger.info(
            "api: ANTHROPIC_API_KEY unset; skipping embedding model preload"
        )
    yield


app = FastAPI(title="Contract Risk Pipeline", lifespan=lifespan)
app.state.limiter = limiter


def get_storage(request: Request) -> StorageClient:
    return request.app.state.storage


def get_queue(request: Request) -> JobQueue:
    return request.app.state.queue


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


class AskRequest(BaseModel):
    question: str
    top_k: int = 8


class RagQueryListItem(BaseModel):
    id: str
    job_id: str
    question: str
    outcome: str
    retrieval_ms: Optional[int]
    generation_ms: Optional[int]
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    grounding_error: Optional[str]
    error: Optional[str]
    created_at: datetime


@app.get("/metrics", include_in_schema=False)
def metrics():
    return FastAPIResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health(request: Request):
    try:
        with get_session() as db:
            db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    try:
        request.app.state.queue.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    healthy = db_ok and redis_ok
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "degraded",
            "services": {"postgres": db_ok, "redis": redis_ok},
        },
    )


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
def submit_job(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    storage: StorageClient = Depends(get_storage),
    queue: JobQueue = Depends(get_queue),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted (Content-Type must be application/pdf)",
        )

    pdf_bytes = file.file.read()

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
    with get_session() as db:
        existing = db.query(JobDedup).filter(JobDedup.key == dedup_key).first()
        if existing is not None:
            job = db.get(Job, existing.job_id)
            if job is not None:
                jobs_submitted_total.labels(outcome="replayed").inc()
                response.status_code = 200
                response.headers["Idempotent-Replay"] = "true"
                return JobCreatedResponse(
                    job_id=job.id, status=job.status, filename=job.filename
                )

    job_id = str(uuid.uuid4())
    object_key = f"{_RAW_PREFIX}/{job_id}.pdf"

    # Commit before upload so a commit failure (or dedup loss) doesn't leave
    # an orphan blob with no DB row pointing at it.
    with get_session() as db:
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
            db.rollback()
            existing = db.query(JobDedup).filter(JobDedup.key == dedup_key).first()
            if existing is not None:
                winner = db.get(Job, existing.job_id)
                if winner is not None:
                    logger.info(
                        "api: idempotency race; returning winner job_id=%s",
                        winner.id,
                    )
                    jobs_submitted_total.labels(outcome="replayed").inc()
                    response.status_code = 200
                    response.headers["Idempotent-Replay"] = "true"
                    return JobCreatedResponse(
                        job_id=winner.id,
                        status=winner.status,
                        filename=winner.filename,
                    )
            raise

    try:
        storage.upload_bytes(object_key, pdf_bytes, content_type="application/pdf")
    except S3Error as exc:
        # Upload failed: blob never wrote. Roll back the DB rows so a client
        # retry isn't a confused dedup replay against a job that points
        # nowhere.
        logger.error("api: storage upload failed for job %s: %s", job_id, exc)
        _delete_pending(job_id, dedup_key)
        raise HTTPException(status_code=503, detail="Storage service unavailable")

    try:
        queue.enqueue(job_id)
    except redis.ConnectionError as exc:
        # Enqueue failed after both DB and blob committed: roll back both so
        # the job doesn't sit forever in QUEUED with no worker entry.
        logger.error("api: failed to enqueue job %s: %s", job_id, exc)
        try:
            storage.delete_object(object_key)
        except S3Error:
            logger.warning(
                "api: orphan blob cleanup failed for job %s; key=%s",
                job_id,
                object_key,
            )
        _delete_pending(job_id, dedup_key)
        raise HTTPException(status_code=503, detail="Queue service unavailable")

    jobs_submitted_total.labels(outcome="created").inc()
    logger.info("api: job submitted job_id=%s filename=%s", job_id, file.filename)
    return JobCreatedResponse(
        job_id=job_id, status=JobStatus.QUEUED, filename=file.filename
    )


def _delete_pending(job_id: str, dedup_key: str) -> None:
    with get_session() as db:
        db.query(JobDedup).filter(JobDedup.key == dedup_key).delete()
        db.query(Job).filter(Job.id == job_id).delete()
        db.commit()


@app.get(
    "/jobs",
    response_model=list[JobListItem],
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(read_limit)
def list_jobs(request: Request, status: Optional[str] = Query(default=None)):
    with get_session() as db:
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


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(read_limit)
def get_job(request: Request, job_id: str):
    with get_session() as db:
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


def _log_rag_query(
    db,
    *,
    job_id: str,
    question: str,
    top_k: int,
    outcome: str,
    chunk_ids: list[str],
    response: Optional[AnswerResponse] = None,
    usage: Optional[dict] = None,
    grounding_err: Optional[str] = None,
    error: Optional[str] = None,
    retrieval_ms: Optional[int] = None,
    generation_ms: Optional[int] = None,
) -> Optional[str]:
    # Best-effort: a failure to persist the audit row must never turn a good
    # answer into an error. Swallow but log loudly, then let the request finish.
    # Returns the persisted RagQuery.id (the X-Trace-Id surfaced to the caller)
    # or None when the write failed.
    usage = usage or {}
    rag_query_id = str(uuid.uuid4())
    try:
        db.add(
            RagQuery(
                id=rag_query_id,
                job_id=job_id,
                question=question,
                top_k=top_k,
                model=settings.anthropic_model,
                outcome=outcome,
                retrieved_chunk_ids=chunk_ids,
                answer=response.answer if response else None,
                refusal_reason=response.refusal_reason if response else None,
                citations=(
                    [c.model_dump() for c in response.citations]
                    if response
                    else None
                ),
                grounding_error=grounding_err,
                error=error,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cache_read_tokens=usage.get("cache_read_input_tokens"),
                cache_creation_tokens=usage.get("cache_creation_input_tokens"),
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
            )
        )
        db.commit()
        return rag_query_id
    except Exception:
        db.rollback()
        rag_query_log_failures_total.inc()
        logger.exception("api: failed to persist rag_query job_id=%s", job_id)
        return None


@app.post(
    "/jobs/{job_id}/ask",
    response_model=AnswerResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(read_limit)
def ask(
    request: Request,
    http_response: Response,
    job_id: str,
    body: AskRequest,
):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    if len(body.question) > 2000:
        raise HTTPException(
            status_code=400, detail="question must be <= 2000 characters"
        )

    if not settings.anthropic_api_key:
        rag_questions_total.labels(outcome="error").inc()
        raise HTTPException(
            status_code=503,
            detail="Generation service unavailable: ANTHROPIC_API_KEY not configured",
        )

    with get_session() as db:
        job = db.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Job is not completed yet (status={job.status},"
                    f" stage={job.stage})"
                ),
            )

        t_retrieval = time.perf_counter()
        with rag_retrieval_latency_seconds.time():
            query_vec = embed_query(body.question)
            chunks = retrieve(db, job_id, query_vec, k=body.top_k)
        retrieval_ms = int((time.perf_counter() - t_retrieval) * 1000)

        chunk_ids = [c.id for c in chunks]

        if not chunks:
            rag_questions_total.labels(outcome="refused").inc()
            logger.info(
                "api: rag refused job_id=%s reason=no_embedded_chunks", job_id
            )
            refusal = AnswerResponse(
                answer="",
                citations=[],
                refusal_reason=(
                    "No embedded chunks available for this job."
                    " Run scripts/backfill_embeddings.py."
                ),
            )
            rag_query_id = _log_rag_query(
                db,
                job_id=job_id,
                question=body.question,
                top_k=body.top_k,
                outcome="refused",
                chunk_ids=chunk_ids,
                response=refusal,
                retrieval_ms=retrieval_ms,
            )
            if rag_query_id:
                http_response.headers["X-Trace-Id"] = rag_query_id
            return refusal

        try:
            t_generation = time.perf_counter()
            with rag_generation_latency_seconds.time():
                response, usage = llm_ask(body.question, chunks)
            generation_ms = int((time.perf_counter() - t_generation) * 1000)
        except Exception as exc:
            rag_questions_total.labels(outcome="error").inc()
            logger.exception("api: rag generation failed job_id=%s", job_id)
            rag_query_id = _log_rag_query(
                db,
                job_id=job_id,
                question=body.question,
                top_k=body.top_k,
                outcome="error",
                chunk_ids=chunk_ids,
                error=str(exc),
                retrieval_ms=retrieval_ms,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Generation failed: {exc}",
                headers={"X-Trace-Id": rag_query_id} if rag_query_id else None,
            )

        rag_tokens_total.labels(direction="in").inc(usage["input_tokens"])
        rag_tokens_total.labels(direction="out").inc(usage["output_tokens"])
        if usage.get("cache_read_input_tokens"):
            rag_tokens_total.labels(direction="cache_read").inc(
                usage["cache_read_input_tokens"]
            )
        if usage.get("cache_creation_input_tokens"):
            rag_tokens_total.labels(direction="cache_creation").inc(
                usage["cache_creation_input_tokens"]
            )

        err = grounding_error(response, chunks)
        if err:
            rag_questions_total.labels(outcome="error").inc()
            logger.warning(
                "api: rag grounding failed job_id=%s err=%s", job_id, err
            )
            rag_query_id = _log_rag_query(
                db,
                job_id=job_id,
                question=body.question,
                top_k=body.top_k,
                outcome="error",
                chunk_ids=chunk_ids,
                response=response,
                usage=usage,
                grounding_err=err,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Generation produced ungrounded output: {err}",
                headers={"X-Trace-Id": rag_query_id} if rag_query_id else None,
            )

        outcome = "refused" if response.refusal_reason else "answered"
        rag_questions_total.labels(outcome=outcome).inc()
        logger.info(
            "api: rag answer job_id=%s outcome=%s tokens_in=%d tokens_out=%d"
            " citations=%d",
            job_id,
            outcome,
            usage["input_tokens"],
            usage["output_tokens"],
            len(response.citations),
        )
        rag_query_id = _log_rag_query(
            db,
            job_id=job_id,
            question=body.question,
            top_k=body.top_k,
            outcome=outcome,
            chunk_ids=chunk_ids,
            response=response,
            usage=usage,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )
        if rag_query_id:
            http_response.headers["X-Trace-Id"] = rag_query_id
        return response


_RAG_OUTCOMES = {"answered", "refused", "error"}


@app.get(
    "/admin/rag_queries",
    response_model=list[RagQueryListItem],
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(read_limit)
def list_rag_queries(
    request: Request,
    outcome: Optional[str] = Query(default=None),
    job_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50),
    before: Optional[str] = Query(default=None),
):
    if outcome is not None and outcome not in _RAG_OUTCOMES:
        raise HTTPException(status_code=400, detail=f"Unknown outcome: {outcome}")

    limit = max(1, min(limit, 200))

    before_dt: Optional[datetime] = None
    if before is not None:
        try:
            before_dt = datetime.fromisoformat(before)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="before must be an ISO 8601 timestamp"
            )

    with get_session() as db:
        q = db.query(RagQuery)
        if outcome is not None:
            q = q.filter(RagQuery.outcome == outcome)
        if job_id is not None:
            q = q.filter(RagQuery.job_id == job_id)
        if before_dt is not None:
            q = q.filter(RagQuery.created_at < before_dt)
        rows = q.order_by(RagQuery.created_at.desc()).limit(limit).all()
        return [
            RagQueryListItem(
                id=r.id,
                job_id=r.job_id,
                question=r.question[:200],
                outcome=r.outcome,
                retrieval_ms=r.retrieval_ms,
                generation_ms=r.generation_ms,
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                grounding_error=r.grounding_error,
                error=r.error,
                created_at=r.created_at,
            )
            for r in rows
        ]


@app.get(
    "/jobs/{job_id}/report",
    response_model=ReportResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(read_limit)
def get_report(
    request: Request,
    job_id: str,
    storage: StorageClient = Depends(get_storage),
):
    with get_session() as db:
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
        if result.report_key is None:
            # Partial assembler write: DB row committed before MinIO upload.
            raise HTTPException(
                status_code=404,
                detail="Report blob not available (partial write)",
            )

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
