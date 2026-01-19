import logging
import uuid
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from shared.minio_client import StorageClient
from shared.models import Job, JobStage, JobStatus, RiskResult

logger = logging.getLogger(__name__)

_REPORT_PREFIX = "contracts/reports"


def _risk_level(score: float) -> str:
    if score < 35:
        return "low"
    if score < 65:
        return "medium"
    return "high"


def run(
    job: Job, db: Session, storage: StorageClient, scored_chunks: list[dict]
) -> None:
    logger.info("assembler: job=%s chunks=%d", job.id, len(scored_chunks))

    if scored_chunks:
        overall = sum(c["chunk_score"] for c in scored_chunks) / len(scored_chunks)
    else:
        overall = 0.0

    overall_int = int(round(overall))
    level = _risk_level(overall)

    clause_summary = dict(
        Counter(c["clause_type"] for c in scored_chunks if c["clause_type"])
    )

    flags = []
    for chunk in scored_chunks:
        for hit in chunk["flags"]:
            text = chunk["text"]
            lower = text.lower()
            pos = lower.find(hit["pattern"])
            start = max(0, pos - 40)
            end = min(len(text), pos + len(hit["pattern"]) + 40)
            excerpt = "..." + text[start:end] + "..."

            flags.append(
                {
                    "chunk_index": chunk["index"],
                    "clause_type": chunk["clause_type"],
                    "matched_pattern": hit["pattern"],
                    "reason": hit["reason"],
                    "severity": hit["severity"],
                    "excerpt": excerpt,
                }
            )

    report = {
        "job_id": job.id,
        "filename": job.filename,
        "overall_score": overall_int,
        "risk_level": level,
        "clause_summary": clause_summary,
        "flags": flags,
        "chunks": [
            {
                "index": c["index"],
                "clause_type": c["clause_type"],
                "confidence": c["confidence"],
                "score": int(round(c["chunk_score"])),
                "text": c["text"],
            }
            for c in scored_chunks
        ],
    }

    report_key = f"{_REPORT_PREFIX}/{job.id}.json"
    storage.upload_json(report_key, report)
    logger.info("assembler: report uploaded → %s", report_key)

    result = RiskResult(
        id=str(uuid.uuid4()),
        job_id=job.id,
        overall_score=overall_int,
        risk_level=level,
        clause_summary=clause_summary,
        flags=flags,
        report_key=report_key,
        created_at=datetime.now(timezone.utc),
    )
    db.add(result)

    job.stage = JobStage.DONE
    job.status = JobStatus.COMPLETED
    job.updated_at = datetime.now(timezone.utc)
    db.commit()
    logger.info("assembler: done, job=%s status=completed", job.id)
