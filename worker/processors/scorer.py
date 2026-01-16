import logging
import re
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.models import Chunk, Job, JobStage

logger = logging.getLogger(__name__)

RISK_PATTERNS = [
    ("sole discretion", "high", "Unilateral decision right"),
    ("unlimited liability", "high", "Uncapped liability exposure"),
    ("perpetual and irrevocable", "high", "Cannot be terminated or reversed"),
    ("indemnify and hold harmless", "high", "Broad indemnification obligation"),
    ("without cause", "medium", "Termination without cause permitted"),
    ("automatic renewal", "medium", "Contract auto-renews without action"),
    ("liquidated damages", "medium", "Pre-set damages clause"),
    ("best efforts", "low", "Ambiguous obligation standard"),
    ("reasonable notice", "low", "Undefined notice period"),
]

CLAUSE_WEIGHTS = {
    "indemnification": 1.0,
    "liability limitation": 0.9,
    "intellectual property assignment": 0.8,
    "termination": 0.7,
    "dispute resolution": 0.6,
    "confidentiality": 0.5,
    "payment terms": 0.5,
    "warranty": 0.4,
    "force majeure": 0.3,
    "governing law": 0.2,
    "general": 0.1,
}

_SEVERITY_SCORES = {"high": 1.0, "medium": 0.5, "low": 0.25}
_BATCH_SIZE = 8


def _apply_risk_patterns(text: str) -> list[dict]:
    lower = text.lower()
    hits = []
    for pattern, severity, reason in RISK_PATTERNS:
        if pattern in lower:
            hits.append({"pattern": pattern, "severity": severity, "reason": reason})
    return hits


def _to_tone_score(result: dict) -> float:
    label = result["label"]
    score = result["score"]
    return score if label == "NEGATIVE" else 1.0 - score


def score_chunks(chunks: list[Chunk], tone_pipeline) -> list[dict]:
    """
    Score each chunk and return a list of per-chunk score dicts:
      { chunk_id, index, clause_type, tone_score, flags, chunk_score }
    """
    texts = [c.text[:512] for c in chunks]  # truncate to avoid token overflow
    tone_results = tone_pipeline(texts, batch_size=_BATCH_SIZE)
    tone_scores = [_to_tone_score(r) for r in tone_results]

    scored = []
    for chunk, ts in zip(chunks, tone_scores):
        hits = _apply_risk_patterns(chunk.text)

        max_flag_score = max(
            (_SEVERITY_SCORES[h["severity"]] for h in hits), default=0.0
        )
        type_weight = CLAUSE_WEIGHTS.get(chunk.clause_type or "general", 0.1)

        tone_contrib = ts * 0.3
        flag_contrib = max_flag_score * 0.4
        type_contrib = type_weight * 0.3

        chunk_score = (tone_contrib + flag_contrib + type_contrib) * 100

        scored.append(
            {
                "chunk_id": chunk.id,
                "index": chunk.index,
                "clause_type": chunk.clause_type,
                "confidence": chunk.confidence,
                "tone_score": ts,
                "flags": hits,
                "chunk_score": chunk_score,
                "text": chunk.text,
            }
        )

    return scored


def run(job: Job, db: Session, tone_pipeline) -> list[dict]:
    chunks: list[Chunk] = list(
        db.scalars(
            select(Chunk).where(Chunk.job_id == job.id).order_by(Chunk.index)
        ).all()
    )
    logger.info("scorer: job=%s chunks=%d", job.id, len(chunks))

    scored = score_chunks(chunks, tone_pipeline)

    job.stage = JobStage.ASSEMBLY
    job.updated_at = datetime.now(timezone.utc)
    db.commit()
    logger.info("scorer: done, stage → assembly")

    return scored
