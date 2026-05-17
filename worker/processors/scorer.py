import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.models import Chunk, Job, JobStage
from worker.processors.clause_labels import CLAUSE_WEIGHTS
from worker.processors.risk_rules import default_matcher

logger = logging.getLogger(__name__)

_SEVERITY_SCORES = {"high": 1.0, "medium": 0.5, "low": 0.25}
_BATCH_SIZE = 8


def _apply_risk_patterns(text: str) -> list[dict]:
    """Return risk hits with rule id, severity, and the exact matched span."""
    return [
        {
            "id": hit.id,
            "severity": hit.severity,
            "reason": hit.reason,
            "matched_text": hit.matched_text,
            "start": hit.start,
            "end": hit.end,
        }
        for hit in default_matcher().match(text)
    ]


def _to_tone_score(result: dict) -> float:
    label = result["label"]
    score = result["score"]
    return score if label == "NEGATIVE" else 1.0 - score


def score_chunks(chunks: list[Chunk], tone_pipeline) -> list[dict]:
    # truncation=True: HF tokenizer truncates at the model's 512-token limit.
    # Slicing chunk.text[:512] was characters, not tokens.
    texts = [c.text for c in chunks]
    tone_results = tone_pipeline(texts, batch_size=_BATCH_SIZE, truncation=True)
    tone_scores = [_to_tone_score(r) for r in tone_results]

    scored = []
    for chunk, ts in zip(chunks, tone_scores):
        rule_text = chunk.extracted_span if chunk.extracted_span else chunk.text
        hits = _apply_risk_patterns(rule_text)

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
                "extracted_span": chunk.extracted_span,
                "extracted_span_category": chunk.extracted_span_category,
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
