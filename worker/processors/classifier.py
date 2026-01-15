import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.models import Chunk, Job, JobStage

logger = logging.getLogger(__name__)

CLAUSE_LABELS = [
    "indemnification",
    "termination",
    "liability limitation",
    "governing law",
    "payment terms",
    "intellectual property assignment",
    "confidentiality",
    "dispute resolution",
    "warranty",
    "force majeure",
]

_CONFIDENCE_THRESHOLD = 0.4
_BATCH_SIZE = 8


def run(job: Job, db: Session, classifier_pipeline) -> None:
    chunks: list[Chunk] = list(
        db.scalars(
            select(Chunk).where(Chunk.job_id == job.id).order_by(Chunk.index)
        ).all()
    )
    logger.info("classifier: job=%s chunks=%d", job.id, len(chunks))

    for batch_start in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[batch_start : batch_start + _BATCH_SIZE]
        texts = [c.text for c in batch]

        try:
            results = classifier_pipeline(
                texts, candidate_labels=CLAUSE_LABELS, batch_size=_BATCH_SIZE
            )
        except Exception as exc:
            raise RuntimeError(
                f"classifier inference failed on batch starting at index {batch_start}"
            ) from exc

        # pipeline returns a single dict for one input; list for multiple
        if isinstance(results, dict):
            results = [results]

        for chunk, result in zip(batch, results):
            top_label = result["labels"][0]
            top_score = result["scores"][0]

            chunk.clause_type = (
                top_label if top_score >= _CONFIDENCE_THRESHOLD else "general"
            )
            chunk.confidence = top_score

        db.commit()
        logger.info("classifier: processed batch starting at %d", batch_start)

    job.stage = JobStage.SCORING
    job.updated_at = datetime.now(timezone.utc)
    db.commit()
    logger.info("classifier: done, stage → scoring")
