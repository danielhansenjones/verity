import concurrent.futures
import json
import logging
import os
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from shared.models import Chunk, Job, JobStage
from shared.settings import settings
from worker.processors.clause_labels import CLAUSE_LABELS

logger = logging.getLogger(__name__)

# More permissive than the val-tuned category-presence threshold (0.95).
# Label assignment tolerates false positives: a wrong label is penalized by the
# rule layer and scoring weights; a miss just emits "general".
_CONFIDENCE_THRESHOLD = 0.5
_BATCH_SIZE = 8

_MAPPING_PATH = os.path.join(
    os.path.dirname(__file__), "category_mapping.json"
)

with open(_MAPPING_PATH) as _f:
    _raw = json.load(_f)
    CATEGORY_MAPPING: dict[str, list[str]] = {
        k: v for k, v in _raw.items() if not k.startswith("_")
    }


def run(
    job: Job,
    db: Session,
    classifier_pipeline,
    span_extractor=None,
) -> None:
    chunks: list[Chunk] = list(
        db.scalars(
            select(Chunk).where(Chunk.job_id == job.id).order_by(Chunk.index)
        ).all()
    )
    logger.info("classifier: job=%s chunks=%d", job.id, len(chunks))

    tier1_threshold = settings.span_extractor_tier1_confidence_threshold
    span_executor = (
        concurrent.futures.ThreadPoolExecutor(max_workers=1)
        if span_extractor is not None
        else None
    )

    try:
        for batch_start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _BATCH_SIZE]
            texts = [c.text for c in batch]

            try:
                results = classifier_pipeline(
                    texts,
                    candidate_labels=CLAUSE_LABELS,
                    batch_size=_BATCH_SIZE,
                    multi_label=True,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"classifier inference failed on batch"
                    f" starting at index {batch_start}"
                ) from exc

            if isinstance(results, dict):
                results = [results]

            for chunk, result in zip(batch, results):
                top_label = result["labels"][0]
                top_score = result["scores"][0]

                chunk.clause_type = (
                    top_label if top_score >= _CONFIDENCE_THRESHOLD else "general"
                )
                chunk.confidence = top_score

                if (
                    span_executor is not None
                    and top_score >= tier1_threshold
                    and top_label in CATEGORY_MAPPING
                ):
                    cuad_categories = CATEGORY_MAPPING[top_label]
                    try:
                        future = span_executor.submit(
                            span_extractor.extract, chunk.text, cuad_categories
                        )
                        spans = future.result(
                            timeout=settings.span_extractor_timeout_s
                        )
                        best_span = None
                        best_cat = None
                        for cat, span in spans.items():
                            if span is not None and (
                                best_span is None
                                or span["score"] > best_span["score"]
                            ):
                                best_span = span
                                best_cat = cat
                        if best_span is not None:
                            chunk.extracted_span = best_span["text"]
                            chunk.extracted_span_category = best_cat
                            logger.debug(
                                "classifier: span extracted for chunk=%s cat=%s",
                                chunk.id,
                                best_cat,
                            )
                    except concurrent.futures.TimeoutError:
                        logger.warning(
                            "classifier: span extraction timed out "
                            "for chunk=%s after %.1fs",
                            chunk.id,
                            settings.span_extractor_timeout_s,
                        )
                    except Exception:
                        logger.warning(
                            "classifier: span extraction failed for chunk=%s",
                            chunk.id,
                            exc_info=True,
                        )

            db.commit()
            logger.info("classifier: processed batch starting at %d", batch_start)
    finally:
        if span_executor is not None:
            span_executor.shutdown(wait=False)

    job.stage = JobStage.SCORING
    job.updated_at = datetime.now(timezone.utc)
    db.commit()
    logger.info("classifier: done, stage → scoring")
