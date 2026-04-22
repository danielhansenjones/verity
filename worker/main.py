import logging
import time

import torch
from prometheus_client import start_http_server
from transformers import pipeline as hf_pipeline

from shared.logging_config import configure_logging
from shared.metrics import (
    job_stage_duration_seconds,
    job_stage_errors_total,
    jobs_completed_total,
    queue_depth,
)
from shared.minio_client import StorageClient
from shared.models import Chunk, Job, JobStage, JobStatus, init_db, get_session
from shared.redis_queue import DeadLetter, JobQueue
from shared.settings import settings
from worker.processors import assembler, classifier, ingestion, scorer
from worker.processors.span_extractor import SpanExtractor

configure_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info("worker: initialising database")
    init_db()

    start_http_server(settings.worker_metrics_port)
    logger.info("worker: metrics server on port %d", settings.worker_metrics_port)

    device = 0 if torch.cuda.is_available() else -1
    logger.info("worker: using device=%s (%s)", device, "GPU" if device == 0 else "CPU")

    logger.info("worker: loading classifier model (facebook/bart-large-mnli)")
    classifier_pipeline = hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )

    logger.info(
        "worker: loading tone model (distilbert-base-uncased-finetuned-sst-2-english)"
    )
    tone_pipeline = hf_pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )

    span_extractor = None
    if settings.span_extractor_enabled:
        if not settings.span_extractor_model_path:
            logger.warning(
                "worker: SPAN_EXTRACTOR_ENABLED=true but SPAN_EXTRACTOR_MODEL_PATH"
                " is not set; cascade disabled"
            )
        else:
            logger.info(
                "worker: loading span extractor from %s",
                settings.span_extractor_model_path,
            )
            span_extractor = SpanExtractor(
                model_path=settings.span_extractor_model_path,
                device="cuda" if device == 0 else "cpu",
            )
            logger.info("worker: span extractor loaded")
    else:
        logger.info("worker: span extractor disabled (SPAN_EXTRACTOR_ENABLED=false)")

    queue = JobQueue()
    storage = StorageClient()

    logger.info("worker: entering job loop")

    while True:
        try:
            queue_depth.set(queue.depth())
        except Exception:
            # Depth sampling is best-effort; do not block job processing on it.
            logger.debug("worker: queue depth sample failed", exc_info=True)

        try:
            inflight = queue.dequeue(timeout=5)
        except Exception:
            logger.exception("worker: failed to dequeue from Redis, retrying in 5s")
            time.sleep(5)
            continue

        if inflight is None:
            continue

        if isinstance(inflight, DeadLetter):
            # Queue has already acked the entry and written to the DLQ stream;
            # we only need to update DB-side bookkeeping.
            logger.error(
                "worker: job %s dead-lettered after %d deliveries",
                inflight.job_id,
                inflight.times_delivered,
            )
            db = get_session()
            try:
                job = db.get(Job, inflight.job_id)
                if job is not None and job.status != JobStatus.COMPLETED:
                    job.status = JobStatus.FAILED
                    job.error = (
                        f"exceeded max deliveries ({inflight.times_delivered})"
                    )
                    db.commit()
                    jobs_completed_total.labels(status="failed").inc()
            finally:
                db.close()
            continue

        job_id = inflight.job_id
        entry_id = inflight.entry_id
        log = logging.LoggerAdapter(logger, {"job_id": job_id, "stage": None})
        # Track whether we can safely ack this entry; set False if the crash
        # path should leave it in PEL for another consumer to reclaim.
        should_ack = True

        db = get_session()
        try:
            job = db.get(Job, job_id)
            if job is None:
                log.warning("worker: job not found, skipping")
                continue

            log.extra["stage"] = str(job.stage)

            # Idempotency guard: a prior attempt may have committed COMPLETED
            # before its ack reached Redis, or the entry may have been reclaimed
            # after completion. Treat as no-op success rather than re-running
            # stages (which would duplicate assembly output / double-count).
            if job.status == JobStatus.COMPLETED or job.stage == JobStage.DONE:
                log.info(
                    "worker: job already completed (status=%s); acking",
                    job.status,
                )
                continue

            job.status = JobStatus.RUNNING
            db.commit()

            try:
                if job.stage == JobStage.INGESTION:
                    log.extra["stage"] = "ingestion"
                    with job_stage_duration_seconds.labels(stage="ingestion").time():
                        ingestion.run(job, db, storage)

                if job.stage == JobStage.CLASSIFICATION:
                    log.extra["stage"] = "classification"
                    with job_stage_duration_seconds.labels(
                        stage="classification"
                    ).time():
                        classifier.run(
                            job, db, classifier_pipeline, span_extractor
                        )

                if job.stage == JobStage.SCORING:
                    log.extra["stage"] = "scoring"
                    with job_stage_duration_seconds.labels(stage="scoring").time():
                        scored = scorer.run(job, db, tone_pipeline)
                else:
                    scored = None

                if job.stage == JobStage.ASSEMBLY:
                    log.extra["stage"] = "assembly"
                    with job_stage_duration_seconds.labels(stage="assembly").time():
                        if scored is None:
                            chunks = (
                                db.query(Chunk)
                                .filter(Chunk.job_id == job.id)
                                .order_by(Chunk.index)
                                .all()
                            )
                            scored = scorer.score_chunks(chunks, tone_pipeline)
                        assembler.run(job, db, storage, scored)

                if job.status == JobStatus.COMPLETED:
                    log.extra["stage"] = str(job.stage)
                    jobs_completed_total.labels(status="completed").inc()

            except Exception as exc:
                job_stage_errors_total.labels(stage=str(job.stage)).inc()
                log.exception("worker: job failed")
                job.error = str(exc)
                if job.retry_count < job.max_retries:
                    job.status = JobStatus.RETRYING
                    job.retry_count += 1
                    # Ack the current entry and enqueue a fresh one; the new
                    # entry resets the idle timer, so transient reclaim storms
                    # cannot double-process retries.
                    queue.enqueue(job_id)
                    log.info(
                        "worker: job re-queued (attempt %d/%d)",
                        job.retry_count,
                        job.max_retries,
                    )
                else:
                    job.status = JobStatus.FAILED
                    jobs_completed_total.labels(status="failed").inc()
                    log.error("worker: job exhausted retries, marked failed")
                db.commit()
        except Exception:
            # Unexpected error outside the stage handlers (db commit, etc).
            # Leave the entry unacked so XAUTOCLAIM can hand it off.
            should_ack = False
            log.exception("worker: unexpected error handling job")
        finally:
            db.close()
            if should_ack:
                try:
                    queue.ack(entry_id)
                except Exception:
                    log.exception(
                        "worker: failed to ack entry %s; entry will be reclaimed",
                        entry_id,
                    )


if __name__ == "__main__":
    main()
