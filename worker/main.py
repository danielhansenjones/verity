import logging
import sys
import time

import torch
from transformers import pipeline as hf_pipeline

from shared.minio_client import StorageClient
from shared.models import Chunk, Job, JobStage, JobStatus, init_db, get_session
from shared.redis_queue import JobQueue
from worker.processors import assembler, classifier, ingestion, scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main():
    logger.info("worker: initialising database")
    init_db()

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

    queue = JobQueue()
    storage = StorageClient()

    logger.info("worker: entering job loop")

    while True:
        try:
            job_id = queue.dequeue(timeout=5)
        except Exception:
            logger.exception("worker: failed to dequeue from Redis, retrying in 5s")
            time.sleep(5)
            continue

        if job_id is None:
            continue

        db = get_session()
        try:
            job = db.get(Job, job_id)
            if job is None:
                logger.warning("worker: job %s not found, skipping", job_id)
                continue

            job.status = JobStatus.RUNNING
            db.commit()

            try:
                if job.stage == JobStage.INGESTION:
                    ingestion.run(job, db, storage)

                if job.stage == JobStage.CLASSIFICATION:
                    classifier.run(job, db, classifier_pipeline)

                if job.stage == JobStage.SCORING:
                    scored = scorer.run(job, db, tone_pipeline)
                else:
                    scored = None

                if job.stage == JobStage.ASSEMBLY:
                    if scored is None:
                        chunks = (
                            db.query(Chunk)
                            .filter(Chunk.job_id == job.id)
                            .order_by(Chunk.index)
                            .all()
                        )
                        scored = scorer.score_chunks(chunks, tone_pipeline)
                    assembler.run(job, db, storage, scored)

            except Exception as exc:
                logger.exception("worker: job %s failed at stage %s", job_id, job.stage)
                job.error = str(exc)
                if job.retry_count < job.max_retries:
                    job.status = JobStatus.RETRYING
                    job.retry_count += 1
                    queue.enqueue(job_id)
                    logger.info(
                        "worker: job %s re-queued (attempt %d/%d)",
                        job_id,
                        job.retry_count,
                        job.max_retries,
                    )
                else:
                    job.status = JobStatus.FAILED
                    logger.error(
                        "worker: job %s exhausted retries, marked failed", job_id
                    )
                db.commit()
        finally:
            db.close()


if __name__ == "__main__":
    main()
