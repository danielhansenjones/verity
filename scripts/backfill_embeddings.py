"""Idempotent backfill for chunks created before the embedding pipeline existed."""
import argparse
import logging

from shared.logging_config import configure_logging
from shared.models import Chunk, get_session
from worker.processors.embeddings import EmbeddingModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--job-id", default=None, help="restrict to one job id")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count chunks pending embedding and exit without writing",
    )
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger("backfill_embeddings")

    with get_session() as db:
        def pending_query():
            q = db.query(Chunk).filter(Chunk.embedding.is_(None))
            if args.job_id:
                q = q.filter(Chunk.job_id == args.job_id)
            return q

        total = pending_query().count()
        logger.info("backfill: %d chunks pending", total)

        if args.dry_run or total == 0:
            return

        logger.info("backfill: loading embedding model")
        model = EmbeddingModel(device=args.device)

        processed = 0
        while True:
            batch = (
                pending_query()
                .order_by(Chunk.created_at)
                .limit(args.batch_size)
                .all()
            )
            if not batch:
                break

            vectors = model.embed_documents([c.text for c in batch])
            for chunk, vec in zip(batch, vectors):
                chunk.embedding = vec
            db.commit()

            processed += len(batch)
            logger.info("backfill: %d / %d", processed, total)

        logger.info("backfill: done, %d chunks embedded", processed)


if __name__ == "__main__":
    main()
