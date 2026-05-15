"""Schema bootstrap. Run once before the API and worker start.

Idempotent - re-running is a no-op for existing tables and indexes. Invoked by
the docker-compose `init-db` service and by CI workflows; should never be
called from request-handling code.
"""
import logging

from shared.logging_config import configure_logging
from shared.models import init_db


def main() -> None:
    configure_logging()
    logger = logging.getLogger("init_db")
    logger.info("init_db: bootstrapping schema")
    init_db()
    logger.info("init_db: done")


if __name__ == "__main__":
    main()
