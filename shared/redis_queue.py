import logging
from typing import Optional, cast

import redis

from shared.settings import settings

logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self):
        self._client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
        )
        self._key = settings.job_queue_key

    def ping(self) -> bool:
        return cast(bool, cast(object, self._client.ping()))

    def enqueue(self, job_id: str) -> None:
        try:
            self._client.lpush(self._key, job_id)
        except redis.ConnectionError as exc:
            logger.error("queue: failed to enqueue job %s: %s", job_id, exc)
            raise

    def dequeue(self, timeout: int = 5) -> Optional[str]:
        try:
            raw = cast(
                Optional[tuple[bytes, bytes]],
                cast(object, self._client.brpop(self._key, timeout=timeout)),
            )
        except redis.ConnectionError as exc:
            logger.error("queue: failed to dequeue: %s", exc)
            raise
        if raw is None:
            return None
        _, job_id = raw
        return job_id.decode() if isinstance(job_id, bytes) else str(job_id)

    def depth(self) -> int:
        return cast(int, cast(object, self._client.llen(self._key)))
