import logging
import os
import socket
from typing import NamedTuple, Optional, cast

import redis

from shared.settings import settings

logger = logging.getLogger(__name__)

_JOB_FIELD = b"job_id"


class InFlightJob(NamedTuple):
    """A job claimed from the stream. Must be acked after successful processing."""

    job_id: str
    entry_id: str


def _default_consumer_name() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


class JobQueue:
    """Redis Streams-backed queue with consumer-group semantics.

    At-least-once delivery: XREADGROUP claims an entry for this consumer, and
    the caller must ack() it on success. If the consumer crashes, XAUTOCLAIM
    reclaims the entry for another consumer after job_queue_idle_ms.
    """

    def __init__(self, consumer_name: Optional[str] = None):
        self._client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
        )
        self._key = settings.job_queue_key
        self._group = settings.job_queue_group
        self._consumer = consumer_name or _default_consumer_name()
        self._idle_ms = settings.job_queue_idle_ms
        self._autoclaim_cursor = "0-0"
        self._group_ready = False

    def _ensure_group(self) -> None:
        if self._group_ready:
            return
        try:
            self._client.xgroup_create(
                self._key, self._group, id="0-0", mkstream=True
            )
        except redis.ResponseError as exc:
            # BUSYGROUP means another process already created it. Not an error.
            if "BUSYGROUP" not in str(exc):
                raise
        self._group_ready = True

    def ping(self) -> bool:
        return cast(bool, cast(object, self._client.ping()))

    def enqueue(self, job_id: str) -> None:
        self._ensure_group()
        try:
            self._client.xadd(self._key, {"job_id": job_id})
        except redis.ConnectionError as exc:
            logger.error("queue: failed to enqueue job %s: %s", job_id, exc)
            raise

    def _parse_entry(self, raw) -> Optional[InFlightJob]:
        # Redis-py returns: [[stream_key, [(entry_id, {field: value}), ...]]]
        if not raw:
            return None
        _, entries = raw[0]
        if not entries:
            return None
        entry_id, fields = entries[0]
        value = fields.get(_JOB_FIELD) or fields.get("job_id")
        if value is None:
            return None
        return InFlightJob(job_id=_decode(value), entry_id=_decode(entry_id))

    def _reclaim(self) -> Optional[InFlightJob]:
        """Pick up an entry idle beyond the threshold from a crashed consumer."""
        try:
            raw_result = cast(
                object,
                self._client.xautoclaim(
                    self._key,
                    self._group,
                    self._consumer,
                    min_idle_time=self._idle_ms,
                    start_id=self._autoclaim_cursor,
                    count=1,
                ),
            )
        except redis.ResponseError as exc:
            logger.debug("queue: xautoclaim failed: %s", exc)
            return None
        except redis.ConnectionError as exc:
            logger.error("queue: xautoclaim connection error: %s", exc)
            raise

        # redis-py returns (next_cursor, [[entry_id, {fields}], ...], [deleted_ids])
        result = cast(tuple, raw_result)
        next_cursor = result[0]
        claimed = result[1] if len(result) > 1 else []
        self._autoclaim_cursor = _decode(next_cursor) if next_cursor else "0-0"

        if not claimed:
            return None
        entry_id, fields = claimed[0]
        value = fields.get(_JOB_FIELD) or fields.get("job_id")
        if value is None:
            # Malformed entry; ack and move on.
            self.ack(_decode(entry_id))
            return None
        return InFlightJob(job_id=_decode(value), entry_id=_decode(entry_id))

    def dequeue(self, timeout: int = 5) -> Optional[InFlightJob]:
        self._ensure_group()

        reclaimed = self._reclaim()
        if reclaimed is not None:
            logger.info(
                "queue: reclaimed stale entry %s (job_id=%s)",
                reclaimed.entry_id,
                reclaimed.job_id,
            )
            return reclaimed

        try:
            raw = cast(
                object,
                self._client.xreadgroup(
                    self._group,
                    self._consumer,
                    {self._key: ">"},
                    count=1,
                    block=timeout * 1000,
                ),
            )
        except redis.ConnectionError as exc:
            logger.error("queue: xreadgroup connection error: %s", exc)
            raise
        return self._parse_entry(raw)

    def ack(self, entry_id: str) -> None:
        try:
            self._client.xack(self._key, self._group, entry_id)
        except redis.ConnectionError as exc:
            logger.error("queue: xack connection error for %s: %s", entry_id, exc)
            raise

    def depth(self) -> int:
        """Stream entries not yet delivered to this consumer group."""
        try:
            info = cast(list, cast(object, self._client.xinfo_groups(self._key)))
        except redis.ResponseError:
            # Stream doesn't exist yet.
            return 0
        for group in info:
            name = group.get("name") or group.get(b"name")
            if _decode(name) == self._group:
                lag = group.get("lag") or group.get(b"lag") or 0
                return int(lag) if lag is not None else 0
        return 0
