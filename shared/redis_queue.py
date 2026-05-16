import logging
import os
import socket
import threading
from typing import NamedTuple, Optional, Union, cast

import redis

from shared.settings import settings

logger = logging.getLogger(__name__)

_JOB_FIELD = b"job_id"

# Module-level redis client so JobQueue instances share one connection pool.
# Building a fresh Redis() per JobQueue allocates a new pool every call site
# (POST /jobs, /health) which is wasteful under a k8s liveness probe.
_shared_client: Optional[redis.Redis] = None
_shared_client_lock = threading.Lock()


def _get_shared_client() -> redis.Redis:
    global _shared_client
    if _shared_client is not None:
        return _shared_client
    # Double-checked: avoid two pools under concurrent first-use from the
    # FastAPI threadpool.
    with _shared_client_lock:
        if _shared_client is None:
            _shared_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
            )
        return _shared_client


class InFlightJob(NamedTuple):
    """A job claimed from the stream. Must be acked after successful processing."""

    job_id: str
    entry_id: str


class DeadLetter(NamedTuple):
    """A job that exceeded max_deliveries and was routed to the DLQ stream.

    The queue has already acked the original entry and written to the DLQ; the
    caller only needs to update DB-side bookkeeping (mark the Job failed).
    """

    job_id: str
    entry_id: str
    times_delivered: int


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

    def __init__(
        self,
        consumer_name: Optional[str] = None,
        _redis_client=None,
    ):
        self._client = _redis_client or _get_shared_client()
        self._key = settings.job_queue_key
        self._group = settings.job_queue_group
        self._consumer = consumer_name or _default_consumer_name()
        self._idle_ms = settings.job_queue_idle_ms
        self._dlq_key = settings.job_queue_dlq_key
        self._max_deliveries = settings.job_queue_max_deliveries
        self._maxlen = settings.job_queue_maxlen
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
            self._client.xadd(
                self._key,
                {"job_id": job_id},
                maxlen=self._maxlen,
                approximate=True,
            )
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

    def _times_delivered(self, entry_id: str) -> Optional[int]:
        try:
            pending = cast(
                list,
                cast(
                    object,
                    self._client.xpending_range(
                        self._key,
                        self._group,
                        min=entry_id,
                        max=entry_id,
                        count=1,
                    ),
                ),
            )
        except redis.ResponseError as exc:
            logger.debug("queue: xpending_range failed for %s: %s", entry_id, exc)
            return None
        if not pending:
            return None
        record = pending[0]
        value = record.get("times_delivered") or record.get(b"times_delivered")
        return int(value) if value is not None else None

    def _deadletter(self, entry_id: str, fields: dict) -> None:
        """Route an entry to the DLQ stream and ack the original."""
        self._client.xadd(
            self._dlq_key,
            fields,
            maxlen=self._maxlen,
            approximate=True,
        )
        self.ack(entry_id)

    def _reclaim(self) -> Optional[Union[InFlightJob, DeadLetter]]:
        """Pick up an entry idle beyond the threshold from a crashed consumer.

        If the entry has been delivered more than max_deliveries times, route it
        to the DLQ stream and return a DeadLetter so the caller can update DB
        state. This prevents a poison-pill job from looping forever.
        """
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
        entry_id_s = _decode(entry_id)
        value = fields.get(_JOB_FIELD) or fields.get("job_id")
        if value is None:
            # Malformed entry.
            self.ack(entry_id_s)
            return None

        # XAUTOCLAIM has already incremented times_delivered, so this reads the
        # post-claim value. We fail on the delivery that pushes us past the
        # ceiling rather than waiting one more round.
        delivered = self._times_delivered(entry_id_s)
        if delivered is not None and delivered > self._max_deliveries:
            logger.error(
                "queue: entry %s (job_id=%s) exceeded max_deliveries=%d "
                "(delivered=%d); routing to DLQ",
                entry_id_s,
                _decode(value),
                self._max_deliveries,
                delivered,
            )
            self._deadletter(entry_id_s, fields)
            return DeadLetter(
                job_id=_decode(value),
                entry_id=entry_id_s,
                times_delivered=delivered,
            )
        return InFlightJob(job_id=_decode(value), entry_id=entry_id_s)

    def dequeue(self, timeout: int = 5) -> Optional[Union[InFlightJob, DeadLetter]]:
        self._ensure_group()

        reclaimed = self._reclaim()
        if isinstance(reclaimed, DeadLetter):
            return reclaimed
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
