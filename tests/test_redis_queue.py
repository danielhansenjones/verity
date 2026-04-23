"""Unit tests for the Redis Streams-backed JobQueue wrapper.

These tests stub the redis client directly rather than going through conftest's
module-level redis stub, so they exercise the XADD/XREADGROUP/XACK flow.
"""

from unittest.mock import MagicMock, patch

import pytest

from shared.redis_queue import DeadLetter, InFlightJob, JobQueue
from shared.settings import settings


@pytest.fixture
def queue_with_mock_client():
    with patch("shared.redis_queue.redis.Redis") as redis_cls:
        client = MagicMock()
        redis_cls.return_value = client
        q = JobQueue(consumer_name="test-consumer")
        # No stale entries by default.
        client.xautoclaim.return_value = ("0-0", [], [])
        yield q, client


def test_enqueue_calls_xadd(queue_with_mock_client):
    q, client = queue_with_mock_client
    q.enqueue("job-123")

    client.xadd.assert_called_once()
    args, _ = client.xadd.call_args
    assert args[0] == q._key
    assert args[1] == {"job_id": "job-123"}


def test_dequeue_parses_xreadgroup_result(queue_with_mock_client):
    q, client = queue_with_mock_client
    client.xreadgroup.return_value = [
        [
            b"contract_jobs",
            [(b"1700000000000-0", {b"job_id": b"job-abc"})],
        ]
    ]

    result = q.dequeue(timeout=1)

    assert result == InFlightJob(job_id="job-abc", entry_id="1700000000000-0")
    client.xreadgroup.assert_called_once()


def test_dequeue_returns_none_when_stream_empty(queue_with_mock_client):
    q, client = queue_with_mock_client
    client.xreadgroup.return_value = []

    assert q.dequeue(timeout=1) is None


def test_dequeue_prefers_reclaimed_entry(queue_with_mock_client):
    q, client = queue_with_mock_client
    client.xautoclaim.return_value = (
        b"0-0",
        [(b"1700000000000-0", {b"job_id": b"stale-job"})],
        [],
    )

    result = q.dequeue(timeout=1)

    assert result == InFlightJob(job_id="stale-job", entry_id="1700000000000-0")
    # XREADGROUP must not be called when XAUTOCLAIM returned something.
    client.xreadgroup.assert_not_called()


def test_ack_calls_xack(queue_with_mock_client):
    q, client = queue_with_mock_client
    q.ack("1700000000000-0")

    client.xack.assert_called_once_with(q._key, q._group, "1700000000000-0")


def test_ensure_group_ignores_busygroup_error(queue_with_mock_client):
    q, client = queue_with_mock_client
    # First enqueue triggers group creation; BUSYGROUP must not bubble.
    import redis as redis_mod

    client.xgroup_create.side_effect = redis_mod.ResponseError(
        "BUSYGROUP Consumer Group name already exists"
    )

    q.enqueue("job-1")  # Should not raise.
    client.xadd.assert_called_once()


def test_depth_returns_lag_from_xinfo_groups(queue_with_mock_client):
    q, client = queue_with_mock_client
    client.xinfo_groups.return_value = [
        {"name": "workers", "lag": 7, "pending": 2},
        {"name": "other", "lag": 99, "pending": 0},
    ]

    assert q.depth() == 7


def test_depth_handles_missing_stream(queue_with_mock_client):
    q, client = queue_with_mock_client
    import redis as redis_mod

    client.xinfo_groups.side_effect = redis_mod.ResponseError("ERR no such key")

    assert q.depth() == 0


def test_enqueue_applies_maxlen_trim(queue_with_mock_client):
    q, client = queue_with_mock_client
    q.enqueue("job-x")

    _, kwargs = client.xadd.call_args
    assert kwargs.get("maxlen") == settings.job_queue_maxlen
    assert kwargs.get("approximate") is True


def test_reclaim_below_threshold_returns_inflight(queue_with_mock_client):
    q, client = queue_with_mock_client
    client.xautoclaim.return_value = (
        b"0-0",
        [(b"1700000000000-0", {b"job_id": b"slow-job"})],
        [],
    )
    # Under the ceiling: normal reclaim.
    client.xpending_range.return_value = [
        {"message_id": b"1700000000000-0", "times_delivered": 2}
    ]

    result = q.dequeue(timeout=1)

    assert result == InFlightJob(job_id="slow-job", entry_id="1700000000000-0")
    # No DLQ write when under the ceiling. xadd is only called via enqueue().
    client.xadd.assert_not_called()


def test_reclaim_above_threshold_routes_to_dlq(queue_with_mock_client):
    q, client = queue_with_mock_client
    client.xautoclaim.return_value = (
        b"0-0",
        [(b"1700000000000-0", {b"job_id": b"poison"})],
        [],
    )
    # Over the ceiling: route to DLQ and ack.
    client.xpending_range.return_value = [
        {"message_id": b"1700000000000-0", "times_delivered": 6}
    ]

    result = q.dequeue(timeout=1)

    assert isinstance(result, DeadLetter)
    assert result.job_id == "poison"
    assert result.entry_id == "1700000000000-0"
    assert result.times_delivered == 6

    # DLQ stream received the fields with trim applied.
    client.xadd.assert_called_once()
    args, kwargs = client.xadd.call_args
    assert args[0] == settings.job_queue_dlq_key
    assert args[1] == {b"job_id": b"poison"}
    assert kwargs.get("maxlen") == settings.job_queue_maxlen
    assert kwargs.get("approximate") is True

    # Original entry acked so it cannot be reclaimed again.
    client.xack.assert_called_once_with(q._key, q._group, "1700000000000-0")

    # XREADGROUP must not be consulted when reclaim produced a result.
    client.xreadgroup.assert_not_called()


def test_reclaim_at_threshold_is_not_dlq(queue_with_mock_client):
    # Boundary: times_delivered == max_deliveries should still process normally.
    # Only strictly greater than max_deliveries triggers DLQ.
    q, client = queue_with_mock_client
    client.xautoclaim.return_value = (
        b"0-0",
        [(b"1700000000000-0", {b"job_id": b"edge"})],
        [],
    )
    client.xpending_range.return_value = [
        {"message_id": b"1700000000000-0", "times_delivered": settings.job_queue_max_deliveries}
    ]

    result = q.dequeue(timeout=1)

    assert isinstance(result, InFlightJob)
    client.xadd.assert_not_called()
