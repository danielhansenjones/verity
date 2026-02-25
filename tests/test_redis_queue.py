"""Unit tests for the Redis Streams-backed JobQueue wrapper.

These tests stub the redis client directly rather than going through conftest's
module-level redis stub, so they exercise the XADD/XREADGROUP/XACK flow.
"""

from unittest.mock import MagicMock, patch

import pytest

from shared.redis_queue import InFlightJob, JobQueue


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
