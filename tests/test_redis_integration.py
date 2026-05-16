"""
Integration tests against a real Redis container.

Uses JobQueue directly with an injected Redis client rather than raw redis-py
commands, which avoids redis-py version format differences and tests production
code. Each test gets its own stream/group/dlq names so tests are independent.

The dedup test uses SQLAlchemy directly against an in-memory SQLite DB.
"""

import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone

import pytest
import redis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from shared.models import Base, Job, JobDedup, JobStage, JobStatus
from shared.redis_queue import DeadLetter, InFlightJob, JobQueue


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_redis(host: str, port: int, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if redis.Redis(host=host, port=port).ping():
                return
        except (redis.ConnectionError, redis.ResponseError):
            time.sleep(0.1)
    raise RuntimeError(f"Redis on {host}:{port} did not respond within {timeout}s")


def _make_queue(
    rc: redis.Redis,
    stream: str,
    group: str,
    consumer: str,
    idle_ms: int = 600_000,
    max_deliveries: int = 5,
) -> JobQueue:
    q = JobQueue(consumer_name=consumer, _redis_client=rc)
    q._key = stream
    q._group = group
    q._dlq_key = stream + ":dlq"
    q._idle_ms = idle_ms
    q._max_deliveries = max_deliveries
    q._group_ready = False
    return q


@pytest.fixture(scope="module")
def redis_container():
    port = _free_port()
    container_name = f"test-redis-{uuid.uuid4().hex[:8]}"
    proc = subprocess.run(
        ["docker", "run", "--rm", "-d",
         "--name", container_name,
         "-p", f"{port}:6379",
         "redis:7"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"docker run failed: {proc.stderr}"

    try:
        _wait_for_redis("127.0.0.1", port)
        yield {"host": "127.0.0.1", "port": port}
    finally:
        subprocess.run(["docker", "stop", container_name], capture_output=True)


@pytest.fixture
def rc(redis_container):
    client = redis.Redis(
        host=redis_container["host"], port=redis_container["port"]
    )
    yield client
    client.close()


@pytest.fixture
def stream():
    return f"jobs:{uuid.uuid4().hex[:8]}"


@pytest.fixture
def group():
    return f"workers:{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def sqlite_session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)
    engine.dispose()


class TestHappyPath:
    def test_enqueue_dequeue_ack_clears_pel(self, rc, stream, group):
        job_id = str(uuid.uuid4())
        q = _make_queue(rc, stream, group, "consumer-a")

        q.enqueue(job_id)
        result = q.dequeue(timeout=2)

        assert isinstance(result, InFlightJob)
        assert result.job_id == job_id

        q.ack(result.entry_id)

        pending = rc.xpending(stream, group)
        assert pending["pending"] == 0


class TestReclaimAfterIdle:
    def test_xautoclaim_transfers_entry(self, rc, stream, group):
        job_id = str(uuid.uuid4())
        # Consumer A dequeues but never acks; idle threshold set very low
        q_a = _make_queue(rc, stream, group, "consumer-a", idle_ms=50)
        q_b = _make_queue(rc, stream, group, "consumer-b", idle_ms=50)

        q_a.enqueue(job_id)
        result_a = q_a.dequeue(timeout=2)
        assert isinstance(result_a, InFlightJob)
        assert result_a.job_id == job_id

        # Wait past the idle threshold; consumer-b should reclaim it
        time.sleep(0.2)
        result_b = q_b.dequeue(timeout=2)

        assert isinstance(result_b, InFlightJob), (
            "Consumer B should have reclaimed the idle entry"
        )
        assert result_b.job_id == job_id

        q_b.ack(result_b.entry_id)
        assert rc.xpending(stream, group)["pending"] == 0


class TestDeadLetterRouting:
    def test_entry_lands_in_dlq_after_max_deliveries(self, rc, stream, group):
        job_id = str(uuid.uuid4())
        max_del = 3
        dlq = stream + ":dlq"

        # Initial delivery
        q0 = _make_queue(rc, stream, group, "consumer-0",
                         idle_ms=50, max_deliveries=max_del)
        q0.enqueue(job_id)
        r = q0.dequeue(timeout=2)
        assert isinstance(r, InFlightJob)

        # Reclaim max_del times without acking - each reclaim increments
        # times_delivered; when it exceeds max_deliveries the queue dead-letters
        dead = None
        for i in range(1, max_del + 2):
            time.sleep(0.2)
            q = _make_queue(rc, stream, group, f"consumer-{i}",
                            idle_ms=50, max_deliveries=max_del)
            result = q.dequeue(timeout=2)
            if isinstance(result, DeadLetter):
                dead = result
                break

        assert dead is not None, "Entry should have been dead-lettered"
        assert dead.job_id == job_id

        # Entry must be in the DLQ stream
        dlq_entries = rc.xrange(dlq, count=10)
        assert dlq_entries, "DLQ should contain the dead-lettered entry"
        _, fields = dlq_entries[0]
        assert fields[b"job_id"].decode() == job_id

        # Original PEL must be empty for this job
        pending_range = rc.xpending_range(
            stream, group, min="-", max="+", count=10
        )
        assert not any(
            r.get("consumer", b"").decode() == f"consumer-{i}"
            for r in pending_range
        ), "No pending entries should remain for dead-lettered job"


class TestDedup:
    def test_same_key_produces_one_job(self, sqlite_session_factory):
        Session = sqlite_session_factory
        dedup_key = f"sha256:{uuid.uuid4().hex}"

        with Session() as session:
            job = Job(
                id=str(uuid.uuid4()),
                status=JobStatus.QUEUED,
                stage=JobStage.INGESTION,
                object_key="contracts/raw/test.pdf",
                filename="test.pdf",
                retry_count=0,
                max_retries=3,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            session.add(job)
            session.flush()
            session.add(JobDedup(key=dedup_key, job_id=job.id))
            session.commit()
            first_job_id = job.id

        # Second submission with same key: look up existing, don't insert
        with Session() as session:
            existing = session.get(JobDedup, dedup_key)
            assert existing is not None
            assert existing.job_id == first_job_id
            assert session.query(Job).filter_by(id=first_job_id).count() == 1
