"""
Integration test for shared.models.init_db against a fresh pgvector-enabled
Postgres. Covers the bug where Base.metadata.create_all ran before the vector
extension existed, and the migration's add_column would then duplicate work
that create_all had already done on a fresh database.

Spawns a one-shot pgvector container per module like test_redis_integration.
Skips when the docker CLI is unavailable.
"""

import shutil
import socket
import subprocess
import time
import uuid

import pytest
from sqlalchemy import create_engine, inspect, text


pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None,
    reason="docker CLI is required to run pgvector integration tests",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_pg(dsn: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            engine = create_engine(dsn)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine.dispose()
            return
        except Exception as exc:
            last_err = exc
            time.sleep(0.2)
    raise RuntimeError(
        f"pgvector container did not become reachable within {timeout}s: {last_err}"
    )


@pytest.fixture(scope="module")
def pgvector_container():
    port = _free_port()
    container_name = f"test-pgvector-{uuid.uuid4().hex[:8]}"
    proc = subprocess.run(
        [
            "docker", "run", "--rm", "-d",
            "--name", container_name,
            "-e", "POSTGRES_USER=test",
            "-e", "POSTGRES_PASSWORD=test",
            "-e", "POSTGRES_DB=test",
            "-p", f"{port}:5432",
            "pgvector/pgvector:pg16-trixie",
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"pgvector container failed to start: {proc.stderr}")

    dsn = f"postgresql+psycopg2://test:test@127.0.0.1:{port}/test"
    try:
        _wait_for_pg(dsn)
        yield dsn
    finally:
        subprocess.run(["docker", "stop", container_name], capture_output=True)


@pytest.fixture
def fresh_engine(pgvector_container, monkeypatch):
    # init_db() reads the module-level _engine for create_all, and alembic's
    # env.py reads settings.postgres_* to build its own DSN. Point both at the
    # test container so the two halves of init_db target the same database.
    from shared import models
    from shared.settings import settings

    # postgresql+psycopg2://test:test@127.0.0.1:PORT/test
    rest = pgvector_container.split("://", 1)[1]
    userpass, hostpart = rest.split("@", 1)
    user, password = userpass.split(":", 1)
    host_port, dbname = hostpart.split("/", 1)

    monkeypatch.setattr(settings, "postgres_user", user)
    monkeypatch.setattr(settings, "postgres_password", password)
    monkeypatch.setattr(settings, "postgres_host", host_port)
    monkeypatch.setattr(settings, "postgres_db", dbname)

    test_engine = create_engine(pgvector_container, pool_pre_ping=True)
    monkeypatch.setattr(models, "_engine", test_engine)
    yield test_engine
    test_engine.dispose()


def test_init_db_on_fresh_pgvector_creates_chunks_with_embedding(fresh_engine):
    from shared.models import init_db

    init_db()

    insp = inspect(fresh_engine)
    cols = {c["name"]: c for c in insp.get_columns("chunks")}
    assert "embedding" in cols, "chunks.embedding missing after init_db on fresh DB"


def test_init_db_is_idempotent(fresh_engine):
    # Two back-to-back invocations must succeed. The first creates the schema,
    # the second exercises the "everything already exists" path that broke when
    # create_all and alembic disagreed on column ownership.
    from shared.models import init_db

    init_db()
    init_db()

    with fresh_engine.connect() as conn:
        ext = conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        assert ext == "vector"

        idx = conn.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'chunks' AND indexname = 'chunks_embedding_idx'"
            )
        ).scalar()
        assert idx == "chunks_embedding_idx"


def test_alembic_marks_revision_after_init_db(fresh_engine):
    from shared.models import init_db

    init_db()

    with fresh_engine.connect() as conn:
        version = conn.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar()
        assert version == "0001"
