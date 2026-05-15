"""Idempotency tests: duplicate submissions return the same job id."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from shared.models import JobDedup
from tests.conftest import session_factory


@pytest.fixture
def api_env(sqlite_engine):
    SessionLocal = sessionmaker(bind=sqlite_engine)

    mock_storage = MagicMock()
    mock_storage.upload_bytes.return_value = "contracts/raw/test.pdf"
    mock_queue = MagicMock()

    with (
        patch("api.main.get_session", session_factory(SessionLocal)),
        patch("api.main.StorageClient", return_value=mock_storage),
        patch("api.main.JobQueue", return_value=mock_queue),
    ):
        from api.main import app
        from api.rate_limit import limiter

        limiter.reset()

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, mock_storage, mock_queue, SessionLocal

        limiter.reset()


def _pdf(body: str = "clause text") -> bytes:
    from tests.seed import _make_pdf_bytes

    return _make_pdf_bytes(body)


def test_same_pdf_returns_same_job_id_via_content_hash(api_env):
    client, _, _, _ = api_env
    body = _pdf("identical content")

    r1 = client.post("/jobs", files={"file": ("a.pdf", body, "application/pdf")})
    r2 = client.post("/jobs", files={"file": ("a.pdf", body, "application/pdf")})

    assert r1.status_code == 201
    assert r2.status_code == 200
    assert r2.headers.get("Idempotent-Replay") == "true"
    assert r1.json()["job_id"] == r2.json()["job_id"]


def test_different_pdfs_get_different_job_ids(api_env):
    client, _, _, _ = api_env

    r1 = client.post("/jobs", files={"file": ("a.pdf", _pdf("A"), "application/pdf")})
    r2 = client.post("/jobs", files={"file": ("b.pdf", _pdf("B"), "application/pdf")})

    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r1.json()["job_id"] != r2.json()["job_id"]


def test_explicit_idempotency_key_dedups_across_different_pdfs(api_env):
    client, _, _, _ = api_env
    headers = {"Idempotency-Key": "order-12345"}

    r1 = client.post(
        "/jobs",
        files={"file": ("a.pdf", _pdf("first"), "application/pdf")},
        headers=headers,
    )
    # Different bytes, same key -> same job.
    r2 = client.post(
        "/jobs",
        files={"file": ("b.pdf", _pdf("second"), "application/pdf")},
        headers=headers,
    )

    assert r1.status_code == 201
    assert r2.status_code == 200
    assert r1.json()["job_id"] == r2.json()["job_id"]


def test_client_key_does_not_collide_with_content_hash(api_env):
    """A client key that happens to equal a content hash hex must not dedup to it."""
    client, _, _, _ = api_env
    body = _pdf("collision-test")
    import hashlib

    content_hex = hashlib.sha256(body).hexdigest()

    r1 = client.post(
        "/jobs", files={"file": ("a.pdf", body, "application/pdf")}
    )
    # Same hex as header; must NOT hit the content-hashed entry because of namespace prefix.
    r2 = client.post(
        "/jobs",
        files={"file": ("b.pdf", _pdf("different"), "application/pdf")},
        headers={"Idempotency-Key": content_hex},
    )

    assert r1.status_code == 201
    assert r2.status_code == 201
    assert r1.json()["job_id"] != r2.json()["job_id"]


def test_dedup_row_persists_with_namespace_prefix(api_env):
    client, _, _, SessionLocal = api_env
    body = _pdf("namespace-check-unique")

    r = client.post("/jobs", files={"file": ("a.pdf", body, "application/pdf")})
    assert r.status_code == 201

    job_id = r.json()["job_id"]
    with SessionLocal() as session:
        row = session.query(JobDedup).filter(JobDedup.job_id == job_id).one()
        assert row.key.startswith("content:")


def test_replay_does_not_re_upload_or_re_enqueue(api_env):
    client, mock_storage, mock_queue, _ = api_env
    body = _pdf("no-double-work")

    client.post("/jobs", files={"file": ("a.pdf", body, "application/pdf")})
    client.post("/jobs", files={"file": ("a.pdf", body, "application/pdf")})

    assert mock_storage.upload_bytes.call_count == 1
    assert mock_queue.enqueue.call_count == 1
