"""Request size limit tests for POST /jobs."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from tests.conftest import reset_api_tables, session_factory


@pytest.fixture
def small_limit_env(sqlite_engine):
    """API client with max_upload_bytes lowered to 1 KiB for cheap tests."""
    SessionLocal = sessionmaker(bind=sqlite_engine)
    reset_api_tables(SessionLocal)

    mock_storage = MagicMock()
    mock_storage.upload_bytes.return_value = "contracts/raw/test.pdf"
    mock_queue = MagicMock()

    with (
        patch("api.main.get_session", session_factory(SessionLocal)),
        patch("api.main.StorageClient", return_value=mock_storage),
        patch("api.main.JobQueue", return_value=mock_queue),
        patch("api.main.settings") as mock_settings,
    ):
        mock_settings.max_upload_bytes = 1024

        from api.main import app
        from api.rate_limit import limiter

        limiter.reset()

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client


def _tiny_pdf() -> bytes:
    from tests.seed import _make_pdf_bytes

    return _make_pdf_bytes("x")


def test_rejects_oversized_upload_with_413(small_limit_env):
    oversize = b"%PDF-1.4\n" + b"A" * 4096
    resp = small_limit_env.post(
        "/jobs",
        files={"file": ("big.pdf", oversize, "application/pdf")},
    )
    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"].lower()


def test_accepts_upload_under_limit(small_limit_env):
    resp = small_limit_env.post(
        "/jobs",
        files={"file": ("small.pdf", _tiny_pdf(), "application/pdf")},
    )
    assert resp.status_code == 201


def test_rejects_missing_content_length_with_411(small_limit_env):
    # httpx always sets Content-Length for in-memory bodies, so strip it
    # via a raw request build to exercise the 411 path.
    req = small_limit_env.build_request(
        "POST",
        "/jobs",
        files={"file": ("small.pdf", _tiny_pdf(), "application/pdf")},
    )
    del req.headers["content-length"]
    req.headers["transfer-encoding"] = "chunked"
    resp = small_limit_env.send(req)
    assert resp.status_code == 411


def test_rejects_invalid_content_length_with_400(small_limit_env):
    req = small_limit_env.build_request(
        "POST",
        "/jobs",
        files={"file": ("small.pdf", _tiny_pdf(), "application/pdf")},
    )
    req.headers["content-length"] = "not-a-number"
    resp = small_limit_env.send(req)
    assert resp.status_code == 400


def test_other_endpoints_unaffected(small_limit_env):
    # /health has no body; the middleware should ignore it.
    resp = small_limit_env.get("/health")
    assert resp.status_code == 200
