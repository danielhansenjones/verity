"""Rate limit tests for protected endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from tests.conftest import reset_api_tables, session_factory


@pytest.fixture
def rate_limited_env(sqlite_engine):
    """API client with tight rate limits so 429s are cheap to trigger."""
    SessionLocal = sessionmaker(bind=sqlite_engine)
    reset_api_tables(SessionLocal)

    mock_storage = MagicMock()
    mock_storage.upload_bytes.return_value = "contracts/raw/test.pdf"
    mock_queue = MagicMock()

    with (
        patch("api.main.get_session", session_factory(SessionLocal)),
        patch("api.main.StorageClient", return_value=mock_storage),
        patch("api.main.JobQueue", return_value=mock_queue),
        patch("api.rate_limit.settings") as mock_rate_settings,
    ):
        mock_rate_settings.rate_limit_submit = "2/minute"
        mock_rate_settings.rate_limit_read = "3/minute"

        from api.main import app
        from api.rate_limit import limiter

        limiter.reset()

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client

        limiter.reset()


def _tiny_pdf(body: str = "x") -> bytes:
    from tests.seed import _make_pdf_bytes

    return _make_pdf_bytes(body)


def test_submit_rate_limit_returns_429_after_breach(rate_limited_env):
    # Limit is 2/minute; the third submission must be rejected.
    # Unique bodies per iteration so idempotency replay doesn't mask the rate check.
    for i in range(2):
        resp = rate_limited_env.post(
            "/jobs",
            files={"file": ("a.pdf", _tiny_pdf(f"body-{i}"), "application/pdf")},
        )
        assert resp.status_code == 201

    resp = rate_limited_env.post(
        "/jobs",
        files={"file": ("a.pdf", _tiny_pdf("body-final"), "application/pdf")},
    )
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_read_rate_limit_returns_429_after_breach(rate_limited_env):
    # Read limit is 3/minute; the fourth GET must be rejected.
    for _ in range(3):
        resp = rate_limited_env.get("/jobs")
        assert resp.status_code == 200

    resp = rate_limited_env.get("/jobs")
    assert resp.status_code == 429


def test_read_and_submit_limits_are_independent(rate_limited_env):
    # Read and submit use separate limit windows.
    for _ in range(3):
        assert rate_limited_env.get("/jobs").status_code == 200

    resp = rate_limited_env.post(
        "/jobs",
        files={"file": ("a.pdf", _tiny_pdf(), "application/pdf")},
    )
    assert resp.status_code == 201


def test_health_is_not_rate_limited(rate_limited_env):
    for _ in range(10):
        resp = rate_limited_env.get("/health")
        assert resp.status_code == 200
