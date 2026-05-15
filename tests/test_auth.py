"""Auth tests: /health is public, protected routes require X-API-Key when configured."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from tests.conftest import reset_api_tables, session_factory


@pytest.fixture
def api_env_with_key(sqlite_engine):
    SessionLocal = sessionmaker(bind=sqlite_engine)
    reset_api_tables(SessionLocal)

    mock_storage = MagicMock()
    mock_storage.upload_bytes.return_value = "contracts/raw/test.pdf"
    mock_storage.presigned_url.return_value = "http://minio:9000/test-report.json"
    mock_queue = MagicMock()

    with (
        patch("api.main.get_session", session_factory(SessionLocal)),
        patch("api.main.StorageClient", return_value=mock_storage),
        patch("api.main.JobQueue", return_value=mock_queue),
        patch("api.auth.settings") as mock_settings,
    ):
        mock_settings.contract_api_key = "test-secret-key"

        from api.main import app
        from api.rate_limit import limiter

        limiter.reset()

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client


def _minimal_pdf() -> bytes:
    from tests.seed import _make_pdf_bytes

    return _make_pdf_bytes("Test contract clause.")


def test_submit_job_rejects_missing_key(api_env_with_key):
    resp = api_env_with_key.post(
        "/jobs", files={"file": ("c.pdf", _minimal_pdf(), "application/pdf")}
    )
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "ApiKey"


def test_submit_job_rejects_wrong_key(api_env_with_key):
    resp = api_env_with_key.post(
        "/jobs",
        files={"file": ("c.pdf", _minimal_pdf(), "application/pdf")},
        headers={"X-API-Key": "wrong"},
    )
    assert resp.status_code == 401


def test_submit_job_accepts_correct_key(api_env_with_key):
    resp = api_env_with_key.post(
        "/jobs",
        files={"file": ("c.pdf", _minimal_pdf(), "application/pdf")},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert resp.status_code == 201


def test_list_jobs_rejects_missing_key(api_env_with_key):
    resp = api_env_with_key.get("/jobs")
    assert resp.status_code == 401


def test_get_job_rejects_missing_key(api_env_with_key):
    resp = api_env_with_key.get("/jobs/some-id")
    assert resp.status_code == 401


def test_get_report_rejects_missing_key(api_env_with_key):
    resp = api_env_with_key.get("/jobs/some-id/report")
    assert resp.status_code == 401


def test_health_is_public_even_when_key_configured(api_env_with_key):
    resp = api_env_with_key.get("/health")
    assert resp.status_code == 200
