"""
All external dependencies (Postgres, MinIO, Redis) are replaced with mocks or
an in-memory SQLite engine so the suite runs without any running services.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from shared.models import Job, JobStage, JobStatus, RiskResult


@pytest.fixture
def api_env(sqlite_engine):
    SessionLocal = sessionmaker(bind=sqlite_engine)

    mock_storage = MagicMock()
    mock_storage.upload_bytes.return_value = "contracts/raw/test.pdf"
    mock_storage.presigned_url.return_value = "http://minio:9000/test-report.json"

    mock_queue = MagicMock()

    with (
        patch("api.main.init_db"),
        patch("api.main.get_session", side_effect=lambda: SessionLocal()),
        patch("api.main.StorageClient", return_value=mock_storage),
        patch("api.main.JobQueue", return_value=mock_queue),
    ):
        from api.main import app

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, mock_storage, mock_queue


def _minimal_pdf() -> bytes:
    """Minimal but valid PDF so the endpoint does not reject the upload."""
    from tests.seed import _make_pdf_bytes

    return _make_pdf_bytes("Test contract clause.")


def test_submit_job_returns_201_with_job_id(api_env):
    client, _, _ = api_env
    resp = client.post(
        "/jobs", files={"file": ("contract.pdf", _minimal_pdf(), "application/pdf")}
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "job_id" in body
    assert body["status"] == "queued"
    assert body["filename"] == "contract.pdf"


def test_submit_job_rejects_non_pdf(api_env):
    client, _, _ = api_env
    resp = client.post(
        "/jobs", files={"file": ("notes.txt", b"not a pdf", "text/plain")}
    )
    assert resp.status_code == 400
    assert "PDF" in resp.json()["detail"]


def test_submit_job_uploads_to_storage(api_env):
    client, mock_storage, _ = api_env
    client.post(
        "/jobs", files={"file": ("contract.pdf", _minimal_pdf(), "application/pdf")}
    )
    mock_storage.upload_bytes.assert_called_once()
    key, data, *_ = mock_storage.upload_bytes.call_args[0]
    assert key.startswith("contracts/raw/")
    assert key.endswith(".pdf")


def test_submit_job_enqueues_job_id(api_env):
    client, _, mock_queue = api_env
    resp = client.post(
        "/jobs", files={"file": ("contract.pdf", _minimal_pdf(), "application/pdf")}
    )
    job_id = resp.json()["job_id"]
    mock_queue.enqueue.assert_called_once_with(job_id)


def test_get_job_returns_status(api_env):
    client, _, _ = api_env
    # Submit first so the job exists in the DB before we query it
    resp = client.post(
        "/jobs", files={"file": ("contract.pdf", _minimal_pdf(), "application/pdf")}
    )
    job_id = resp.json()["job_id"]

    resp = client.get(f"/jobs/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == job_id
    assert body["status"] == "queued"
    assert body["stage"] == "ingestion"
    assert body["filename"] == "contract.pdf"
    assert body["retry_count"] == 0


def test_get_job_not_found(api_env):
    client, _, _ = api_env
    resp = client.get(f"/jobs/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_list_jobs_returns_list(api_env):
    client, _, _ = api_env
    resp = client.get("/jobs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_list_jobs_includes_submitted_job(api_env):
    client, _, _ = api_env
    client.post("/jobs", files={"file": ("a.pdf", _minimal_pdf(), "application/pdf")})
    client.post("/jobs", files={"file": ("b.pdf", _minimal_pdf(), "application/pdf")})

    resp = client.get("/jobs")
    assert len(resp.json()) >= 2


def test_list_jobs_filters_by_status(api_env):
    client, _, _ = api_env
    client.post("/jobs", files={"file": ("q.pdf", _minimal_pdf(), "application/pdf")})

    resp = client.get("/jobs?status=queued")
    assert resp.status_code == 200
    for item in resp.json():
        assert item["status"] == "queued"


def test_list_jobs_invalid_status_returns_400(api_env):
    client, _, _ = api_env
    resp = client.get("/jobs?status=not_a_status")
    assert resp.status_code == 400


def test_get_report_job_not_found(api_env):
    client, _, _ = api_env
    resp = client.get(f"/jobs/{uuid.uuid4()}/report")
    assert resp.status_code == 404


def test_get_report_returns_409_when_job_not_completed(api_env):
    client, _, _ = api_env
    resp = client.post(
        "/jobs", files={"file": ("c.pdf", _minimal_pdf(), "application/pdf")}
    )
    job_id = resp.json()["job_id"]

    resp = client.get(f"/jobs/{job_id}/report")
    assert resp.status_code == 409
    assert "not completed" in resp.json()["detail"]


def test_get_report_returns_404_when_no_result_record(api_env, sqlite_engine):
    """A completed job with no RiskResult row is an inconsistent state we must not crash on."""
    client, _, _ = api_env
    SessionLocal = sessionmaker(bind=sqlite_engine)

    resp = client.post(
        "/jobs", files={"file": ("d.pdf", _minimal_pdf(), "application/pdf")}
    )
    job_id = resp.json()["job_id"]

    with SessionLocal() as session:
        job = session.get(Job, job_id)
        job.status = JobStatus.COMPLETED
        session.commit()

    resp = client.get(f"/jobs/{job_id}/report")
    assert resp.status_code == 404


def test_get_report_success(api_env, sqlite_engine):
    client, mock_storage, _ = api_env
    SessionLocal = sessionmaker(bind=sqlite_engine)

    resp = client.post(
        "/jobs", files={"file": ("e.pdf", _minimal_pdf(), "application/pdf")}
    )
    job_id = resp.json()["job_id"]

    with SessionLocal() as session:
        job = session.get(Job, job_id)
        job.status = JobStatus.COMPLETED
        job.stage = JobStage.DONE

        result = RiskResult(
            id=str(uuid.uuid4()),
            job_id=job_id,
            overall_score=72,
            risk_level="high",
            clause_summary={"indemnification": 2, "termination": 1},
            flags=[{"chunk_index": 0, "severity": "high", "reason": "test"}],
            report_key=f"contracts/reports/{job_id}.json",
            created_at=datetime.now(timezone.utc),
        )
        session.add(result)
        session.commit()

    resp = client.get(f"/jobs/{job_id}/report")
    assert resp.status_code == 200

    body = resp.json()
    assert body["job_id"] == job_id
    assert body["overall_score"] == 72
    assert body["risk_level"] == "high"
    assert body["clause_summary"] == {"indemnification": 2, "termination": 1}
    assert "report_url" in body
    assert body["report_url"] == "http://minio:9000/test-report.json"
