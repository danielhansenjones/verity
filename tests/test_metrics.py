"""Metrics tests: /metrics endpoint exposes Prometheus text and counters increment."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from tests.conftest import reset_api_tables


@pytest.fixture
def api_env(sqlite_engine):
    SessionLocal = sessionmaker(bind=sqlite_engine)
    reset_api_tables(SessionLocal)

    mock_storage = MagicMock()
    mock_storage.upload_bytes.return_value = "contracts/raw/test.pdf"
    mock_queue = MagicMock()

    with (
        patch("api.main.init_db"),
        patch("api.main.get_session", side_effect=lambda: SessionLocal()),
        patch("api.main.StorageClient", return_value=mock_storage),
        patch("api.main.JobQueue", return_value=mock_queue),
    ):
        from api.main import app
        from api.rate_limit import limiter

        limiter.reset()

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client


def _pdf(body: str = "metric-test") -> bytes:
    from tests.seed import _make_pdf_bytes

    return _make_pdf_bytes(body)


def test_metrics_endpoint_returns_prometheus_text(api_env):
    resp = api_env.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    # Metric names appear in the exposition format.
    body = resp.text
    assert "jobs_submitted_total" in body
    assert "jobs_completed_total" in body
    assert "job_stage_duration_seconds" in body


def test_metrics_endpoint_is_public(api_env):
    # /metrics must be scrape-able without an API key; auth is disabled in this
    # fixture anyway, but the dependency list should not include require_api_key.
    resp = api_env.get("/metrics")
    assert resp.status_code == 200


def test_submit_increments_created_counter(api_env):
    before = api_env.get("/metrics").text
    api_env.post(
        "/jobs", files={"file": ("a.pdf", _pdf("unique-one"), "application/pdf")}
    )
    after = api_env.get("/metrics").text

    # At least one more "created" sample after the submit.
    def created_count(text: str) -> float:
        for line in text.splitlines():
            if line.startswith('jobs_submitted_total{outcome="created"}'):
                return float(line.split()[-1])
        return 0.0

    assert created_count(after) == created_count(before) + 1.0


def test_replay_increments_replayed_counter(api_env):
    body = _pdf("unique-two")
    api_env.post("/jobs", files={"file": ("a.pdf", body, "application/pdf")})
    before = api_env.get("/metrics").text

    # Same body -> replay.
    api_env.post("/jobs", files={"file": ("a.pdf", body, "application/pdf")})
    after = api_env.get("/metrics").text

    def replayed_count(text: str) -> float:
        for line in text.splitlines():
            if line.startswith('jobs_submitted_total{outcome="replayed"}'):
                return float(line.split()[-1])
        return 0.0

    assert replayed_count(after) == replayed_count(before) + 1.0
