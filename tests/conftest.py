import importlib.util
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock


import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.models import Base, Chunk, Job, JobStage, JobStatus

TEST_DOCUMENTS_DIR = Path(__file__).parent / "test_documents"


def session_factory(SessionLocal):
    """Wrap a SessionLocal as a context manager matching production get_session()."""

    @contextmanager
    def _factory():
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    return _factory


def reset_api_tables(SessionLocal):
    """Clear tables that accumulate across tests sharing the session-scoped engine.

    JobDedup makes identical PDF bodies from different tests collide. Callers that
    submit via the API need a clean slate per test to keep assertions stable.
    """
    # Late import avoids a circular import at module load.
    from shared.models import JobDedup, RiskResult

    with SessionLocal() as session:
        session.query(JobDedup).delete()
        session.query(RiskResult).delete()
        session.query(Chunk).delete()
        session.query(Job).delete()
        session.commit()


@pytest.fixture(scope="session")
def sqlite_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(sqlite_engine):
    """Rolled back after each test to prevent state leakage between tests."""
    Session = sessionmaker(bind=sqlite_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def mock_storage():
    storage = MagicMock()
    storage.upload_bytes.return_value = "contracts/raw/test.pdf"
    storage.upload_json.return_value = "contracts/reports/test.json"
    storage.presigned_url.return_value = "http://minio:9000/contracts/test.json"
    return storage


@pytest.fixture
def mock_queue():
    return MagicMock()


def make_classifier_pipeline(label: str = "indemnification", score: float = 0.85):
    def _pipeline(texts, candidate_labels=None, batch_size=None, multi_label=False):
        if isinstance(texts, str):
            texts = [texts]
        results = [
            {
                "labels": [label] + [l for l in (candidate_labels or []) if l != label],
                "scores": [score] + [0.01] * max(0, len(candidate_labels or []) - 1),
            }
            for _ in texts
        ]
        return results[0] if len(results) == 1 else results

    return _pipeline


def make_tone_pipeline(label: str = "NEGATIVE", score: float = 0.9):
    def _pipeline(texts, batch_size=None):
        if isinstance(texts, str):
            texts = [texts]
        return [{"label": label, "score": score} for _ in texts]

    return _pipeline


@pytest.fixture
def make_job(db_session):
    def _factory(
        status: JobStatus = JobStatus.QUEUED,
        stage: JobStage = JobStage.INGESTION,
        filename: str = "contract.pdf",
    ) -> Job:
        job = Job(
            id=str(uuid.uuid4()),
            status=status,
            stage=stage,
            object_key="contracts/raw/test.pdf",
            filename=filename,
            retry_count=0,
            max_retries=3,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db_session.add(job)
        db_session.commit()
        return job

    return _factory


@pytest.fixture
def make_chunk(db_session):
    def _factory(
        job_id: str,
        text: str,
        index: int = 0,
        clause_type: str = None,
        confidence: float = None,
    ) -> Chunk:
        chunk = Chunk(
            id=str(uuid.uuid4()),
            job_id=job_id,
            index=index,
            text=text,
            token_count=len(text.split()),
            clause_type=clause_type,
            confidence=confidence,
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(chunk)
        db_session.commit()
        return chunk

    return _factory


def _load_seed_module():
    spec = importlib.util.spec_from_file_location(
        "seed", Path(__file__).parent / "seed.py"
    )
    assert spec is not None
    loader = spec.loader
    assert loader is not None
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def make_pdf_bytes():
    return _load_seed_module()._make_pdf_bytes


@pytest.fixture(scope="session")
def seed_contracts():
    return _load_seed_module().SAMPLE_CONTRACTS
