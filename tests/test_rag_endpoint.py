import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from api.llm import AnswerResponse, Citation
from shared.models import Chunk, Job, JobStage, JobStatus, RagQuery
from tests.conftest import reset_api_tables, session_factory


@pytest.fixture
def rag_env(sqlite_engine, monkeypatch):
    SessionLocal = sessionmaker(bind=sqlite_engine)
    reset_api_tables(SessionLocal)

    monkeypatch.setattr("api.main.settings.anthropic_api_key", "test-key")

    with (
        patch("api.main.preload_embedding_model"),
        patch("api.main.get_session", session_factory(SessionLocal)),
        patch("api.main.StorageClient", return_value=MagicMock()),
        patch("api.main.JobQueue", return_value=MagicMock()),
    ):
        from api.main import app
        from api.rate_limit import limiter

        limiter.reset()

        with TestClient(app, raise_server_exceptions=True) as client:
            yield client, SessionLocal


def _seed_job(SessionLocal, status: JobStatus, stage: JobStage = JobStage.DONE) -> str:
    job_id = str(uuid.uuid4())
    with SessionLocal() as session:
        session.add(
            Job(
                id=job_id,
                status=status,
                stage=stage,
                object_key=f"contracts/raw/{job_id}.pdf",
                filename="contract.pdf",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        )
        session.commit()
    return job_id


def _seed_chunks(SessionLocal, job_id: str, texts: list[str]) -> list[Chunk]:
    chunks = []
    with SessionLocal() as session:
        for i, text in enumerate(texts):
            chunk = Chunk(
                id=str(uuid.uuid4()),
                job_id=job_id,
                index=i,
                text=text,
                token_count=len(text.split()),
                created_at=datetime.now(timezone.utc),
            )
            session.add(chunk)
            chunks.append(chunk)
        session.commit()
        for c in chunks:
            session.refresh(c)
            session.expunge(c)
    return chunks


def test_ask_returns_404_when_job_missing(rag_env):
    client, _ = rag_env
    resp = client.post(
        "/jobs/nonexistent/ask", json={"question": "What is the governing law?"}
    )
    assert resp.status_code == 404


def test_ask_returns_409_when_job_not_completed(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.RUNNING, stage=JobStage.CLASSIFICATION)

    resp = client.post(f"/jobs/{job_id}/ask", json={"question": "anything"})
    assert resp.status_code == 409
    assert "not completed" in resp.json()["detail"].lower()


def test_ask_returns_503_when_anthropic_key_missing(rag_env, monkeypatch):
    client, SessionLocal = rag_env
    monkeypatch.setattr("api.main.settings.anthropic_api_key", None)
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)

    resp = client.post(f"/jobs/{job_id}/ask", json={"question": "anything"})
    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]


def test_ask_returns_400_for_empty_question(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)

    resp = client.post(f"/jobs/{job_id}/ask", json={"question": "   "})
    assert resp.status_code == 400


def test_ask_returns_400_for_oversized_question(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)

    resp = client.post(
        f"/jobs/{job_id}/ask", json={"question": "x" * 2001}
    )
    assert resp.status_code == 400


def test_ask_refuses_when_no_embedded_chunks(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)

    with (
        patch("api.main.embed_query", return_value=[0.0] * 384),
        patch("api.main.retrieve", return_value=[]),
    ):
        resp = client.post(
            f"/jobs/{job_id}/ask", json={"question": "What is the governing law?"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == ""
    assert body["citations"] == []
    assert body["refusal_reason"] is not None
    assert "backfill" in body["refusal_reason"].lower()


def test_ask_returns_grounded_answer(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    chunks = _seed_chunks(
        SessionLocal,
        job_id,
        ["Delaware law governs this agreement.", "Either party may terminate."],
    )

    fake_answer = AnswerResponse(
        answer="Delaware law governs.",
        citations=[
            Citation(
                chunk_id=chunks[0].id,
                chunk_index=0,
                quote="Delaware law governs",
            )
        ],
    )
    fake_usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }

    with (
        patch("api.main.embed_query", return_value=[0.0] * 384),
        patch("api.main.retrieve", return_value=chunks),
        patch("api.main.llm_ask", return_value=(fake_answer, fake_usage)),
    ):
        resp = client.post(
            f"/jobs/{job_id}/ask", json={"question": "What is the governing law?"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Delaware law governs."
    assert len(body["citations"]) == 1
    assert body["citations"][0]["chunk_id"] == chunks[0].id
    assert body["refusal_reason"] is None


def test_ask_passes_llm_refusal_through(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    chunks = _seed_chunks(SessionLocal, job_id, ["Delaware law governs."])

    fake_refusal = AnswerResponse(
        answer="",
        citations=[],
        refusal_reason="No clause in the retrieved chunks addresses indemnification.",
    )
    fake_usage = {"input_tokens": 80, "output_tokens": 15}

    with (
        patch("api.main.embed_query", return_value=[0.0] * 384),
        patch("api.main.retrieve", return_value=chunks),
        patch("api.main.llm_ask", return_value=(fake_refusal, fake_usage)),
    ):
        resp = client.post(
            f"/jobs/{job_id}/ask", json={"question": "What is the indemnification cap?"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == ""
    assert body["refusal_reason"] is not None
    assert "indemnification" in body["refusal_reason"].lower()


def test_ask_returns_502_when_llm_fabricates_chunk_id(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    chunks = _seed_chunks(SessionLocal, job_id, ["Delaware law governs."])

    fake_answer = AnswerResponse(
        answer="The cap is five million dollars.",
        citations=[
            Citation(
                chunk_id="bogus-chunk-id",
                chunk_index=99,
                quote="five million",
            )
        ],
    )
    fake_usage = {"input_tokens": 100, "output_tokens": 30}

    with (
        patch("api.main.embed_query", return_value=[0.0] * 384),
        patch("api.main.retrieve", return_value=chunks),
        patch("api.main.llm_ask", return_value=(fake_answer, fake_usage)),
    ):
        resp = client.post(
            f"/jobs/{job_id}/ask", json={"question": "What is the cap?"}
        )

    assert resp.status_code == 502
    assert "ungrounded" in resp.json()["detail"].lower()


def test_ask_returns_502_when_llm_fabricates_quote(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    chunks = _seed_chunks(SessionLocal, job_id, ["Delaware law governs."])

    fake_answer = AnswerResponse(
        answer="The cap is five million dollars.",
        citations=[
            Citation(
                chunk_id=chunks[0].id,
                chunk_index=0,
                quote="five million dollar cap",
            )
        ],
    )
    fake_usage = {"input_tokens": 100, "output_tokens": 30}

    with (
        patch("api.main.embed_query", return_value=[0.0] * 384),
        patch("api.main.retrieve", return_value=chunks),
        patch("api.main.llm_ask", return_value=(fake_answer, fake_usage)),
    ):
        resp = client.post(
            f"/jobs/{job_id}/ask", json={"question": "What is the cap?"}
        )

    assert resp.status_code == 502


def test_ask_returns_502_when_llm_call_raises(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    chunks = _seed_chunks(SessionLocal, job_id, ["Delaware law governs."])

    with (
        patch("api.main.embed_query", return_value=[0.0] * 384),
        patch("api.main.retrieve", return_value=chunks),
        patch(
            "api.main.llm_ask",
            side_effect=RuntimeError("upstream model unavailable"),
        ),
    ):
        resp = client.post(
            f"/jobs/{job_id}/ask", json={"question": "anything"}
        )

    assert resp.status_code == 502
    assert "generation failed" in resp.json()["detail"].lower()


def test_ask_persists_rag_query_on_answer(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    chunks = _seed_chunks(
        SessionLocal, job_id, ["Delaware law governs this agreement."]
    )

    fake_answer = AnswerResponse(
        answer="Delaware law governs.",
        citations=[
            Citation(
                chunk_id=chunks[0].id, chunk_index=0, quote="Delaware law governs"
            )
        ],
    )
    fake_usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 5,
        "cache_creation_input_tokens": 0,
    }

    with (
        patch("api.main.embed_query", return_value=[0.0] * 384),
        patch("api.main.retrieve", return_value=chunks),
        patch("api.main.llm_ask", return_value=(fake_answer, fake_usage)),
    ):
        resp = client.post(
            f"/jobs/{job_id}/ask", json={"question": "What is the governing law?"}
        )

    assert resp.status_code == 200

    with SessionLocal() as session:
        rows = session.query(RagQuery).filter(RagQuery.job_id == job_id).all()

    assert len(rows) == 1
    row = rows[0]
    assert row.outcome == "answered"
    assert row.question == "What is the governing law?"
    assert row.retrieved_chunk_ids == [chunks[0].id]
    assert row.answer == "Delaware law governs."
    assert row.citations[0]["chunk_id"] == chunks[0].id
    assert row.input_tokens == 100
    assert row.cache_read_tokens == 5
    assert row.grounding_error is None
    assert row.error is None


def test_ask_persists_rag_query_on_grounding_failure(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    chunks = _seed_chunks(SessionLocal, job_id, ["Delaware law governs."])

    fake_answer = AnswerResponse(
        answer="The cap is five million dollars.",
        citations=[
            Citation(
                chunk_id=chunks[0].id,
                chunk_index=0,
                quote="five million dollar cap",
            )
        ],
    )
    fake_usage = {"input_tokens": 100, "output_tokens": 30}

    with (
        patch("api.main.embed_query", return_value=[0.0] * 384),
        patch("api.main.retrieve", return_value=chunks),
        patch("api.main.llm_ask", return_value=(fake_answer, fake_usage)),
    ):
        resp = client.post(
            f"/jobs/{job_id}/ask", json={"question": "What is the cap?"}
        )

    assert resp.status_code == 502

    with SessionLocal() as session:
        rows = session.query(RagQuery).filter(RagQuery.job_id == job_id).all()

    assert len(rows) == 1
    assert rows[0].outcome == "error"
    assert rows[0].grounding_error is not None
    assert rows[0].input_tokens == 100


def test_ask_clamps_top_k_to_max_via_retrieve(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)

    retrieve_mock = MagicMock(return_value=[])
    with (
        patch("api.main.embed_query", return_value=[0.0] * 384),
        patch("api.main.retrieve", retrieve_mock),
    ):
        client.post(
            f"/jobs/{job_id}/ask",
            json={"question": "anything", "top_k": 9999},
        )

    # The endpoint passes top_k through; the clamp lives in retrieve().
    _, kwargs = retrieve_mock.call_args
    assert kwargs["k"] == 9999
