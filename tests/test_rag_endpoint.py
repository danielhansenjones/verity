import uuid
from datetime import datetime, timedelta, timezone
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


def _seed_rag_query(
    SessionLocal,
    job_id: str,
    outcome: str,
    *,
    question: str = "q",
    created_at: datetime = None,
    **fields,
) -> str:
    rag_id = str(uuid.uuid4())
    with SessionLocal() as session:
        session.add(
            RagQuery(
                id=rag_id,
                job_id=job_id,
                question=question,
                top_k=8,
                model="test-model",
                outcome=outcome,
                retrieved_chunk_ids=[],
                created_at=created_at or datetime.now(timezone.utc),
                **fields,
            )
        )
        session.commit()
    return rag_id


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

    trace_id = resp.headers["X-Trace-Id"]
    with SessionLocal() as session:
        row = session.get(RagQuery, trace_id)
    assert row is not None
    assert row.outcome == "refused"


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


def test_ask_sets_trace_id_header_on_answered(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    chunks = _seed_chunks(SessionLocal, job_id, ["Delaware law governs this agreement."])

    fake_answer = AnswerResponse(
        answer="Delaware law governs.",
        citations=[
            Citation(chunk_id=chunks[0].id, chunk_index=0, quote="Delaware law governs")
        ],
    )
    fake_usage = {"input_tokens": 100, "output_tokens": 20}

    with (
        patch("api.main.embed_query", return_value=[0.0] * 384),
        patch("api.main.retrieve", return_value=chunks),
        patch("api.main.llm_ask", return_value=(fake_answer, fake_usage)),
    ):
        resp = client.post(
            f"/jobs/{job_id}/ask", json={"question": "What is the governing law?"}
        )

    assert resp.status_code == 200
    trace_id = resp.headers["X-Trace-Id"]

    with SessionLocal() as session:
        row = session.get(RagQuery, trace_id)
    assert row is not None
    assert row.outcome == "answered"
    assert row.job_id == job_id


def test_ask_sets_trace_id_header_on_refused(rag_env):
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
            f"/jobs/{job_id}/ask", json={"question": "indemnification cap?"}
        )

    assert resp.status_code == 200
    trace_id = resp.headers["X-Trace-Id"]

    with SessionLocal() as session:
        row = session.get(RagQuery, trace_id)
    assert row is not None
    assert row.outcome == "refused"


def test_ask_sets_trace_id_header_on_generation_error(rag_env):
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
        resp = client.post(f"/jobs/{job_id}/ask", json={"question": "anything"})

    assert resp.status_code == 502
    trace_id = resp.headers["X-Trace-Id"]

    with SessionLocal() as session:
        row = session.get(RagQuery, trace_id)
    assert row is not None
    assert row.outcome == "error"
    assert row.error is not None


def test_ask_sets_trace_id_header_on_grounding_error(rag_env):
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
        resp = client.post(f"/jobs/{job_id}/ask", json={"question": "cap?"})

    assert resp.status_code == 502
    trace_id = resp.headers["X-Trace-Id"]

    with SessionLocal() as session:
        row = session.get(RagQuery, trace_id)
    assert row is not None
    assert row.outcome == "error"
    assert row.grounding_error is not None


def test_ask_log_failure_increments_counter_and_still_succeeds(rag_env):
    from prometheus_client import REGISTRY

    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    chunks = _seed_chunks(SessionLocal, job_id, ["Delaware law governs."])

    fake_answer = AnswerResponse(
        answer="Delaware law governs.",
        citations=[
            Citation(chunk_id=chunks[0].id, chunk_index=0, quote="Delaware law governs")
        ],
    )
    fake_usage = {"input_tokens": 100, "output_tokens": 20}

    before = REGISTRY.get_sample_value("rag_query_log_failures_total") or 0.0

    # Forcing RagQuery(...) to raise inside _log_rag_query exercises the except
    # branch: rollback, counter increment, swallow, and let the answer through.
    with (
        patch("api.main.embed_query", return_value=[0.0] * 384),
        patch("api.main.retrieve", return_value=chunks),
        patch("api.main.llm_ask", return_value=(fake_answer, fake_usage)),
        patch("api.main.RagQuery", side_effect=RuntimeError("db down")),
    ):
        resp = client.post(
            f"/jobs/{job_id}/ask", json={"question": "What is the governing law?"}
        )

    assert resp.status_code == 200
    assert resp.json()["answer"] == "Delaware law governs."
    assert "x-trace-id" not in {k.lower() for k in resp.headers.keys()}

    after = REGISTRY.get_sample_value("rag_query_log_failures_total") or 0.0
    assert after == before + 1.0


def test_admin_rag_queries_filters_by_outcome_and_job(rag_env):
    client, SessionLocal = rag_env
    job_a = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    job_b = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    _seed_rag_query(SessionLocal, job_a, "answered", question="ans-a")
    _seed_rag_query(SessionLocal, job_a, "error", question="err-a", error="boom")
    _seed_rag_query(SessionLocal, job_b, "answered", question="ans-b")

    resp = client.get("/admin/rag_queries", params={"job_id": job_a})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    assert all(r["job_id"] == job_a for r in rows)

    resp = client.get("/admin/rag_queries", params={"outcome": "answered"})
    rows = resp.json()
    assert {r["job_id"] for r in rows} == {job_a, job_b}
    assert all(r["outcome"] == "answered" for r in rows)

    resp = client.get(
        "/admin/rag_queries", params={"job_id": job_a, "outcome": "error"}
    )
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["outcome"] == "error"
    assert rows[0]["error"] == "boom"


def test_admin_rag_queries_rejects_unknown_outcome(rag_env):
    client, _ = rag_env
    resp = client.get("/admin/rag_queries", params={"outcome": "bogus"})
    assert resp.status_code == 400


def test_admin_rag_queries_rejects_invalid_before(rag_env):
    client, _ = rag_env
    resp = client.get("/admin/rag_queries", params={"before": "not-a-timestamp"})
    assert resp.status_code == 400


def test_admin_rag_queries_truncates_question_to_200_chars(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    long_q = "x" * 500
    _seed_rag_query(SessionLocal, job_id, "answered", question=long_q)

    resp = client.get("/admin/rag_queries")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert len(rows[0]["question"]) == 200
    assert rows[0]["question"] == "x" * 200


def test_admin_rag_queries_clamps_limit_upper_bound(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    with SessionLocal() as session:
        for i in range(205):
            session.add(
                RagQuery(
                    id=str(uuid.uuid4()),
                    job_id=job_id,
                    question=f"q{i}",
                    top_k=8,
                    model="m",
                    outcome="answered",
                    retrieved_chunk_ids=[],
                    created_at=datetime.now(timezone.utc),
                )
            )
        session.commit()

    resp = client.get("/admin/rag_queries", params={"limit": 9999})
    assert resp.status_code == 200
    assert len(resp.json()) == 200


def test_admin_rag_queries_clamps_limit_lower_bound(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    for _ in range(3):
        _seed_rag_query(SessionLocal, job_id, "answered")

    resp = client.get("/admin/rag_queries", params={"limit": 0})
    assert resp.status_code == 200
    assert len(resp.json()) == 1

    resp = client.get("/admin/rag_queries", params={"limit": -10})
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_admin_rag_queries_honors_in_range_limit(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    for _ in range(5):
        _seed_rag_query(SessionLocal, job_id, "answered")

    resp = client.get("/admin/rag_queries", params={"limit": 3})
    assert resp.status_code == 200
    assert len(resp.json()) == 3


def test_admin_rag_queries_paginates_with_before_cursor(rag_env):
    client, SessionLocal = rag_env
    job_id = _seed_job(SessionLocal, status=JobStatus.COMPLETED)
    base = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    with SessionLocal() as session:
        for i in range(5):
            session.add(
                RagQuery(
                    id=str(uuid.uuid4()),
                    job_id=job_id,
                    question=f"q{i}",
                    top_k=8,
                    model="m",
                    outcome="answered",
                    retrieved_chunk_ids=[],
                    created_at=base + timedelta(minutes=i),
                )
            )
        session.commit()

    page1 = client.get("/admin/rag_queries", params={"limit": 2}).json()
    assert [r["question"] for r in page1] == ["q4", "q3"]

    cursor = page1[-1]["created_at"]
    page2 = client.get(
        "/admin/rag_queries", params={"limit": 2, "before": cursor}
    ).json()
    assert [r["question"] for r in page2] == ["q2", "q1"]

    cursor2 = page2[-1]["created_at"]
    page3 = client.get(
        "/admin/rag_queries", params={"limit": 2, "before": cursor2}
    ).json()
    assert [r["question"] for r in page3] == ["q0"]
