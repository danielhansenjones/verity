import uuid
from datetime import datetime, timezone

import pytest

from api.llm import (
    AnswerResponse,
    Citation,
    _normalize,
    build_user_prompt,
    grounding_error,
)
from shared.models import Chunk


def _chunk(index: int, text: str, job_id: str = "job-1") -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        job_id=job_id,
        index=index,
        text=text,
        token_count=len(text.split()),
        created_at=datetime.now(timezone.utc),
    )


def test_build_user_prompt_includes_chunk_markers():
    chunks = [
        _chunk(0, "Delaware law governs this agreement."),
        _chunk(1, "Either party may terminate with notice."),
    ]
    prompt = build_user_prompt("What is the governing law?", chunks)

    assert "What is the governing law?" in prompt
    assert f"[chunk_id={chunks[0].id} chunk_index=0]" in prompt
    assert f"[chunk_id={chunks[1].id} chunk_index=1]" in prompt
    assert "Delaware law" in prompt


def test_normalize_collapses_whitespace_and_lowercases():
    assert _normalize("Hello   World\n\nfoo") == "hello world foo"


def test_grounding_error_passes_when_quote_is_substring():
    chunk = _chunk(0, "Delaware law governs this agreement.")
    response = AnswerResponse(
        answer="Delaware law governs.",
        citations=[
            Citation(chunk_id=chunk.id, chunk_index=0, quote="Delaware law governs")
        ],
    )

    assert grounding_error(response, [chunk]) is None


def test_grounding_error_passes_with_whitespace_variations():
    chunk = _chunk(0, "Delaware law\ngoverns this  agreement.")
    response = AnswerResponse(
        answer="Delaware law governs.",
        citations=[
            Citation(
                chunk_id=chunk.id,
                chunk_index=0,
                quote="Delaware law governs this agreement",
            )
        ],
    )

    assert grounding_error(response, [chunk]) is None


def test_grounding_error_flags_unknown_chunk_id():
    chunk = _chunk(0, "Delaware law governs.")
    response = AnswerResponse(
        answer="Delaware law governs.",
        citations=[
            Citation(
                chunk_id="fabricated-id",
                chunk_index=99,
                quote="Delaware law governs",
            )
        ],
    )

    err = grounding_error(response, [chunk])
    assert err is not None
    assert "fabricated-id" in err


def test_grounding_error_flags_fabricated_quote():
    chunk = _chunk(0, "Delaware law governs.")
    response = AnswerResponse(
        answer="Delaware caps liability at five million dollars.",
        citations=[
            Citation(
                chunk_id=chunk.id,
                chunk_index=0,
                quote="liability cap of five million dollars",
            )
        ],
    )

    err = grounding_error(response, [chunk])
    assert err is not None
    assert "quote not found" in err


def test_grounding_error_flags_refusal_with_citations():
    chunk = _chunk(0, "Delaware law governs.")
    response = AnswerResponse(
        answer="",
        citations=[Citation(chunk_id=chunk.id, chunk_index=0, quote="Delaware")],
        refusal_reason="no evidence",
    )

    err = grounding_error(response, [chunk])
    assert err == "refusal response must not include citations"


def test_grounding_error_flags_refusal_with_nonempty_answer():
    response = AnswerResponse(
        answer="The cap is unclear.",
        citations=[],
        refusal_reason="no evidence",
    )

    err = grounding_error(response, [])
    assert err == "refusal response must have an empty answer"


def test_grounding_error_passes_for_clean_refusal():
    response = AnswerResponse(answer="", citations=[], refusal_reason="no evidence")

    assert grounding_error(response, []) is None


@pytest.mark.parametrize(
    "answer,reason",
    [
        ("some answer", None),
        ("", "no evidence in the provided chunks"),
    ],
)
def test_answer_response_round_trips_through_pydantic(answer, reason):
    response = AnswerResponse(answer=answer, citations=[], refusal_reason=reason)
    dumped = response.model_dump()
    restored = AnswerResponse.model_validate(dumped)
    assert restored == response
