"""
_apply_risk_patterns and _to_tone_score are pure functions with no DB or pipeline
overhead, so they are tested exhaustively at the unit level.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from worker.processors.scorer import (
    _apply_risk_patterns,
    _to_tone_score,
    score_chunks,
    run,
    RISK_PATTERNS,
    CLAUSE_WEIGHTS,
    _SEVERITY_SCORES,
)
from shared.models import Chunk, Job, JobStage, JobStatus
from tests.conftest import make_tone_pipeline


@pytest.mark.parametrize(
    "pattern,expected_severity",
    [
        ("sole discretion", "high"),
        ("unlimited liability", "high"),
        ("perpetual and irrevocable", "high"),
        ("indemnify and hold harmless", "high"),
        ("without cause", "medium"),
        ("automatic renewal", "medium"),
        ("liquidated damages", "medium"),
        ("best efforts", "low"),
        ("reasonable notice", "low"),
    ],
)
def test_pattern_detected_with_correct_severity(pattern, expected_severity):
    hits = _apply_risk_patterns(f"This agreement includes {pattern} as a provision.")
    assert len(hits) == 1
    assert hits[0]["severity"] == expected_severity
    assert hits[0]["pattern"] == pattern


def test_pattern_matching_is_case_insensitive():
    hits = _apply_risk_patterns("Party A shall INDEMNIFY AND HOLD HARMLESS Party B.")
    assert any(h["pattern"] == "indemnify and hold harmless" for h in hits)


def test_no_patterns_in_clean_text():
    hits = _apply_risk_patterns("Both parties agree to act in good faith at all times.")
    assert hits == []


def test_multiple_patterns_detected_in_one_chunk():
    text = (
        "The contract shall renew via automatic renewal. "
        "Termination without cause is permitted. "
        "Liquidated damages apply to late payment."
    )
    hits = _apply_risk_patterns(text)
    patterns_found = {h["pattern"] for h in hits}
    assert "automatic renewal" in patterns_found
    assert "without cause" in patterns_found
    assert "liquidated damages" in patterns_found


def test_each_pattern_includes_reason_field():
    hits = _apply_risk_patterns("The contract uses sole discretion for all decisions.")
    assert hits[0]["reason"] == "Unilateral decision right"


def test_all_defined_patterns_are_detectable():
    for pattern, severity, reason in RISK_PATTERNS:
        hits = _apply_risk_patterns(f"Text containing {pattern} here.")
        assert any(h["pattern"] == pattern for h in hits), (
            f"pattern not detected: {pattern!r}"
        )


def test_tone_negative_label_returns_high_score():
    result = _to_tone_score({"label": "NEGATIVE", "score": 0.9})
    assert result == pytest.approx(0.9)


def test_tone_positive_label_returns_low_score():
    result = _to_tone_score({"label": "POSITIVE", "score": 0.85})
    assert result == pytest.approx(1.0 - 0.85)


def test_tone_score_bounded_between_zero_and_one():
    for label, score in [("NEGATIVE", 1.0), ("POSITIVE", 1.0), ("NEGATIVE", 0.0)]:
        result = _to_tone_score({"label": label, "score": score})
        assert 0.0 <= result <= 1.0


def _make_chunk_obj(clause_type: str, text: str, index: int = 0) -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        index=index,
        text=text,
        token_count=len(text.split()),
        clause_type=clause_type,
        confidence=0.9,
    )


def test_score_chunks_returns_one_entry_per_chunk():
    chunks = [
        _make_chunk_obj("indemnification", "indemnify and hold harmless", index=0),
        _make_chunk_obj("governing law", "governed by Delaware law.", index=1),
    ]
    scored = score_chunks(chunks, make_tone_pipeline())
    assert len(scored) == 2


def test_score_chunks_includes_required_fields():
    chunks = [_make_chunk_obj("termination", "without cause termination.", index=0)]
    result = score_chunks(chunks, make_tone_pipeline())[0]

    for field in (
        "chunk_id",
        "index",
        "clause_type",
        "confidence",
        "tone_score",
        "flags",
        "chunk_score",
        "text",
    ):
        assert field in result, f"missing field: {field}"


def test_high_risk_clause_type_produces_higher_score():
    indemnity = _make_chunk_obj("indemnification", "standard clause text", index=0)
    gov_law = _make_chunk_obj("governing law", "standard clause text", index=1)

    tone = make_tone_pipeline(label="POSITIVE", score=0.9)
    scored = score_chunks([indemnity, gov_law], tone)

    assert scored[0]["chunk_score"] > scored[1]["chunk_score"]


def test_high_severity_flag_raises_chunk_score():
    """High-severity patterns contribute 0.4× to the score formula; the flag alone can swing the result."""
    flagged = _make_chunk_obj("general", "The party has sole discretion.", index=0)
    unflagged = _make_chunk_obj("general", "The party has some discretion.", index=1)

    tone = make_tone_pipeline(label="POSITIVE", score=0.9)
    scored = score_chunks([flagged, unflagged], tone)
    assert scored[0]["chunk_score"] > scored[1]["chunk_score"]


def test_score_formula_manual_calculation():
    tone = make_tone_pipeline(label="NEGATIVE", score=0.8)
    chunk = _make_chunk_obj(
        "indemnification", "The party has sole discretion in all matters.", index=0
    )

    scored = score_chunks([chunk], tone)
    result = scored[0]["chunk_score"]

    expected = (0.8 * 0.3 + 1.0 * 0.4 + 1.0 * 0.3) * 100
    assert result == pytest.approx(expected, abs=0.01)


def test_chunk_with_no_flags_and_positive_tone_is_low_risk():
    tone = make_tone_pipeline(label="POSITIVE", score=0.95)
    chunk = _make_chunk_obj(
        "governing law", "Governed by the laws of Delaware.", index=0
    )
    scored = score_chunks([chunk], tone)
    assert scored[0]["chunk_score"] < 35


def test_run_advances_stage_to_assembly(make_job, make_chunk, db_session):
    job = make_job(stage=JobStage.SCORING)
    make_chunk(
        job.id,
        "Payment terms clause.",
        index=0,
        clause_type="payment terms",
        confidence=0.8,
    )

    run(job, db_session, make_tone_pipeline())

    assert job.stage == JobStage.ASSEMBLY


def test_run_returns_scored_chunk_list(make_job, make_chunk, db_session):
    job = make_job(stage=JobStage.SCORING)
    make_chunk(
        job.id,
        "Indemnify and hold harmless.",
        index=0,
        clause_type="indemnification",
        confidence=0.9,
    )
    make_chunk(
        job.id,
        "Best efforts shall apply.",
        index=1,
        clause_type="warranty",
        confidence=0.7,
    )

    scored = run(job, db_session, make_tone_pipeline())

    assert len(scored) == 2
    assert all("chunk_score" in s for s in scored)


def test_run_empty_job_returns_empty_list(make_job, db_session):
    job = make_job(stage=JobStage.SCORING)
    scored = run(job, db_session, make_tone_pipeline())
    assert scored == []
    assert job.stage == JobStage.ASSEMBLY
