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
    CLAUSE_WEIGHTS,
    _SEVERITY_SCORES,
)
from worker.processors.risk_rules import default_matcher
from shared.models import Chunk, Job, JobStage, JobStatus
from tests.conftest import make_tone_pipeline


@pytest.mark.parametrize(
    "text,expected_rule_id,expected_severity",
    [
        ("The party has sole discretion.", "unilateral_discretion", "high"),
        (
            "The party has sole and absolute discretion.",
            "unilateral_discretion",
            "high",
        ),
        ("Unlimited liability applies here.", "unlimited_liability", "high"),
        (
            "This is perpetual and irrevocable by nature.",
            "perpetual_irrevocable",
            "high",
        ),
        (
            "Party A shall indemnify and hold harmless Party B.",
            "broad_indemnification",
            "high",
        ),
        ("Termination without cause is allowed.", "termination_without_cause", "medium"),
        ("Contract renews via automatic renewal.", "automatic_renewal", "medium"),
        ("Liquidated damages apply here.", "liquidated_damages", "medium"),
        ("Party shall use best efforts.", "best_efforts", "low"),
        ("Provide reasonable notice prior.", "reasonable_notice", "low"),
    ],
)
def test_rule_detected_with_correct_severity(text, expected_rule_id, expected_severity):
    hits = _apply_risk_patterns(text)
    assert any(
        h["id"] == expected_rule_id and h["severity"] == expected_severity for h in hits
    ), f"rule {expected_rule_id} not detected in: {text!r}"


def test_pattern_matching_is_case_insensitive():
    hits = _apply_risk_patterns("Party A shall INDEMNIFY AND HOLD HARMLESS Party B.")
    assert any(h["id"] == "broad_indemnification" for h in hits)


def test_no_patterns_in_clean_text():
    hits = _apply_risk_patterns("Both parties agree to act in good faith at all times.")
    assert hits == []


def test_multiple_patterns_detected_in_one_chunk():
    text = (
        "The contract shall renew via automatic renewal. "
        "Termination without cause is permitted. "
        "Liquidated damages apply to late payment."
    )
    ids = {h["id"] for h in _apply_risk_patterns(text)}
    assert "automatic_renewal" in ids
    assert "termination_without_cause" in ids
    assert "liquidated_damages" in ids


def test_each_hit_includes_reason_and_matched_span():
    hits = _apply_risk_patterns("The party has sole discretion over all matters.")
    assert hits[0]["reason"] == "Unilateral decision right"
    assert hits[0]["matched_text"].lower() == "sole discretion"
    # Offsets point into the input so the assembler can build an excerpt.
    assert hits[0]["start"] >= 0
    assert hits[0]["end"] > hits[0]["start"]


def test_all_defined_rules_are_detectable():
    matcher = default_matcher()
    # Each rule's patterns must compile and be reachable; this just ensures the
    # rule set loaded without error.
    assert len(matcher.rule_ids) >= 9


def test_rule_handles_lexical_variants():
    # The hardcoded substring matcher would miss these; the DSL must catch them.
    variants = [
        "The party has sole, exclusive discretion over renewals.",
        "The party has sole and unfettered discretion.",
        "Party shall defend, indemnify and hold harmless the vendor.",
        "Contract automatically renews each year.",
    ]
    for text in variants:
        assert _apply_risk_patterns(text), f"no hit for variant: {text!r}"


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
