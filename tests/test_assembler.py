import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, call

import pytest

from worker.processors.assembler import _risk_level, run
from shared.models import Job, JobStage, JobStatus, RiskResult


@pytest.mark.parametrize(
    "score,expected",
    [
        (0, "low"),
        (1, "low"),
        (34, "low"),
        (35, "medium"),
        (50, "medium"),
        (64, "medium"),
        (65, "high"),
        (80, "high"),
        (100, "high"),
    ],
)
def test_risk_level_thresholds(score, expected):
    assert _risk_level(score) == expected


def _scored_chunk(
    index: int,
    clause_type: str,
    chunk_score: float,
    flags: list | None = None,
    text: str = "sample text",
    confidence: float = 0.9,
) -> dict:
    return {
        "chunk_id": str(uuid.uuid4()),
        "index": index,
        "clause_type": clause_type,
        "confidence": confidence,
        "tone_score": 0.5,
        "flags": flags or [],
        "chunk_score": chunk_score,
        "text": text,
    }


def test_overall_score_is_mean_of_chunk_scores(make_job, db_session, mock_storage):
    job = make_job(stage=JobStage.ASSEMBLY)
    scored = [
        _scored_chunk(0, "indemnification", 80.0),
        _scored_chunk(1, "governing law", 20.0),
    ]
    run(job, db_session, mock_storage, scored)

    result = db_session.query(RiskResult).filter(RiskResult.job_id == job.id).first()
    assert result.overall_score == 50


def test_overall_score_rounds_to_int(make_job, db_session, mock_storage):
    job = make_job(stage=JobStage.ASSEMBLY)
    scored = [
        _scored_chunk(0, "termination", 33.4),
        _scored_chunk(1, "warranty", 33.4),
        _scored_chunk(2, "confidentiality", 33.5),
    ]
    run(job, db_session, mock_storage, scored)

    result = db_session.query(RiskResult).filter(RiskResult.job_id == job.id).first()
    assert isinstance(result.overall_score, int)


def test_empty_chunk_list_produces_score_zero(make_job, db_session, mock_storage):
    job = make_job(stage=JobStage.ASSEMBLY)
    run(job, db_session, mock_storage, [])

    result = db_session.query(RiskResult).filter(RiskResult.job_id == job.id).first()
    assert result.overall_score == 0
    assert result.risk_level == "low"


def test_clause_summary_counts_each_type(make_job, db_session, mock_storage):
    job = make_job(stage=JobStage.ASSEMBLY)
    scored = [
        _scored_chunk(0, "indemnification", 70.0),
        _scored_chunk(1, "indemnification", 65.0),
        _scored_chunk(2, "termination", 40.0),
        _scored_chunk(3, "governing law", 10.0),
    ]
    run(job, db_session, mock_storage, scored)

    result = db_session.query(RiskResult).filter(RiskResult.job_id == job.id).first()
    assert result.clause_summary["indemnification"] == 2
    assert result.clause_summary["termination"] == 1
    assert result.clause_summary["governing law"] == 1


def _hit(rule_id: str, severity: str, reason: str, text: str, matched: str) -> dict:
    """Build a risk hit matching the new scorer output shape (post-spaCy migration)."""
    start = text.lower().find(matched.lower())
    assert start >= 0, f"matched text {matched!r} not in source text"
    return {
        "id": rule_id,
        "severity": severity,
        "reason": reason,
        "matched_text": matched,
        "start": start,
        "end": start + len(matched),
    }


def test_flags_are_collected_from_chunks(make_job, db_session, mock_storage):
    job = make_job(stage=JobStage.ASSEMBLY)
    text = (
        "The licensee shall indemnify and hold harmless the provider against claims."
    )
    scored = [
        _scored_chunk(
            0,
            "indemnification",
            75.0,
            flags=[
                _hit(
                    "broad_indemnification",
                    "high",
                    "Broad indemnification",
                    text,
                    "indemnify and hold harmless",
                )
            ],
            text=text,
        ),
    ]
    run(job, db_session, mock_storage, scored)

    result = db_session.query(RiskResult).filter(RiskResult.job_id == job.id).first()
    assert len(result.flags) == 1
    flag = result.flags[0]
    assert flag["rule_id"] == "broad_indemnification"
    assert flag["matched_text"] == "indemnify and hold harmless"
    assert flag["severity"] == "high"
    assert flag["chunk_index"] == 0
    assert flag["clause_type"] == "indemnification"


def test_flag_excerpt_contains_matched_pattern(make_job, db_session, mock_storage):
    job = make_job(stage=JobStage.ASSEMBLY)
    text = (
        "Both parties agree that automatic renewal applies unless written notice "
        "is given."
    )
    scored = [
        _scored_chunk(
            0,
            "termination",
            45.0,
            flags=[
                _hit(
                    "automatic_renewal",
                    "medium",
                    "Contract auto-renews",
                    text,
                    "automatic renewal",
                )
            ],
            text=text,
        ),
    ]
    run(job, db_session, mock_storage, scored)

    result = db_session.query(RiskResult).filter(RiskResult.job_id == job.id).first()
    excerpt = result.flags[0]["excerpt"]
    assert "automatic renewal" in excerpt.lower()


def test_chunk_with_multiple_flags_generates_multiple_flag_rows(
    make_job, db_session, mock_storage
):
    job = make_job(stage=JobStage.ASSEMBLY)
    text = (
        "At sole discretion, they shall indemnify and hold harmless, terminate "
        "without cause."
    )
    scored = [
        _scored_chunk(
            0,
            "indemnification",
            90.0,
            flags=[
                _hit(
                    "unilateral_discretion",
                    "high",
                    "Unilateral right",
                    text,
                    "sole discretion",
                ),
                _hit(
                    "broad_indemnification",
                    "high",
                    "Broad indemnification",
                    text,
                    "indemnify and hold harmless",
                ),
                _hit(
                    "termination_without_cause",
                    "medium",
                    "Termination without cause",
                    text,
                    "without cause",
                ),
            ],
            text=text,
        ),
    ]
    run(job, db_session, mock_storage, scored)

    result = db_session.query(RiskResult).filter(RiskResult.job_id == job.id).first()
    assert len(result.flags) == 3


def test_run_uploads_report_json_to_storage(make_job, db_session, mock_storage):
    job = make_job(stage=JobStage.ASSEMBLY)
    run(job, db_session, mock_storage, [_scored_chunk(0, "governing law", 15.0)])

    mock_storage.upload_json.assert_called_once()
    report_key, report_payload = mock_storage.upload_json.call_args[0]
    assert report_key.startswith("contracts/reports/")
    assert report_key.endswith(".json")


def test_report_payload_has_required_keys(make_job, db_session, mock_storage):
    job = make_job(stage=JobStage.ASSEMBLY)
    run(job, db_session, mock_storage, [_scored_chunk(0, "confidentiality", 30.0)])

    _, report = mock_storage.upload_json.call_args[0]
    for key in (
        "job_id",
        "filename",
        "overall_score",
        "risk_level",
        "clause_summary",
        "flags",
        "chunks",
    ):
        assert key in report, f"missing key in report: {key}"


def test_report_chunks_contain_expected_fields(make_job, db_session, mock_storage):
    job = make_job(stage=JobStage.ASSEMBLY)
    run(job, db_session, mock_storage, [_scored_chunk(0, "payment terms", 40.0)])

    _, report = mock_storage.upload_json.call_args[0]
    chunk_entry = report["chunks"][0]
    for field in ("index", "clause_type", "confidence", "score", "text"):
        assert field in chunk_entry


def test_run_marks_job_completed(make_job, db_session, mock_storage):
    job = make_job(stage=JobStage.ASSEMBLY)
    run(job, db_session, mock_storage, [])
    assert job.status == JobStatus.COMPLETED


def test_run_advances_stage_to_done(make_job, db_session, mock_storage):
    job = make_job(stage=JobStage.ASSEMBLY)
    run(job, db_session, mock_storage, [])
    assert job.stage == JobStage.DONE


def test_run_persists_risk_result(make_job, db_session, mock_storage):
    job = make_job(stage=JobStage.ASSEMBLY)
    run(job, db_session, mock_storage, [_scored_chunk(0, "indemnification", 80.0)])

    result = db_session.query(RiskResult).filter(RiskResult.job_id == job.id).first()
    assert result is not None
    assert result.job_id == job.id
    assert result.risk_level in ("low", "medium", "high")
