import uuid
from datetime import datetime, timezone

import pytest

from worker.processors.classifier import run, CLAUSE_LABELS, _CONFIDENCE_THRESHOLD
from shared.models import Chunk, Job, JobStage, JobStatus
from tests.conftest import make_classifier_pipeline


def test_high_confidence_assigns_top_label(make_job, make_chunk, db_session):
    job = make_job(stage=JobStage.CLASSIFICATION)
    make_chunk(
        job.id, "The licensee shall indemnify and hold harmless the provider.", index=0
    )

    classifier = make_classifier_pipeline(label="indemnification", score=0.92)
    run(job, db_session, classifier)

    chunk = db_session.query(Chunk).filter(Chunk.job_id == job.id).first()
    assert chunk.clause_type == "indemnification"
    assert chunk.confidence == pytest.approx(0.92)


def test_below_threshold_assigns_general(make_job, make_chunk, db_session):
    job = make_job(stage=JobStage.CLASSIFICATION)
    make_chunk(job.id, "Some generic clause text that is hard to classify.", index=0)

    classifier = make_classifier_pipeline(label="warranty", score=0.35)
    run(job, db_session, classifier)

    chunk = db_session.query(Chunk).filter(Chunk.job_id == job.id).first()
    assert chunk.clause_type == "general"


def test_exactly_at_threshold_assigns_label(make_job, make_chunk, db_session):
    job = make_job(stage=JobStage.CLASSIFICATION)
    make_chunk(job.id, "Termination clause text.", index=0)

    classifier = make_classifier_pipeline(
        label="termination", score=_CONFIDENCE_THRESHOLD
    )
    run(job, db_session, classifier)

    chunk = db_session.query(Chunk).filter(Chunk.job_id == job.id).first()
    assert chunk.clause_type == "termination"


def test_all_chunks_are_classified(make_job, make_chunk, db_session):
    job = make_job(stage=JobStage.CLASSIFICATION)
    for i in range(5):
        make_chunk(job.id, f"Clause text number {i}.", index=i)

    run(job, db_session, make_classifier_pipeline())

    chunks = db_session.query(Chunk).filter(Chunk.job_id == job.id).all()
    assert len(chunks) == 5
    assert all(c.clause_type is not None for c in chunks)
    assert all(c.confidence is not None for c in chunks)


def test_batch_boundary_nine_chunks_requires_two_batches(
    make_job, make_chunk, db_session
):
    job = make_job(stage=JobStage.CLASSIFICATION)
    for i in range(9):
        make_chunk(job.id, f"Clause {i} text.", index=i)

    call_count = {"n": 0}
    base = make_classifier_pipeline()

    def counting_pipeline(texts, candidate_labels=None, batch_size=None):
        call_count["n"] += 1
        return base(texts, candidate_labels=candidate_labels, batch_size=batch_size)

    run(job, db_session, counting_pipeline)

    assert call_count["n"] == 2


def test_single_chunk_pipeline_result_coerced_correctly(
    make_job, make_chunk, db_session
):
    """HuggingFace pipeline returns a bare dict instead of a list for single-item inputs."""
    job = make_job(stage=JobStage.CLASSIFICATION)
    make_chunk(job.id, "Only one clause in this document.", index=0)

    def single_dict_pipeline(texts, candidate_labels=None, batch_size=None):
        labels = candidate_labels or []
        return {
            "labels": ["confidentiality"]
            + [l for l in labels if l != "confidentiality"],
            "scores": [0.88] + [0.01] * (len(labels) - 1),
        }

    run(job, db_session, single_dict_pipeline)

    chunk = db_session.query(Chunk).filter(Chunk.job_id == job.id).first()
    assert chunk.clause_type == "confidentiality"


def test_stage_advances_to_scoring(make_job, make_chunk, db_session):
    job = make_job(stage=JobStage.CLASSIFICATION)
    make_chunk(job.id, "Payment terms clause.", index=0)

    run(job, db_session, make_classifier_pipeline())

    assert job.stage == JobStage.SCORING


def test_empty_chunk_list_still_advances_stage(make_job, db_session):
    """Jobs with zero chunks are unusual but possible if text extraction yields nothing."""
    job = make_job(stage=JobStage.CLASSIFICATION)
    run(job, db_session, make_classifier_pipeline())
    assert job.stage == JobStage.SCORING


def test_all_clause_labels_can_be_assigned(make_job, make_chunk, db_session):
    for label in CLAUSE_LABELS:
        job = make_job(stage=JobStage.CLASSIFICATION)
        make_chunk(job.id, f"A clause about {label}.", index=0)

        run(job, db_session, make_classifier_pipeline(label=label, score=0.9))

        chunk = db_session.query(Chunk).filter(Chunk.job_id == job.id).first()
        assert chunk.clause_type == label
        db_session.rollback()
