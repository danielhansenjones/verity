import io
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pypdf
import pytest

from worker.processors.ingestion import (
    _approx_tokens,
    _chunk_text,
    _split_by_pattern,
    _split_oversized,
    run,
    _NUMBERED_CLAUSE,
    _CAPS_HEADING,
    _MAX_TOKENS,
)
from shared.models import Chunk, Job, JobStage, JobStatus

TEST_DOCUMENTS_DIR = Path(__file__).parent / "test_documents"

NDA_TEXT = """\
NON-DISCLOSURE AGREEMENT

1. CONFIDENTIALITY OBLIGATIONS
The Receiving Party agrees to keep all Confidential Information strictly
confidential and shall indemnify and hold harmless the Disclosing Party.

1.1 These obligations are perpetual and irrevocable.

2. TERM AND TERMINATION
Either party may terminate this Agreement without cause upon reasonable notice.

3. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware.
"""

SAAS_TEXT = """\
SOFTWARE AS A SERVICE AGREEMENT

Article I - PAYMENT TERMS
Customer agrees to pay all fees. Liquidated damages of 1.5% per month apply.

Article II - WARRANTY
Provider warrants performance using best efforts standards.

Article III - INDEMNIFICATION
Customer shall indemnify and hold harmless Provider against third-party claims.
"""


def test_approx_tokens_counts_whitespace_separated_words():
    assert _approx_tokens("one two three") == 3


def test_approx_tokens_empty_string():
    assert _approx_tokens("") == 0


def test_approx_tokens_single_word():
    assert _approx_tokens("indemnification") == 1


def test_split_oversized_produces_chunks_within_limit():
    words = ["word"] * 900
    text = " ".join(words)
    chunks = _split_oversized(text, max_tokens=400)
    assert len(chunks) == 3
    for chunk in chunks:
        assert _approx_tokens(chunk) <= 400


def test_split_oversized_small_text_returns_single_chunk():
    text = "short text here"
    chunks = _split_oversized(text, max_tokens=400)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_oversized_exact_boundary():
    words = ["word"] * 400
    chunks = _split_oversized(" ".join(words), max_tokens=400)
    assert len(chunks) == 1


def test_chunk_text_splits_on_numbered_clauses():
    chunks = _chunk_text(NDA_TEXT)
    assert len(chunks) >= 3
    joined = " ".join(chunks)
    assert "CONFIDENTIALITY" in joined
    assert "TERMINATION" in joined
    assert "GOVERNING LAW" in joined


def test_chunk_text_splits_on_article_headings():
    chunks = _chunk_text(SAAS_TEXT)
    assert len(chunks) >= 3
    joined = " ".join(chunks)
    assert "PAYMENT TERMS" in joined
    assert "WARRANTY" in joined
    assert "INDEMNIFICATION" in joined


def test_chunk_text_falls_back_to_paragraph_split():
    text = "First paragraph with some content.\n\nSecond paragraph about liability.\n\nThird paragraph."
    chunks = _chunk_text(text)
    assert len(chunks) == 3


def test_chunk_text_filters_empty_results():
    text = "\n\n\n\nActual content here.\n\n\n\n"
    chunks = _chunk_text(text)
    assert all(c.strip() for c in chunks)


def test_chunk_text_oversized_section_is_further_split():
    long_body = " ".join(["word"] * (_MAX_TOKENS + 50))
    text = f"1. LONG CLAUSE\n{long_body}"
    chunks = _chunk_text(text)
    for chunk in chunks:
        assert _approx_tokens(chunk) <= _MAX_TOKENS


def test_split_by_pattern_returns_none_when_no_match():
    result = _split_by_pattern(_NUMBERED_CLAUSE, "No numbered clauses here at all.")
    assert result is None


def test_split_by_pattern_returns_sections_when_pattern_found():
    result = _split_by_pattern(_NUMBERED_CLAUSE, NDA_TEXT)
    assert result is not None
    assert len(result) >= 2


def test_run_stores_chunks_in_db(make_job, db_session, mock_storage, make_pdf_bytes):
    nda_pdf = make_pdf_bytes(NDA_TEXT)
    mock_storage.download_bytes.return_value = nda_pdf

    job = make_job(stage=JobStage.INGESTION)
    run(job, db_session, mock_storage)

    chunks = db_session.query(Chunk).filter(Chunk.job_id == job.id).all()
    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.token_count > 0
        assert chunk.index >= 0


def test_run_advances_stage_to_classification(
    make_job, db_session, mock_storage, make_pdf_bytes
):
    mock_storage.download_bytes.return_value = make_pdf_bytes(NDA_TEXT)

    job = make_job(stage=JobStage.INGESTION)
    run(job, db_session, mock_storage)

    assert job.stage == JobStage.CLASSIFICATION


def test_run_chunk_indices_are_sequential(
    make_job, db_session, mock_storage, make_pdf_bytes
):
    mock_storage.download_bytes.return_value = make_pdf_bytes(SAAS_TEXT)

    job = make_job(stage=JobStage.INGESTION)
    run(job, db_session, mock_storage)

    chunks = (
        db_session.query(Chunk)
        .filter(Chunk.job_id == job.id)
        .order_by(Chunk.index)
        .all()
    )
    indices = [c.index for c in chunks]
    assert indices == list(range(len(indices)))


def test_run_downloads_from_correct_object_key(
    make_job, db_session, mock_storage, make_pdf_bytes
):
    mock_storage.download_bytes.return_value = make_pdf_bytes(NDA_TEXT)

    job = make_job(stage=JobStage.INGESTION)
    run(job, db_session, mock_storage)

    mock_storage.download_bytes.assert_called_once_with(job.object_key)


@pytest.mark.parametrize("pdf_filename", ["Document.pdf", "Document4.pdf"])
def test_real_pdf_is_parseable_by_pypdf(pdf_filename):
    pdf_path = TEST_DOCUMENTS_DIR / pdf_filename
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(io.BytesIO(f.read()))
    assert len(reader.pages) > 0


@pytest.mark.parametrize("pdf_filename", ["Document.pdf", "Document4.pdf"])
def test_real_pdf_produces_non_empty_chunks(pdf_filename):
    pdf_path = TEST_DOCUMENTS_DIR / pdf_filename
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(io.BytesIO(f.read()))
    full_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

    chunks = _chunk_text(full_text)

    assert len(chunks) > 0
    assert all(c.strip() for c in chunks)


@pytest.mark.parametrize("pdf_filename", ["Document.pdf", "Document4.pdf"])
def test_real_pdf_chunks_respect_token_limit(pdf_filename):
    pdf_path = TEST_DOCUMENTS_DIR / pdf_filename
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(io.BytesIO(f.read()))
    full_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

    chunks = _chunk_text(full_text)

    oversized = [c for c in chunks if _approx_tokens(c) > _MAX_TOKENS]
    assert oversized == [], f"{len(oversized)} chunks exceeded {_MAX_TOKENS} tokens"


@pytest.mark.parametrize("pdf_filename", ["Document.pdf", "Document4.pdf"])
def test_real_pdf_run_ingestion_end_to_end(
    pdf_filename, make_job, db_session, mock_storage
):
    pdf_path = TEST_DOCUMENTS_DIR / pdf_filename
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    mock_storage.download_bytes.return_value = pdf_bytes

    job = make_job(stage=JobStage.INGESTION, filename=pdf_filename)
    run(job, db_session, mock_storage)

    chunks = db_session.query(Chunk).filter(Chunk.job_id == job.id).all()
    assert len(chunks) > 0
    assert job.stage == JobStage.CLASSIFICATION
