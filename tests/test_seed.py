import io
from pathlib import Path

import pypdf
import pytest

from worker.processors.ingestion import _chunk_text, _approx_tokens, _MAX_TOKENS
from worker.processors.scorer import _apply_risk_patterns

TEST_DOCUMENTS_DIR = Path(__file__).parent / "test_documents"


def test_make_pdf_bytes_starts_with_pdf_header(make_pdf_bytes):
    data = make_pdf_bytes("Hello world.")
    assert data[:4] == b"%PDF"


def test_make_pdf_bytes_is_pypdf_parseable(make_pdf_bytes):
    data = make_pdf_bytes("This is a contract clause about indemnification.")
    reader = pypdf.PdfReader(io.BytesIO(data))
    assert len(reader.pages) > 0


def test_make_pdf_bytes_text_is_extractable(make_pdf_bytes):
    text = "CONFIDENTIALITY CLAUSE - The parties agree to maintain secrecy."
    data = make_pdf_bytes(text)
    reader = pypdf.PdfReader(io.BytesIO(data))
    extracted = " ".join(page.extract_text() or "" for page in reader.pages)
    # pypdf may alter whitespace / encoding slightly; check key terms
    assert "CONFIDENTIALITY" in extracted


def test_make_pdf_bytes_empty_text_does_not_raise(make_pdf_bytes):
    data = make_pdf_bytes("")
    assert data[:4] == b"%PDF"


def test_two_different_texts_produce_different_pdfs(make_pdf_bytes):
    pdf_a = make_pdf_bytes("Clause A about indemnification.")
    pdf_b = make_pdf_bytes("Clause B about governing law.")
    assert pdf_a != pdf_b


def test_nda_sample_extracts_expected_keywords(make_pdf_bytes, seed_contracts):
    nda_filename, nda_text = seed_contracts[0]
    assert "nda" in nda_filename.lower()

    data = make_pdf_bytes(nda_text)
    reader = pypdf.PdfReader(io.BytesIO(data))
    extracted = " ".join(page.extract_text() or "" for page in reader.pages).lower()

    for keyword in (
        "confidential",
        "termination",
        "governing law",
        "intellectual property",
    ):
        assert keyword in extracted, f"expected keyword missing from NDA: {keyword!r}"


def test_saas_sample_extracts_expected_keywords(make_pdf_bytes, seed_contracts):
    saas_filename, saas_text = seed_contracts[1]
    assert "saas" in saas_filename.lower()

    data = make_pdf_bytes(saas_text)
    reader = pypdf.PdfReader(io.BytesIO(data))
    extracted = " ".join(page.extract_text() or "" for page in reader.pages).lower()

    for keyword in ("payment", "warranty", "indemnif", "force majeure"):
        assert keyword in extracted, f"expected keyword missing from SaaS: {keyword!r}"


def test_nda_text_triggers_high_risk_patterns(seed_contracts):
    _, nda_text = seed_contracts[0]
    hits = _apply_risk_patterns(nda_text)
    high_hits = [h for h in hits if h["severity"] == "high"]
    assert len(high_hits) >= 2, (
        f"NDA sample should contain at least 2 high-severity patterns, found: {high_hits}"
    )


def test_saas_text_triggers_risk_patterns(seed_contracts):
    _, saas_text = seed_contracts[1]
    hits = _apply_risk_patterns(saas_text)
    assert len(hits) >= 2, (
        f"SaaS sample should trigger at least 2 risk patterns, found: {hits}"
    )


def test_nda_text_chunks_into_multiple_sections(seed_contracts):
    _, nda_text = seed_contracts[0]
    chunks = _chunk_text(nda_text)
    assert len(chunks) >= 3, "NDA has 6 numbered sections - expect at least 3 chunks"


def test_saas_text_chunks_into_multiple_sections(seed_contracts):
    _, saas_text = seed_contracts[1]
    chunks = _chunk_text(saas_text)
    assert len(chunks) >= 3, "SaaS contract has 5 Articles - expect at least 3 chunks"


def test_both_contracts_produce_distinct_chunk_sets(seed_contracts):
    _, nda_text = seed_contracts[0]
    _, saas_text = seed_contracts[1]
    nda_chunks = set(_chunk_text(nda_text))
    saas_chunks = set(_chunk_text(saas_text))
    assert nda_chunks != saas_chunks


@pytest.mark.parametrize("pdf_filename", ["Document.pdf", "Document4.pdf"])
def test_real_document_is_a_valid_pdf(pdf_filename):
    path = TEST_DOCUMENTS_DIR / pdf_filename
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(io.BytesIO(f.read()))
    assert len(reader.pages) > 0


@pytest.mark.parametrize("pdf_filename", ["Document.pdf", "Document4.pdf"])
def test_real_document_contains_contract_language(pdf_filename):
    path = TEST_DOCUMENTS_DIR / pdf_filename
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(io.BytesIO(f.read()))
    text = " ".join(page.extract_text() or "" for page in reader.pages).lower()

    contract_terms = ["agreement", "party", "shall", "pursuant"]
    found = [t for t in contract_terms if t in text]
    assert len(found) >= 3, (
        f"Expected contract language in {pdf_filename}, found only: {found}"
    )


@pytest.mark.parametrize("pdf_filename", ["Document.pdf", "Document4.pdf"])
def test_real_document_risk_pattern_scan_runs_cleanly(pdf_filename):
    """Financial credit-agreement vocabulary differs from consumer-contract risk patterns, so match count is not meaningful here."""
    path = TEST_DOCUMENTS_DIR / pdf_filename
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(io.BytesIO(f.read()))
    text = " ".join(page.extract_text() or "" for page in reader.pages)

    hits = _apply_risk_patterns(text)

    assert isinstance(hits, list)
    for hit in hits:
        assert "pattern" in hit
        assert "severity" in hit
        assert "reason" in hit


@pytest.mark.parametrize("pdf_filename", ["Document.pdf", "Document4.pdf"])
def test_real_document_chunks_stay_within_token_limit(pdf_filename):
    path = TEST_DOCUMENTS_DIR / pdf_filename
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(io.BytesIO(f.read()))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

    chunks = _chunk_text(text)
    oversized = [c for c in chunks if _approx_tokens(c) > _MAX_TOKENS]
    assert oversized == [], (
        f"{len(oversized)} chunks in {pdf_filename} exceeded {_MAX_TOKENS}-token limit"
    )


@pytest.mark.parametrize("pdf_filename", ["Document.pdf", "Document4.pdf"])
def test_real_document_produces_many_chunks(pdf_filename):
    """Multi-page documents should produce far more chunks than a short synthetic contract."""
    path = TEST_DOCUMENTS_DIR / pdf_filename
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(io.BytesIO(f.read()))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

    chunks = _chunk_text(text)
    assert len(chunks) > 10, (
        f"Expected many chunks from {pdf_filename}, got {len(chunks)}"
    )
