import io
import logging
import re
import uuid
from datetime import datetime, timezone

import pypdf
from sqlalchemy.orm import Session

from shared.minio_client import StorageClient
from shared.models import Chunk, Job, JobStage

logger = logging.getLogger(__name__)

_NUMBERED_CLAUSE = re.compile(
    r"^(\d+\.\d+|\d+\.|Article\s+[IVXivx\d]+)\s", re.MULTILINE
)
_CAPS_HEADING = re.compile(r"^([A-Z][A-Z\s]{3,})\s*$", re.MULTILINE)

_MAX_TOKENS = 400


def _approx_tokens(text: str) -> int:
    return len(text.split())


def _split_oversized(text: str, max_tokens: int = _MAX_TOKENS) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)
    ]


def _split_by_pattern(pattern: re.Pattern, full_text: str) -> list[str] | None:
    parts = pattern.split(full_text)
    if len(parts) <= 2:
        return None
    sections = []
    if parts[0].strip():
        sections.append(parts[0].strip())
    for i in range(1, len(parts) - 1, 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if heading or body:
            sections.append((heading + " " + body).strip())
    return sections


def _chunk_text(full_text: str) -> list[str]:
    sections = (
        _split_by_pattern(_NUMBERED_CLAUSE, full_text)
        or _split_by_pattern(_CAPS_HEADING, full_text)
        or [s.strip() for s in re.split(r"\n\n+", full_text) if s.strip()]
    )

    final: list[str] = []
    for section in sections:
        if _approx_tokens(section) > _MAX_TOKENS:
            final.extend(_split_oversized(section))
        else:
            final.append(section)

    return [s for s in final if s.strip()]


def run(job: Job, db: Session, storage: StorageClient) -> None:
    logger.info("ingestion: job=%s key=%s", job.id, job.object_key)

    pdf_bytes = storage.download_bytes(job.object_key)
    if not pdf_bytes:
        raise ValueError(f"downloaded object is empty: {job.object_key}")

    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise ValueError(f"failed to parse PDF {job.object_key}: {exc}") from exc

    full_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

    if not full_text.strip():
        raise ValueError(f"no text could be extracted from PDF {job.object_key}")

    sections = _chunk_text(full_text)

    if not sections:
        raise ValueError(
            "chunking produced no output - PDF may contain only images or non-text content"
        )

    logger.info("ingestion: %d chunks produced", len(sections))

    for idx, text in enumerate(sections):
        chunk = Chunk(
            id=str(uuid.uuid4()),
            job_id=job.id,
            index=idx,
            text=text,
            token_count=_approx_tokens(text),
            created_at=datetime.now(timezone.utc),
        )
        db.add(chunk)

    job.stage = JobStage.CLASSIFICATION
    job.updated_at = datetime.now(timezone.utc)
    db.commit()
    logger.info("ingestion: done, stage → classification")
