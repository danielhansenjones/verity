from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from shared.settings import settings
from shared.types import Vector


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStage(str, enum.Enum):
    INGESTION = "ingestion"
    CLASSIFICATION = "classification"
    SCORING = "scoring"
    ASSEMBLY = "assembly"
    DONE = "done"


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.QUEUED)
    stage: Mapped[JobStage] = mapped_column(Enum(JobStage), default=JobStage.INGESTION)
    object_key: Mapped[str] = mapped_column(String)
    filename: Mapped[str] = mapped_column(String)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
    )

    chunks: Mapped[list[Chunk]] = relationship(
        "Chunk", back_populates="job", order_by="Chunk.index"
    )
    result: Mapped[Optional[RiskResult]] = relationship(
        "RiskResult", back_populates="job", uselist=False
    )

    def __repr__(self) -> str:
        return (
            f"Job(id={self.id!r}, status={self.status!s}, stage={self.stage!s})"
        )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("jobs.id"))
    index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    clause_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    extracted_span: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extracted_span_category: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(384), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    job: Mapped[Job] = relationship("Job", back_populates="chunks")

    # Composite covers the dominant access path ("chunks for one job in order")
    # and the single-column lookup on job_id. Unique constraint prevents a re-
    # ingestion from inserting duplicate (job_id, index) rows.
    __table_args__ = (
        UniqueConstraint("job_id", "index", name="chunks_job_id_index_uq"),
        Index(
            "chunks_embedding_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return (
            f"Chunk(id={self.id!r}, job_id={self.job_id!r},"
            f" index={self.index}, clause_type={self.clause_type!r})"
        )


class JobDedup(Base):
    """Maps an idempotency key to a Job so duplicate submissions return the same id."""

    __tablename__ = "job_dedup"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )


class RiskResult(Base):
    __tablename__ = "risk_results"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.id"), unique=True
    )
    overall_score: Mapped[int] = mapped_column(Integer)
    risk_level: Mapped[str] = mapped_column(String)
    clause_summary: Mapped[dict[str, Any]] = mapped_column(JSON)
    flags: Mapped[list[Any]] = mapped_column(JSON)
    report_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    job: Mapped[Job] = relationship("Job", back_populates="result")

    def __repr__(self) -> str:
        return (
            f"RiskResult(id={self.id!r}, job_id={self.job_id!r},"
            f" overall_score={self.overall_score}, risk_level={self.risk_level!r})"
        )


_engine = create_engine(
    f"postgresql+psycopg2://{settings.postgres_user}:{settings.postgres_password}"
    f"@{settings.postgres_host}/{settings.postgres_db}",
    pool_pre_ping=True,
)
_SessionFactory = sessionmaker(bind=_engine)


def get_session():
    return _SessionFactory()


def init_db():
    # On postgresql, the pgvector extension must exist before create_all can
    # emit `vector(384)` column DDL or the HNSW index on chunks.embedding.
    # On other dialects (sqlite test fixture), the Vector TypeDecorator falls
    # back to JSON and the HNSW directive is ignored.
    if _engine.dialect.name == "postgresql":
        with _engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(_engine)

    if _engine.dialect.name == "postgresql":
        with _engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
                    "ON chunks USING hnsw (embedding vector_cosine_ops)"
                )
            )
