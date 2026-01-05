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
    Integer,
    JSON,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from shared.settings import settings


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class JobStage(str, enum.Enum):
    INGESTION = "ingestion"
    CLASSIFICATION = "classification"
    SCORING = "scoring"
    ASSEMBLY = "assembly"
    DONE = "done"


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.QUEUED)
    stage: Mapped[JobStage] = mapped_column(Enum(JobStage), default=JobStage.INGESTION)
    object_key: Mapped[str] = mapped_column(String)
    filename: Mapped[str] = mapped_column(String)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    chunks: Mapped[list[Chunk]] = relationship(
        "Chunk", back_populates="job", order_by="Chunk.index"
    )
    result: Mapped[Optional[RiskResult]] = relationship(
        "RiskResult", back_populates="job", uselist=False
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    job_id: Mapped[str] = mapped_column(String, ForeignKey("jobs.id"))
    index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    clause_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    job: Mapped[Job] = relationship("Job", back_populates="chunks")


class RiskResult(Base):
    __tablename__ = "risk_results"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    job_id: Mapped[str] = mapped_column(String, ForeignKey("jobs.id"), unique=True)
    overall_score: Mapped[int] = mapped_column(Integer)
    risk_level: Mapped[str] = mapped_column(String)
    clause_summary: Mapped[dict[str, Any]] = mapped_column(JSON)
    flags: Mapped[list[Any]] = mapped_column(JSON)
    report_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    job: Mapped[Job] = relationship("Job", back_populates="result")


_engine = create_engine(
    f"postgresql+psycopg2://{settings.postgres_user}:{settings.postgres_password}"
    f"@{settings.postgres_host}/{settings.postgres_db}",
    pool_pre_ping=True,
)
_SessionFactory = sessionmaker(bind=_engine)


def get_session():
    return _SessionFactory()


def init_db():
    Base.metadata.create_all(_engine)
