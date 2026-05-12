"""add pgvector extension and chunks.embedding

Revision ID: 0001
Revises:
Create Date: 2026-05-12

"""
from typing import Sequence, Union

from alembic import op


revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Idempotent across the hybrid bootstrap (Base.metadata.create_all may have
    # already created the column on a fresh DB) and the legacy upgrade path
    # (existing dev volumes where chunks predates this branch).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector(384)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS embedding")
    # Extension intentionally left in place; other schemas may depend on it.
