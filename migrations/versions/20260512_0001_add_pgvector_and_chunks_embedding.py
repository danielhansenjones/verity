"""add pgvector extension and chunks.embedding

Revision ID: 0001
Revises:
Create Date: 2026-05-12

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector


revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column(
        "chunks",
        sa.Column("embedding", Vector(384), nullable=True),
    )
    op.create_index(
        "chunks_embedding_idx",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("chunks_embedding_idx", table_name="chunks")
    op.drop_column("chunks", "embedding")
    # Extension intentionally left in place; other schemas may depend on it.
