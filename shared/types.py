from pgvector.sqlalchemy import Vector as PgVector
from sqlalchemy.types import JSON, TypeDecorator


class Vector(TypeDecorator):
    # pgvector on PostgreSQL; JSON fallback on other dialects so the SQLite
    # test fixture can still create the chunks table. Cosine-distance queries
    # require a real pgvector backend and run against testcontainers, not SQLite.
    impl = JSON
    cache_ok = True

    # Forward pgvector's distance operators so column-level expressions like
    # `Chunk.embedding.cosine_distance(...)` resolve to the right SQL.
    comparator_factory = PgVector.comparator_factory

    def __init__(self, dim: int):
        super().__init__()
        self._dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PgVector(self._dim))
        return dialect.type_descriptor(JSON())
