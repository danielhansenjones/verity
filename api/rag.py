import logging
from typing import Optional

from sqlalchemy.orm import Session

from shared.models import Chunk
from worker.processors.embeddings import EmbeddingModel


logger = logging.getLogger(__name__)

_TOP_K_DEFAULT = 8
_TOP_K_MAX = 20

_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    # Process-level singleton: the model is ~130 MB and 5-10 s to load,
    # so first /ask request would block. preload_embedding_model() wires
    # the eager load into the API lifespan so the cost is paid at startup.
    global _model
    if _model is None:
        logger.info("rag: loading embedding model")
        _model = EmbeddingModel(device="cpu")
        logger.info("rag: embedding model ready (dim=%d)", _model.dimension)
    return _model


def preload_embedding_model() -> None:
    get_embedding_model()


def embed_query(text: str) -> list[float]:
    return get_embedding_model().embed_query(text)


def retrieve(
    db: Session,
    job_id: str,
    query_embedding: list[float],
    k: int = _TOP_K_DEFAULT,
) -> list[Chunk]:
    k = min(max(k, 1), _TOP_K_MAX)
    return (
        db.query(Chunk)
        .filter(Chunk.job_id == job_id)
        .filter(Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(k)
        .all()
    )
