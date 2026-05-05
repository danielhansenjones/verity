from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture
def mock_sentence_transformer():
    with patch("worker.processors.embeddings.SentenceTransformer") as MockST:
        instance = MockST.return_value
        instance.get_embedding_dimension.return_value = 384
        yield instance


def test_embed_documents_does_not_add_query_prefix(mock_sentence_transformer):
    from worker.processors.embeddings import EmbeddingModel

    mock_sentence_transformer.encode.return_value = np.array(
        [[0.1] * 384, [0.2] * 384]
    )

    model = EmbeddingModel()
    result = model.embed_documents(["clause one", "clause two"])

    args, kwargs = mock_sentence_transformer.encode.call_args
    assert args[0] == ["clause one", "clause two"]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["batch_size"] == 32
    assert len(result) == 2
    assert len(result[0]) == 384


def test_embed_query_adds_bge_instruction_prefix(mock_sentence_transformer):
    from worker.processors.embeddings import EmbeddingModel, _BGE_QUERY_PREFIX

    mock_sentence_transformer.encode.return_value = np.array([[0.1] * 384])

    model = EmbeddingModel()
    result = model.embed_query("what is the governing law")

    args, kwargs = mock_sentence_transformer.encode.call_args
    assert args[0] == [_BGE_QUERY_PREFIX + "what is the governing law"]
    assert kwargs["normalize_embeddings"] is True
    assert len(result) == 384


def test_dimension_property_reports_model_dimension(mock_sentence_transformer):
    from worker.processors.embeddings import EmbeddingModel

    model = EmbeddingModel()
    assert model.dimension == 384


def test_embed_documents_respects_batch_size_override(mock_sentence_transformer):
    from worker.processors.embeddings import EmbeddingModel

    mock_sentence_transformer.encode.return_value = np.array([[0.0] * 384] * 64)

    model = EmbeddingModel()
    model.embed_documents(["text"] * 64, batch_size=16)

    _, kwargs = mock_sentence_transformer.encode.call_args
    assert kwargs["batch_size"] == 16
