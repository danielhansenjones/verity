"""Tests for tokenization and sliding window logic."""

import pytest
from transformers import AutoTokenizer

from cuad.src.features import prepare_train_features, prepare_validation_features


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")


def _make_examples(question, context, answer_text="", answer_start=-1, is_impossible=True):
    return {
        "id": ["ex0"],
        "question": [question],
        "context": [context],
        "answer_text": [answer_text],
        "answer_start": [answer_start],
        "is_impossible": [is_impossible],
    }


def test_train_features_answerable(tokenizer):
    context = "The company shall not assign this agreement without prior consent."
    question = "What are the assignment restrictions?"
    answer = "shall not assign this agreement without prior consent"
    answer_start = context.index(answer)

    examples = _make_examples(question, context, answer, answer_start, False)
    features = prepare_train_features(examples, tokenizer, max_seq_length=512, doc_stride=128)

    # At least one window should have a non-CLS start position
    found_span = any(
        sp != 0 for sp in features["start_positions"]
    )
    assert found_span, "Expected at least one window to contain the answer span"


def test_train_features_impossible(tokenizer):
    context = "Governing law shall be the state of Delaware."
    question = "What is the termination clause?"
    examples = _make_examples(question, context)
    features = prepare_train_features(examples, tokenizer, max_seq_length=512, doc_stride=128)

    # All windows should map to CLS (index 0) for no-answer
    assert all(sp == 0 for sp in features["start_positions"])
    assert all(ep == 0 for ep in features["end_positions"])


def test_train_features_span_outside_window(tokenizer):
    # Build a long context where the answer appears only in the second window.
    # First window should label the span as no-answer (CLS).
    filler = "This is filler text. " * 30  # ~600 chars
    answer = "liability is capped at one million dollars"
    context = filler + answer + " Additional terms apply."
    answer_start = context.index(answer)
    question = "What is the liability cap?"

    examples = _make_examples(question, context, answer, answer_start, False)
    features = prepare_train_features(examples, tokenizer, max_seq_length=128, doc_stride=32)

    # There should be multiple windows
    assert len(features["start_positions"]) > 1

    # Not every window should have a non-zero start (the answer is only in some windows)
    starts = features["start_positions"]
    assert 0 in starts, "At least one window should label the span as no-answer (CLS=0)"


def test_validation_features_retains_example_id(tokenizer):
    context = "Termination requires 30 days written notice."
    question = "What is the notice period?"
    examples = _make_examples(question, context)
    features = prepare_validation_features(examples, tokenizer, max_seq_length=512, doc_stride=128)

    assert "example_id" in features
    assert features["example_id"][0] == "ex0"


def test_validation_features_offset_mapping_nulled_for_non_context(tokenizer):
    context = "Renewal is automatic unless terminated."
    question = "How is renewal handled?"
    examples = _make_examples(question, context)
    features = prepare_validation_features(examples, tokenizer, max_seq_length=512, doc_stride=128)

    # Offset mapping entries for non-context tokens (question, special tokens)
    # should be None
    offsets = features["offset_mapping"][0]
    assert None in offsets, "Expected None entries for non-context tokens"
