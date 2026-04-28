"""Tests for the shared eval harness."""

import pandas as pd
import pytest

from cuad.src.eval_harness import evaluate, _normalize, _token_f1, _exact_match


def test_normalize():
    assert _normalize("Liability Cap!") == "liability cap"
    assert _normalize("  MULTIPLE   spaces  ") == "multiple spaces"


def test_token_f1_exact():
    assert _token_f1("liability cap", "liability cap") == pytest.approx(1.0)


def test_token_f1_partial():
    f1 = _token_f1("liability cap one million", "liability cap")
    assert 0.0 < f1 < 1.0


def test_token_f1_no_overlap():
    assert _token_f1("governing law", "termination notice") == pytest.approx(0.0)


def test_exact_match():
    assert _exact_match("governing law", "governing law") == pytest.approx(1.0)
    assert _exact_match("Governing Law", "governing law") == pytest.approx(1.0)
    assert _exact_match("governing law", "termination") == pytest.approx(0.0)


def _make_predictions(rows):
    return pd.DataFrame(rows)


def _make_ground_truth(rows):
    return pd.DataFrame(rows)


def test_evaluate_all_correct():
    preds = _make_predictions([
        {"contract_id": "c1", "category": "Governing Law", "predicted_text": "Delaware", "predicted_null": False},
        {"contract_id": "c1", "category": "Cap On Liability", "predicted_text": None, "predicted_null": True},
    ])
    gt = _make_ground_truth([
        {"contract_id": "c1", "category": "Governing Law", "gold_spans": ["Delaware"], "is_impossible": False},
        {"contract_id": "c1", "category": "Cap On Liability", "gold_spans": [], "is_impossible": True},
    ])

    result = evaluate(preds, gt, low_n_categories=[])
    assert result.macro_f1_full == pytest.approx(1.0)
    assert result.macro_em == pytest.approx(1.0)


def test_evaluate_all_wrong():
    preds = _make_predictions([
        {"contract_id": "c1", "category": "Governing Law", "predicted_text": "California", "predicted_null": False},
    ])
    gt = _make_ground_truth([
        {"contract_id": "c1", "category": "Governing Law", "gold_spans": ["Delaware"], "is_impossible": False},
    ])

    result = evaluate(preds, gt, low_n_categories=[])
    assert result.macro_f1_full == pytest.approx(0.0)
    assert result.macro_em == pytest.approx(0.0)


def test_evaluate_trimmed_macro_excludes_low_n():
    preds = _make_predictions([
        {"contract_id": "c1", "category": "Governing Law", "predicted_text": "Delaware", "predicted_null": False},
        {"contract_id": "c1", "category": "Rare Category", "predicted_text": "wrong", "predicted_null": False},
    ])
    gt = _make_ground_truth([
        {"contract_id": "c1", "category": "Governing Law", "gold_spans": ["Delaware"], "is_impossible": False},
        {"contract_id": "c1", "category": "Rare Category", "gold_spans": ["something else"], "is_impossible": False},
    ])

    result = evaluate(preds, gt, low_n_categories=["Rare Category"])
    assert result.macro_f1_full < 1.0
    assert result.macro_f1_trimmed == pytest.approx(1.0)


def test_evaluate_multiple_gold_spans():
    # Prediction matches any one of multiple gold spans
    preds = _make_predictions([
        {"contract_id": "c1", "category": "IP Ownership", "predicted_text": "second span", "predicted_null": False},
    ])
    gt = _make_ground_truth([
        {
            "contract_id": "c1",
            "category": "IP Ownership",
            "gold_spans": ["first span", "second span"],
            "is_impossible": False,
        }
    ])

    result = evaluate(preds, gt, low_n_categories=[])
    assert result.macro_f1_full == pytest.approx(1.0)
