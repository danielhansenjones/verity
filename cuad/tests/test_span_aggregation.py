"""Tests for span aggregation across sliding windows."""

from cuad.src.span_aggregation import aggregate_predictions


def _make_feature(example_id, start_logits, end_logits, offset_mapping):
    return {
        "example_id": example_id,
        "start_logits": start_logits,
        "end_logits": end_logits,
        "offset_mapping": offset_mapping,
    }


def _make_example(ex_id, contract_id, category, context):
    return {
        "id": ex_id,
        "contract_id": contract_id,
        "category": category,
        "context": context,
    }


def test_single_window_answer():
    context = "The company must provide 30 days notice."
    # Simulate one window where token 3-5 spans "30 days notice"
    n = 10
    offsets = [(i * 4, i * 4 + 4) for i in range(n)]
    offsets[0] = None  # CLS

    start_logits = [-10.0] * n
    end_logits = [-10.0] * n
    start_logits[3] = 5.0
    end_logits[5] = 5.0

    raw = [_make_feature("ex0", start_logits, end_logits, offsets)]
    examples = [_make_example("ex0", "c1", "Notice Period", context)]

    result = aggregate_predictions(raw, examples)
    pred = result[("c1", "Notice Period")]
    assert pred is not None
    assert pred["start"] == offsets[3][0]
    assert pred["end"] == offsets[5][1]


def test_no_answer():
    context = "No relevant clause here."
    n = 8
    # All logits high at CLS (index 0) => no-answer wins
    start_logits = [10.0] + [-5.0] * (n - 1)
    end_logits = [10.0] + [-5.0] * (n - 1)
    offsets = [None] + [(i * 4, i * 4 + 4) for i in range(1, n)]

    raw = [_make_feature("ex1", start_logits, end_logits, offsets)]
    examples = [_make_example("ex1", "c1", "Termination", context)]

    result = aggregate_predictions(
        raw, examples, null_score_diff_threshold=0.0
    )
    assert result[("c1", "Termination")] is None


def test_multiple_windows_best_wins():
    context = "The liability is capped at one million dollars per year."
    n = 8
    offsets_w1 = [None] + [(i * 5, i * 5 + 5) for i in range(1, n)]
    offsets_w2 = [None] + [(i * 5 + 20, i * 5 + 25) for i in range(1, n)]

    # Window 1: weak span signal
    sl_w1 = [-10.0] * n
    el_w1 = [-10.0] * n
    sl_w1[2] = 2.0
    el_w1[4] = 2.0

    # Window 2: stronger span signal - should win
    sl_w2 = [-10.0] * n
    el_w2 = [-10.0] * n
    sl_w2[2] = 8.0
    el_w2[4] = 8.0

    raw = [
        _make_feature("ex2", sl_w1, el_w1, offsets_w1),
        _make_feature("ex2", sl_w2, el_w2, offsets_w2),
    ]
    examples = [_make_example("ex2", "c2", "Cap On Liability", context)]

    result = aggregate_predictions(raw, examples)
    pred = result[("c2", "Cap On Liability")]
    assert pred is not None
    # Window 2 had higher score so its offsets should win
    assert pred["start"] == offsets_w2[2][0]


def test_max_answer_length_enforced():
    context = "A" * 200
    n = 10
    offsets = [None] + [(i, i + 1) for i in range(1, n)]

    # High start at 1, high end only at 9 (span length 9, exceeds limit=3).
    # Null (CLS) score = -1 + -1 = -2.
    # Best valid short span: start[1]+end[3] = 5 + (-10) = -5.
    # Null wins: -2 - (-5) = 3 > threshold=0 => no-answer.
    sl = [-1.0] + [5.0] + [-10.0] * (n - 2)
    el = [-1.0] + [-10.0] * (n - 2) + [5.0]

    raw = [_make_feature("ex3", sl, el, offsets)]
    examples = [_make_example("ex3", "c3", "IP Ownership", context)]

    result = aggregate_predictions(
        raw, examples, max_answer_length=3, null_score_diff_threshold=0.0
    )
    pred = result.get(("c3", "IP Ownership"))
    assert pred is None
