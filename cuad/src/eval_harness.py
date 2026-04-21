import string
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    model_name: str
    macro_f1_full: float
    macro_f1_trimmed: float
    macro_em: float
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = sum(
        min(pred_tokens.count(t), gold_tokens.count(t)) for t in common
    ) / len(pred_tokens)
    recall = sum(
        min(pred_tokens.count(t), gold_tokens.count(t)) for t in common
    ) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, gold: str) -> float:
    return float(_normalize(prediction) == _normalize(gold))


def _score_prediction(
    pred_text: str | None, gold_spans: list[str], is_impossible: bool
) -> tuple[float, float]:
    """Returns (f1, em) for one (prediction, gold) pair."""
    if is_impossible:
        # gold is no-answer
        if pred_text is None:
            return 1.0, 1.0
        return 0.0, 0.0

    if pred_text is None:
        return 0.0, 0.0

    # Match against the best gold span
    best_f1 = max(_token_f1(pred_text, g) for g in gold_spans)
    best_em = max(_exact_match(pred_text, g) for g in gold_spans)
    return best_f1, best_em


def evaluate(
    predictions,
    ground_truth,
    low_n_categories: list[str],
) -> EvalResult:
    merged = ground_truth.merge(
        predictions, on=["contract_id", "category"], how="left"
    )

    per_category: dict[str, list[tuple[float, float]]] = {}
    answered_when_should = 0
    answered_when_shouldnt = 0
    unanswered_when_should = 0
    unanswered_when_shouldnt = 0

    for _, row in merged.iterrows():
        cat = row["category"]
        gold_spans = row["gold_spans"] if isinstance(row["gold_spans"], list) else []
        is_impossible = bool(row.get("is_impossible", False))
        pred_text = (
            None if row.get("predicted_null", True) else row.get("predicted_text")
        )

        f1, em = _score_prediction(pred_text, gold_spans, is_impossible)
        per_category.setdefault(cat, []).append((f1, em))

        if is_impossible:
            if pred_text is None:
                unanswered_when_shouldnt += 1
            else:
                answered_when_shouldnt += 1
        else:
            if pred_text is not None:
                answered_when_should += 1
            else:
                unanswered_when_should += 1

    category_stats = {}
    for cat, scores in per_category.items():
        f1s, ems = zip(*scores)
        n = len(f1s)
        pred_count = sum(
            1 for _, row in merged[merged["category"] == cat].iterrows()
            if not row.get("predicted_null", True)
        )
        category_stats[cat] = {
            "f1": sum(f1s) / n,
            "em": sum(ems) / n,
            "n_test": n,
            "n_predicted": pred_count,
        }

    all_f1 = [v["f1"] for v in category_stats.values()]
    trimmed_f1 = [
        v["f1"]
        for cat, v in category_stats.items()
        if cat not in low_n_categories
    ]
    all_em = [v["em"] for v in category_stats.values()]

    return EvalResult(
        model_name="",
        macro_f1_full=sum(all_f1) / len(all_f1) if all_f1 else 0.0,
        macro_f1_trimmed=sum(trimmed_f1) / len(trimmed_f1) if trimmed_f1 else 0.0,
        macro_em=sum(all_em) / len(all_em) if all_em else 0.0,
        per_category=category_stats,
        confusion={
            "answered_correct": answered_when_should,
            "answered_incorrect": answered_when_shouldnt,
            "unanswered_incorrect": unanswered_when_should,
            "unanswered_correct": unanswered_when_shouldnt,
        },
    )
