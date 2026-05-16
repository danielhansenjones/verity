import collections

import numpy as np


def aggregate_predictions(
    raw_predictions: list[dict],
    examples: list[dict],
    n_best_size: int = 20,
    max_answer_length: int = 512,
    null_score_diff_threshold: float = 0.0,
) -> dict[tuple[str, str], dict]:
    """
    Aggregate per-window logits into one best prediction per (contract_id, category).

    raw_predictions: list of dicts with keys start_logits, end_logits, example_id
    examples: list of dicts with keys id, contract_id, category, context, offset_mapping
    """
    features_per_example = collections.defaultdict(list)
    for feat in raw_predictions:
        features_per_example[feat["example_id"]].append(feat)

    results = {}

    for example in examples:
        example_id = example["id"]
        contract_id = example["contract_id"]
        category = example["category"]
        key = (contract_id, category)

        if key in results:
            continue

        features = features_per_example[example_id]
        context = example["context"]

        min_null_score = None
        valid_answers = []

        for feat in features:
            start_logits = feat["start_logits"]
            end_logits = feat["end_logits"]
            offset_mapping = feat["offset_mapping"]

            cls_score = start_logits[0] + end_logits[0]
            if min_null_score is None or cls_score < min_null_score:
                min_null_score = cls_score

            start_indices = np.argsort(start_logits)[-n_best_size:][::-1].tolist()
            end_indices = np.argsort(end_logits)[-n_best_size:][::-1].tolist()

            for start_idx in start_indices:
                for end_idx in end_indices:
                    if offset_mapping[start_idx] is None:
                        continue
                    if offset_mapping[end_idx] is None:
                        continue
                    if end_idx < start_idx:
                        continue
                    if end_idx - start_idx + 1 > max_answer_length:
                        continue

                    char_start = offset_mapping[start_idx][0]
                    char_end = offset_mapping[end_idx][1]
                    span_text = context[char_start:char_end]

                    valid_answers.append(
                        {
                            "score": start_logits[start_idx] + end_logits[end_idx],
                            "text": span_text,
                            "start": char_start,
                            "end": char_end,
                        }
                    )

        if not valid_answers:
            results[key] = None
            continue

        best = max(valid_answers, key=lambda x: x["score"])

        # SQuAD2 no-answer decision: null wins if its score exceeds best span score
        # by more than the threshold
        if min_null_score is not None and (
            min_null_score - best["score"] > null_score_diff_threshold
        ):
            results[key] = None
        else:
            results[key] = best

    return results
