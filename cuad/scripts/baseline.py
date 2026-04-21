"""BART-MNLI zero-shot baseline on the CUAD test set."""

import json
import os
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from cuad.src.dataset import load_contracts, load_qa_examples
from cuad.src.eval_harness import evaluate


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cuad_v1")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "splits")
EDA_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "eda")
BASELINES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "baselines"
)

MAX_SEQ_LENGTH = 512
DOC_STRIDE = 128
BATCH_SIZE = 32


def sliding_windows(text: str, tokenizer, max_len: int, stride: int) -> list[str]:
    tokens = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    windows = []
    start = 0
    while start < len(tokens):
        end = min(start + max_len, len(tokens))
        windows.append(tokenizer.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += max_len - stride
    return windows


def _build_contract_windows(
    contract_ids: list[str],
    contracts: dict[str, str],
    tokenizer,
) -> dict[str, list[str]]:
    return {
        cid: sliding_windows(contracts[cid], tokenizer, MAX_SEQ_LENGTH, DOC_STRIDE)
        for cid in contract_ids
    }


def _score_all(
    classifier,
    categories: list[str],
    contract_windows: dict[str, list[str]],
) -> dict[tuple[str, str], tuple[float, str]]:
    """
    Score every window against all categories in BATCH_SIZE chunks with progress.
    Returns {(contract_id, category): (best_score, best_window)}.
    """
    all_windows: list[str] = []
    index: list[tuple[str, int]] = []

    for cid, windows in contract_windows.items():
        for i, w in enumerate(windows):
            all_windows.append(w)
            index.append((cid, i))

    n_batches = (len(all_windows) + BATCH_SIZE - 1) // BATCH_SIZE
    results = []
    for i in tqdm(range(0, len(all_windows), BATCH_SIZE), total=n_batches, unit="batch",
                  desc=f"{len(all_windows)} windows x {len(categories)} labels"):
        batch = all_windows[i : i + BATCH_SIZE]
        raw = classifier(
            batch,
            candidate_labels=categories,
            hypothesis_template="This text is about {}",
            multi_label=True,
            batch_size=BATCH_SIZE,
        )
        if isinstance(raw, dict):
            results.append(raw)
        else:
            results.extend(raw)

    best: dict[tuple[str, str], tuple[float, str]] = {}
    for (cid, win_idx), result in zip(index, results):
        window = contract_windows[cid][win_idx]
        for cat, score in zip(result["labels"], result["scores"]):
            key = (cid, cat)
            if key not in best or score > best[key][0]:
                best[key] = (score, window)

    return best


def _predictions_from_scores(
    contract_ids: list[str],
    categories: list[str],
    scores: dict[tuple[str, str], tuple[float, str]],
    threshold: float,
) -> list[dict]:
    rows = []
    for cid in contract_ids:
        for cat in categories:
            best_score, best_window = scores.get((cid, cat), (0.0, ""))
            rows.append({
                "contract_id": cid,
                "category": cat,
                "predicted_text": best_window if best_score >= threshold else None,
                "predicted_null": best_score < threshold,
            })
    return rows


def tune_threshold(
    classifier,
    tokenizer,
    val_examples: pd.DataFrame,
    contracts: dict[str, str],
    categories: list[str],
    thresholds: list[float],
) -> float:
    print("Tuning threshold on val set ...")
    val_ids = val_examples["contract_id"].unique().tolist()
    contract_windows = _build_contract_windows(val_ids, contracts, tokenizer)

    print(f"  Scoring {len(val_ids)} contracts x {len(categories)} categories ...")
    scores = _score_all(classifier, categories, contract_windows)

    ground_truth = _build_ground_truth(val_examples)
    with open(os.path.join(EDA_DIR, "low_n_categories.json")) as f:
        low_n = json.load(f)

    best_t = thresholds[0]
    best_f1 = -1.0
    for t in thresholds:
        rows = _predictions_from_scores(val_ids, categories, scores, t)
        result = evaluate(pd.DataFrame(rows), ground_truth, low_n)
        if result.macro_f1_trimmed > best_f1:
            best_f1 = result.macro_f1_trimmed
            best_t = t
        print(f"  threshold={t:.2f}  trimmed macro-F1={result.macro_f1_trimmed:.4f}")

    print(f"Best threshold: {best_t} (val trimmed macro-F1={best_f1:.4f})")
    return best_t


def _build_ground_truth(examples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (contract_id, category), group in examples.groupby(
        ["contract_id", "category"]
    ):
        gold_spans = group[~group["is_impossible"]]["answer_text"].tolist()
        rows.append({
            "contract_id": contract_id,
            "category": category,
            "gold_spans": gold_spans,
            "is_impossible": len(gold_spans) == 0,
        })
    return pd.DataFrame(rows)


def run_inference(
    classifier,
    tokenizer,
    examples: pd.DataFrame,
    contracts: dict[str, str],
    categories: list[str],
    threshold: float,
) -> pd.DataFrame:
    contract_ids = examples["contract_id"].unique().tolist()
    contract_windows = _build_contract_windows(contract_ids, contracts, tokenizer)

    print(
        f"  Scoring {len(contract_ids)} contracts "
        f"x {len(categories)} categories ..."
    )
    scores = _score_all(classifier, categories, contract_windows)
    rows = _predictions_from_scores(contract_ids, categories, scores, threshold)
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(BASELINES_DIR, exist_ok=True)

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {'cuda' if device == 0 else 'cpu'}")

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
        dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

    qa_examples = load_qa_examples(DATA_DIR)
    contracts = load_contracts(DATA_DIR)

    with open(os.path.join(DATA_DIR, "category_list.json")) as f:
        categories = json.load(f)

    with open(os.path.join(SPLITS_DIR, "val_contracts.json")) as f:
        val_ids = json.load(f)
    with open(os.path.join(SPLITS_DIR, "test_contracts.json")) as f:
        test_ids = json.load(f)

    val_examples = qa_examples[qa_examples["contract_id"].isin(val_ids)]
    test_examples = qa_examples[qa_examples["contract_id"].isin(test_ids)]

    thresholds = [round(0.3 + i * 0.05, 2) for i in range(15)]
    threshold = tune_threshold(
        classifier, tokenizer, val_examples, contracts, categories, thresholds
    )

    print(f"\nRunning test inference (threshold={threshold}) ...")
    t0 = time.time()
    predictions = run_inference(
        classifier, tokenizer, test_examples, contracts, categories, threshold
    )
    elapsed = time.time() - t0

    predictions.to_parquet(
        os.path.join(BASELINES_DIR, "bart_mnli_zero_shot_predictions.parquet"),
        index=False,
    )

    with open(os.path.join(EDA_DIR, "low_n_categories.json")) as f:
        low_n = json.load(f)

    ground_truth = _build_ground_truth(test_examples)
    result = evaluate(predictions, ground_truth, low_n)
    result.model_name = "bart-large-mnli-zero-shot"
    result.extra["latency_seconds"] = elapsed
    result.extra["n_predictions"] = len(predictions)
    result.extra["threshold"] = threshold

    results_dict = {
        "model": result.model_name,
        "macro_f1_full": result.macro_f1_full,
        "macro_f1_trimmed": result.macro_f1_trimmed,
        "macro_em": result.macro_em,
        "per_category": result.per_category,
        "confusion": result.confusion,
        "extra": result.extra,
    }
    with open(
        os.path.join(BASELINES_DIR, "bart_mnli_zero_shot_results.json"), "w"
    ) as f:
        json.dump(results_dict, f, indent=2)

    print("\nResults:")
    print(f"  macro F1 (full):    {result.macro_f1_full:.4f}")
    print(f"  macro F1 (trimmed): {result.macro_f1_trimmed:.4f}")
    print(f"  macro EM:           {result.macro_em:.4f}")
    print(f"  elapsed:            {elapsed:.0f}s")


if __name__ == "__main__":
    main()
