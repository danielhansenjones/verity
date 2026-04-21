"""Evaluate all three models and produce the comparison table and plot."""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from cuad.src.dataset import build_split, load_qa_examples
from cuad.src.eval_harness import EvalResult, evaluate
from cuad.src.features import prepare_validation_features
from cuad.src.span_aggregation import aggregate_predictions


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cuad_v1")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "splits")
EDA_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "eda")
BASELINES_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "baselines")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "models")
EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "eval")
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")


def run_roberta_inference(
    model_dir: str,
    test_examples: Dataset,
    max_seq_length: int = 512,
    doc_stride: int = 128,
    batch_size: int = 8,
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_dir,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    tokenized = test_examples.map(
        lambda ex: prepare_validation_features(
            ex, tokenizer, max_seq_length, doc_stride
        ),
        batched=True,
        remove_columns=test_examples.column_names,
    )

    all_start_logits = []
    all_end_logits = []

    n_batches = (len(tokenized) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(
            range(0, len(tokenized), batch_size), total=n_batches, desc="Inference"
        ):
            batch = tokenized[i : i + batch_size]
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_start_logits.extend(outputs.start_logits.cpu().float().tolist())
            all_end_logits.extend(outputs.end_logits.cpu().float().tolist())

    # Pre-extract columns once rather than one dataset read per window
    example_ids = tokenized["example_id"]
    offset_mappings = tokenized["offset_mapping"]

    raw_preds = [
        {
            "start_logits": sl,
            "end_logits": el,
            "example_id": example_ids[i],
            "offset_mapping": offset_mappings[i],
        }
        for i, (sl, el) in enumerate(zip(all_start_logits, all_end_logits))
    ]

    examples_list = test_examples.to_list()
    aggregated = aggregate_predictions(raw_preds, examples_list)

    rows = []
    for (contract_id, category), pred in aggregated.items():
        rows.append({
            "contract_id": contract_id,
            "category": category,
            "predicted_text": pred["text"] if pred else None,
            "predicted_null": pred is None,
        })
    return pd.DataFrame(rows)


def build_ground_truth(test_examples: Dataset) -> pd.DataFrame:
    df = test_examples.to_pandas()
    rows = []
    for (contract_id, category), group in df.groupby(["contract_id", "category"]):
        gold_spans = group[~group["is_impossible"]]["answer_text"].tolist()
        rows.append({
            "contract_id": contract_id,
            "category": category,
            "gold_spans": gold_spans,
            "is_impossible": len(gold_spans) == 0,
        })
    return pd.DataFrame(rows)


def result_to_dict(result: EvalResult) -> dict:
    return {
        "model": result.model_name,
        "macro_f1_full": result.macro_f1_full,
        "macro_f1_trimmed": result.macro_f1_trimmed,
        "macro_em": result.macro_em,
        "per_category": result.per_category,
        "confusion": result.confusion,
        "extra": result.extra,
    }


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    qa_examples = load_qa_examples(DATA_DIR)

    with open(os.path.join(SPLITS_DIR, "test_contracts.json")) as f:
        test_ids = json.load(f)
    with open(os.path.join(EDA_DIR, "low_n_categories.json")) as f:
        low_n = json.load(f)

    test_dataset = build_split(qa_examples, test_ids)
    ground_truth = build_ground_truth(test_dataset)

    with open(os.path.join(BASELINES_DIR, "bart_mnli_zero_shot_results.json")) as f:
        bart_dict = json.load(f)
    bart_result = EvalResult(
        model_name=bart_dict["model"],
        macro_f1_full=bart_dict["macro_f1_full"],
        macro_f1_trimmed=bart_dict["macro_f1_trimmed"],
        macro_em=bart_dict["macro_em"],
        per_category=bart_dict["per_category"],
        confusion=bart_dict["confusion"],
        extra=bart_dict.get("extra", {}),
    )

    print("Running roberta-base-squad2 eval ...")
    base_preds = run_roberta_inference(
        os.path.join(MODELS_DIR, "roberta_base_squad2_cuad"),
        test_dataset,
    )
    base_preds.to_parquet(
        os.path.join(
            MODELS_DIR, "roberta_base_squad2_cuad", "test_predictions.parquet"
        ),
        index=False,
    )
    base_result = evaluate(base_preds, ground_truth, low_n)
    base_result.model_name = "roberta-base-squad2-cuad"

    print("Running roberta-large-squad2 eval ...")
    large_preds = run_roberta_inference(
        os.path.join(MODELS_DIR, "roberta_large_squad2_cuad"),
        test_dataset,
    )
    large_preds.to_parquet(
        os.path.join(
            MODELS_DIR, "roberta_large_squad2_cuad", "test_predictions.parquet"
        ),
        index=False,
    )
    large_result = evaluate(large_preds, ground_truth, low_n)
    large_result.model_name = "roberta-large-squad2-cuad"

    results = [bart_result, base_result, large_result]

    combined = [result_to_dict(r) for r in results]
    with open(os.path.join(EVAL_DIR, "results_all_models.json"), "w") as f:
        json.dump(combined, f, indent=2)

    rows = []
    for r in results:
        for cat, stats in r.per_category.items():
            rows.append({"model": r.model_name, "category": cat, **stats})
    per_cat_df = pd.DataFrame(rows)
    per_cat_df.to_parquet(
        os.path.join(EVAL_DIR, "per_category_f1.parquet"), index=False
    )

    categories = sorted(per_cat_df["category"].unique())
    model_names = [r.model_name for r in results]
    n_models = len(model_names)
    n_cats = len(categories)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(22, 7))
    for i, model_name in enumerate(model_names):
        model_df = (
            per_cat_df[per_cat_df["model"] == model_name].set_index("category")
        )
        f1s = [
            model_df.loc[c, "f1"] if c in model_df.index else 0.0
            for c in categories
        ]
        offsets = [
            j + (i - n_models / 2) * width + width / 2 for j in range(n_cats)
        ]
        ax.bar(offsets, f1s, width=width, label=model_name)

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(categories, rotation=90, fontsize=6)
    ax.set_ylabel("F1")
    ax.set_title("Per-category F1: all models")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "per_category_f1_grouped.png"), dpi=150)
    plt.close(fig)
    print("Saved per_category_f1_grouped.png")

    print("\nModel comparison:")
    print(f"{'Model':<35} {'F1 (full)':>10} {'F1 (trim)':>10} {'EM':>8}")
    print("-" * 65)
    for r in results:
        print(
            f"{r.model_name:<35} "
            f"{r.macro_f1_full:>10.4f} "
            f"{r.macro_f1_trimmed:>10.4f} "
            f"{r.macro_em:>8.4f}"
        )


if __name__ == "__main__":
    main()
