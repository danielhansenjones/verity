"""EDA: per-category counts, context/answer length distributions."""

import json
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


# Suppress the "sequence length > model max" warning that fires when tokenizing
# long contracts for length-counting purposes (we are not running inference).
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cuad_v1")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "splits")
EDA_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "eda")
PLOTS_DIR = os.path.join(EDA_DIR, "plots")

LOW_N_THRESHOLD = 30


def load_split_contracts(split_name: str) -> list[str]:
    path = os.path.join(SPLITS_DIR, f"{split_name}_contracts.json")
    with open(path) as f:
        return json.load(f)


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = pd.read_parquet(os.path.join(DATA_DIR, "qa_examples.parquet"))
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

    train_ids = set(load_split_contracts("train"))
    val_ids = set(load_split_contracts("val"))
    test_ids = set(load_split_contracts("test"))

    def split_label(cid: str) -> str:
        if cid in train_ids:
            return "train"
        if cid in val_ids:
            return "val"
        if cid in test_ids:
            return "test"
        return "unknown"

    df["split"] = df["contract_id"].map(split_label)

    # Per-category positive span counts by split
    positives = df[~df["is_impossible"]]
    counts = (
        positives.groupby(["category", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
    )
    counts_dict = counts.to_dict(orient="index")
    with open(os.path.join(EDA_DIR, "per_category_counts.json"), "w") as f:
        json.dump(counts_dict, f, indent=2)
    print(f"Per-category counts: {len(counts_dict)} categories")

    # Low-N categories (< LOW_N_THRESHOLD positive spans in test set)
    low_n = sorted(
        cat for cat, row in counts.iterrows() if row.get("test", 0) < LOW_N_THRESHOLD
    )
    with open(os.path.join(EDA_DIR, "low_n_categories.json"), "w") as f:
        json.dump(low_n, f, indent=2)
    print(f"Low-N categories (<{LOW_N_THRESHOLD} test positives): {len(low_n)}")

    # Per-category train vs test bar chart
    categories = sorted(counts.index)
    x = range(len(categories))
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(
        [i - 0.2 for i in x],
        [counts.loc[c, "train"] for c in categories],
        width=0.4,
        label="train",
    )
    ax.bar(
        [i + 0.2 for i in x],
        [counts.loc[c, "test"] for c in categories],
        width=0.4,
        label="test",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=90, fontsize=7)
    ax.set_ylabel("Positive spans")
    ax.set_title("CUAD per-category positive span counts (train vs test)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "per_category_train_test.png"), dpi=150)
    plt.close(fig)
    print("Saved per_category_train_test.png")

    # Context token length distribution (sample up to 2000 unique contracts for speed)
    unique_contexts = df.drop_duplicates("contract_id")["context"].tolist()
    sample = unique_contexts[:2000]
    ctx_lengths = []
    for i in tqdm(range(0, len(sample), 64), desc="Tokenising contexts"):
        batch = sample[i : i + 64]
        encoded = tokenizer(batch, truncation=False, add_special_tokens=False)
        ctx_lengths.extend(len(ids) for ids in encoded["input_ids"])

    fig, ax = plt.subplots()
    ax.hist(ctx_lengths, bins=50)
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Contracts")
    ax.set_title("Context token length distribution")
    ax.axvline(512, color="red", linestyle="--", label="512 (max_seq_length)")
    ax.legend()
    fig.savefig(os.path.join(PLOTS_DIR, "context_token_length_dist.png"), dpi=150)
    plt.close(fig)
    print("Saved context_token_length_dist.png")

    # Answer token length distribution
    ans_texts = positives["answer_text"].tolist()
    sample_ans = ans_texts[:5000]
    ans_lengths = []
    for i in tqdm(range(0, len(sample_ans), 128), desc="Tokenising answers"):
        batch = sample_ans[i : i + 128]
        encoded = tokenizer(batch, truncation=False, add_special_tokens=False)
        ans_lengths.extend(len(ids) for ids in encoded["input_ids"])

    fig, ax = plt.subplots()
    ax.hist(ans_lengths, bins=50)
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Spans")
    ax.set_title("Answer span token length distribution")
    fig.savefig(os.path.join(PLOTS_DIR, "answer_token_length_dist.png"), dpi=150)
    plt.close(fig)
    print("Saved answer_token_length_dist.png")


if __name__ == "__main__":
    main()
