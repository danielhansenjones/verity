"""Build contract-level train/val/test splits from the downloaded CUAD data."""

import glob
import json
import os

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cuad_v1")
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "splits")


def _contract_ids_from_parquet_glob(pattern: str) -> list[str]:
    files = glob.glob(pattern, recursive=True)
    if not files:
        return []
    frames = [pd.read_parquet(f, columns=["title"]) for f in sorted(files)]
    titles = pd.concat(frames, ignore_index=True)["title"].dropna().unique().tolist()
    return sorted(str(t).strip() for t in titles)


def _split_from_hf_cache(
    cache_dir: str,
) -> tuple[list[str], list[str]] | None:
    """
    Return (train_ids, test_ids) if the HuggingFace snapshot contains
    distinct train and test parquet shards. Returns None if only a single
    shard is present (all data lumped into one 'train' file).
    """
    train_ids = _contract_ids_from_parquet_glob(
        os.path.join(cache_dir, "**", "train-*.parquet")
    )
    test_ids = _contract_ids_from_parquet_glob(
        os.path.join(cache_dir, "**", "test-*.parquet")
    )
    if train_ids and test_ids:
        return train_ids, test_ids
    return None


def _deterministic_split(
    all_contracts: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Contract-level 80/10/10 split on alphabetically sorted IDs.
    Used when the dataset does not ship with a pre-defined test split.
    """
    n = len(all_contracts)
    n_test = max(1, round(n * 0.10))
    n_val = max(1, round(n * 0.10))
    test = all_contracts[:n_test]
    val = all_contracts[n_test: n_test + n_val]
    train = all_contracts[n_test + n_val:]
    return train, val, test


def main() -> None:
    os.makedirs(SPLITS_DIR, exist_ok=True)

    qa = pd.read_parquet(os.path.join(DATA_DIR, "qa_examples.parquet"))
    all_contracts = sorted(qa["contract_id"].unique().tolist())
    print(f"Total contracts: {len(all_contracts)}")

    cache_dir = os.path.join(DATA_DIR, "hf_cache")
    hf_split = _split_from_hf_cache(cache_dir) if os.path.isdir(cache_dir) else None

    if hf_split is not None:
        hf_train_ids, hf_test_ids = hf_split
        contract_set = set(all_contracts)
        test_contracts = [c for c in hf_test_ids if c in contract_set]
        remaining = sorted(c for c in all_contracts if c not in set(test_contracts))
        n_val = max(1, round(len(remaining) * 0.10))
        val_contracts = remaining[:n_val]
        train_contracts = remaining[n_val:]
        print("Split source: HuggingFace dataset test shard")
    else:
        print("Split source: deterministic 80/10/10 (no HF test shard found)")
        train_contracts, val_contracts, test_contracts = _deterministic_split(
            all_contracts
        )

    print(f"  train: {len(train_contracts)}")
    print(f"  val:   {len(val_contracts)}")
    print(f"  test:  {len(test_contracts)}")
    print(f"  total: {len(train_contracts) + len(val_contracts) + len(test_contracts)}")

    assert not (set(train_contracts) & set(val_contracts)), "train/val overlap"
    assert not (set(train_contracts) & set(test_contracts)), "train/test overlap"
    assert not (set(val_contracts) & set(test_contracts)), "val/test overlap"

    with open(os.path.join(SPLITS_DIR, "train_contracts.json"), "w") as f:
        json.dump(train_contracts, f, indent=2)
    with open(os.path.join(SPLITS_DIR, "val_contracts.json"), "w") as f:
        json.dump(val_contracts, f, indent=2)
    with open(os.path.join(SPLITS_DIR, "test_contracts.json"), "w") as f:
        json.dump(test_contracts, f, indent=2)

    print(f"Splits written to {SPLITS_DIR}")


if __name__ == "__main__":
    main()
