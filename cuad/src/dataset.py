import json

import pandas as pd
from datasets import Dataset


def load_qa_examples(data_dir: str) -> pd.DataFrame:
    return pd.read_parquet(f"{data_dir}/qa_examples.parquet")


def build_split(qa_examples: pd.DataFrame, contract_ids: list[str]) -> Dataset:
    mask = qa_examples["contract_id"].isin(contract_ids)
    subset = qa_examples[mask].reset_index(drop=True).copy()
    if "id" not in subset.columns:
        subset["id"] = subset.index.astype(str)
    return Dataset.from_pandas(subset, preserve_index=False)


def load_contracts(data_dir: str) -> dict[str, str]:
    with open(f"{data_dir}/contracts.json") as f:
        return json.load(f)
