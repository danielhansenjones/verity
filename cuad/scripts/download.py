"""Download CUAD v1 from HuggingFace and persist to cuad/data/cuad_v1/."""

import json
import os
import re
import sys

import pandas as pd
from huggingface_hub import hf_hub_download


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cuad_v1")

# theatticusproject/cuad stores the full dataset as a single SQuAD-format JSON.
HF_REPO = "theatticusproject/cuad"
HF_FILE = "CUAD_v1/CUAD_v1.json"

EXPECTED_CATEGORIES = 41


def _extract_category(question: str) -> str:
    match = re.search(r'"([^"]+)"', question)
    return match.group(1) if match else question[:80]


def _parse_squad_json(path: str) -> tuple[list[dict], dict[str, str], set[str]]:
    print(f"  Parsing {path} ...")
    with open(path) as f:
        data = json.load(f)

    rows: list[dict] = []
    contracts: dict[str, str] = {}
    categories: set[str] = set()

    for article in data["data"]:
        contract_id = article["title"].strip()
        for para in article["paragraphs"]:
            context = para["context"]
            contracts[contract_id] = context
            for qa in para["qas"]:
                question = qa["question"]
                category = _extract_category(question)
                categories.add(category)
                answers = qa.get("answers", [])
                if answers:
                    for ans in answers:
                        rows.append({
                            "contract_id": contract_id,
                            "category": category,
                            "question": question,
                            "context": context,
                            "answer_start": ans["answer_start"],
                            "answer_text": ans["text"],
                            "is_impossible": False,
                        })
                else:
                    rows.append({
                        "contract_id": contract_id,
                        "category": category,
                        "question": question,
                        "context": context,
                        "answer_start": -1,
                        "answer_text": "",
                        "is_impossible": True,
                    })

    return rows, contracts, categories


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    cache_dir = os.path.join(DATA_DIR, "hf_cache")
    print(f"Downloading {HF_REPO}/{HF_FILE} from HuggingFace ...")
    local_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILE,
        repo_type="dataset",
        local_dir=cache_dir,
    )
    print(f"  Saved to {local_path}")

    rows, contracts, categories = _parse_squad_json(local_path)

    print(f"  {len(rows)} QA rows")
    print(f"  {len(categories)} categories")
    print(f"  {len(contracts)} contracts")

    if len(categories) != EXPECTED_CATEGORIES:
        print(
            f"WARNING: expected {EXPECTED_CATEGORIES} categories, "
            f"got {len(categories)}",
            file=sys.stderr,
        )

    df = pd.DataFrame(rows)
    df["id"] = df.index.astype(str)
    positive_count = int((~df["is_impossible"]).sum())
    print(f"  {positive_count} positive spans")

    parquet_path = os.path.join(DATA_DIR, "qa_examples.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {parquet_path}")

    contracts_path = os.path.join(DATA_DIR, "contracts.json")
    with open(contracts_path, "w") as f:
        json.dump(contracts, f)
    print(f"Saved {contracts_path}")

    category_list = sorted(categories)
    category_path = os.path.join(DATA_DIR, "category_list.json")
    with open(category_path, "w") as f:
        json.dump(category_list, f, indent=2)
    print(f"Saved {category_path}")

    print("\nVerification:")
    loaded = pd.read_parquet(parquet_path)
    print(f"  Parquet rows:  {len(loaded)}")
    print(f"  Positive rows: {int((~loaded['is_impossible']).sum())}")
    print(f"  Categories:    {loaded['category'].nunique()}")
    print(f"  Contracts:     {loaded['contract_id'].nunique()}")


if __name__ == "__main__":
    main()
