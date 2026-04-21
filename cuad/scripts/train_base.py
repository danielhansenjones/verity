"""Fine-tune deepset/roberta-base-squad2 on CUAD."""

import json
import os

from cuad.src.train import train


SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "splits")
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "models", "roberta_base_squad2_cuad"
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cuad_v1")


def main() -> None:
    with open(os.path.join(SPLITS_DIR, "train_contracts.json")) as f:
        train_ids = json.load(f)
    with open(os.path.join(SPLITS_DIR, "val_contracts.json")) as f:
        val_ids = json.load(f)

    print(f"train contracts: {len(train_ids)}")
    print(f"val contracts:   {len(val_ids)}")

    train(
        model_name_or_path="deepset/roberta-base-squad2",
        output_dir=OUTPUT_DIR,
        data_dir=DATA_DIR,
        train_contract_ids=train_ids,
        val_contract_ids=val_ids,
        batch_size=128,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
        learning_rate=4e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=False,
        bf16=True,
        max_seq_length=512,
        doc_stride=128,
        dataloader_num_workers=4,
    )

    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
