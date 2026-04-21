import json
import os

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from cuad.src.dataset import build_split, load_qa_examples
from cuad.src.features import prepare_train_features


def train(
    model_name_or_path: str,
    output_dir: str,
    data_dir: str,
    train_contract_ids: list[str],
    val_contract_ids: list[str],
    batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    num_train_epochs: int = 4,
    learning_rate: float = 3e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    gradient_checkpointing: bool = False,
    bf16: bool = True,
    max_seq_length: int = 512,
    doc_stride: int = 256,
    dataloader_num_workers: int = 4,  # 8 caused RAM saturation during map phase
) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)

    qa_examples = load_qa_examples(data_dir)
    train_dataset = build_split(qa_examples, train_contract_ids)
    val_dataset = build_split(qa_examples, val_contract_ids)

    def tokenize(examples):
        return prepare_train_features(examples, tokenizer, max_seq_length, doc_stride)

    # default 1000-row batch explodes to ~136k windows via sliding-window expansion,
    # producing ~20 GB of Python objects before Arrow can flush to disk
    tokenized_train = train_dataset.map(
        tokenize, batched=True, remove_columns=train_dataset.column_names,
        batch_size=32, num_proc=4,
    )
    tokenized_val = val_dataset.map(
        tokenize, batched=True, remove_columns=val_dataset.column_names,
        batch_size=32, num_proc=4,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        bf16=bf16,
        fp16=False,
        gradient_checkpointing=gradient_checkpointing,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)

    log_history = trainer.state.log_history
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)
