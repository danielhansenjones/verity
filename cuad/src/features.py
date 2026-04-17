from transformers import PreTrainedTokenizerFast


def prepare_train_features(
    examples: dict,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int = 512,
    doc_stride: int = 128,
) -> dict:
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer_start = examples["answer_start"][sample_idx]
        answer_text = examples["answer_text"][sample_idx]
        is_impossible = examples["is_impossible"][sample_idx]

        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized.sequence_ids(i)
        # Find the context token range within this window
        context_start = next(
            (j for j, s in enumerate(sequence_ids) if s == 1), None
        )
        context_end = next(
            (j for j, s in reversed(list(enumerate(sequence_ids))) if s == 1), None
        )

        if is_impossible or context_start is None or context_end is None:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        char_start = answer_start
        char_end = answer_start + len(answer_text)

        # If the answer falls outside the current window, label as no-answer
        if (
            offsets[context_start][0] > char_start
            or offsets[context_end][1] < char_end
        ):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        token_start = context_start
        while token_start <= context_end and offsets[token_start][0] <= char_start:
            token_start += 1
        start_positions.append(token_start - 1)

        token_end = context_end
        while token_end >= context_start and offsets[token_end][1] >= char_end:
            token_end -= 1
        end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def prepare_validation_features(
    examples: dict,
    tokenizer: PreTrainedTokenizerFast,
    max_seq_length: int = 512,
    doc_stride: int = 128,
) -> dict:
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized.pop("overflow_to_sample_mapping")

    # Retain example_id and offset_mapping for span aggregation
    example_ids = []
    for i in range(len(tokenized["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = tokenized.sequence_ids(i)
        # Zero out offsets for non-context tokens so aggregation ignores them
        tokenized["offset_mapping"][i] = [
            offset if sequence_ids[j] == 1 else None
            for j, offset in enumerate(tokenized["offset_mapping"][i])
        ]

    tokenized["example_id"] = example_ids
    return tokenized
