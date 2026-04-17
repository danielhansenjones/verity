import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from cuad.src.span_aggregation import aggregate_predictions


def _question(category: str) -> str:
    return f'Highlight the parts (if any) related to "{category}"'


_QUESTIONS = {cat: _question(cat) for cat in [
    "Termination For Convenience",
    "Notice Period To Terminate Renewal",
    "Cap On Liability",
    "Uncapped Liability",
    "IP Ownership Assignment",
    "Joint IP Ownership",
    "Governing Law",
    "Renewal Term",
    "Auto Renewal",
    "Third Party Beneficiary",
    "Exclusivity",
    "Non-Compete",
    "Confidentiality",
]}


class SpanExtractor:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_seq_length: int = 512,
        doc_stride: int = 128,
    ):
        self._device = device
        self._max_seq_length = max_seq_length
        self._doc_stride = doc_stride

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForQuestionAnswering.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
        ).to(self._device)
        self._model.eval()

    def extract(
        self,
        chunk_text: str,
        cuad_categories: list[str],
    ) -> dict[str, dict | None]:
        results = {}

        for category in cuad_categories:
            question = _QUESTIONS.get(
                category, f"Highlight the parts related to \"{category}\""
            )
            pred = self._run_qa(question, chunk_text)
            results[category] = pred

        return results

    def _run_qa(self, question: str, context: str) -> dict | None:
        tokenized = self._tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self._max_seq_length,
            stride=self._doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        tokenized.pop("overflow_to_sample_mapping")
        offset_mapping_raw = tokenized.pop("offset_mapping").tolist()
        sequence_ids = [
            tokenized.sequence_ids(i) for i in range(len(offset_mapping_raw))
        ]

        offset_mapping = [
            [
                offset if sequence_ids[win][j] == 1 else None
                for j, offset in enumerate(offset_mapping_raw[win])
            ]
            for win in range(len(offset_mapping_raw))
        ]

        input_ids = tokenized["input_ids"].to(self._device)
        attention_mask = tokenized["attention_mask"].to(self._device)

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)

        start_logits = outputs.start_logits.cpu().float().tolist()
        end_logits = outputs.end_logits.cpu().float().tolist()

        raw_preds = [
            {
                "start_logits": start_logits[i],
                "end_logits": end_logits[i],
                "example_id": "single",
                "offset_mapping": offset_mapping[i],
            }
            for i in range(len(start_logits))
        ]

        examples = [
            {
                "id": "single",
                "contract_id": "_",
                "category": "_",
                "context": context,
            }
        ]

        aggregated = aggregate_predictions(raw_preds, examples)
        return aggregated.get(("_", "_"))
