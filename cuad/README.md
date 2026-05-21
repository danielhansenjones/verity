# CUAD Span Extraction (v2)

Tier-2 of the classification cascade: a fine-tuned extractive QA model on CUAD v1, integrated on top of the zero-shot tier-1 classifier. Fires when tier-1 confidence is high enough and the predicted clause type maps to a CUAD category. Returns a verbatim span rather than just a label.

## Headline

| Model                    | Macro F1 (trimmed) |
|--------------------------|--------------------|
| BART-MNLI zero-shot      | 0.41               |
| RoBERTa-base fine-tuned  | 0.73               |
| RoBERTa-large fine-tuned | 0.75               |

Trimmed macro-F1 excludes categories with fewer than 30 positive spans in the test set; full macro-F1 over all 41 categories is recorded alongside in the artifacts.

## Caveats

**Category coverage.** CUAD has 41 clause categories. Macro-F1 is reported in two forms: full (all 41) and trimmed (excluding low-N categories where per-category F1 is unstable). The trimmed figure is the meaningful one for model comparison.

**Corpus skew.** The CUAD corpus is heavily skewed toward US commercial contracts. Performance on contracts from other jurisdictions, or contract types underrepresented in CUAD (e.g., employment agreements, consumer contracts), is unknown. Further fine-tuning on domain-specific data is required before deploying to those contexts.

**Span contiguity assumption.** Extractive QA treats every answer as a contiguous span. CUAD occasionally annotates disjoint spans for a single (contract, category) pair; the model can only predict one span per forward pass, a ceiling on recall for those categories.

**Split methodology.** Deterministic alphabetical contract-level 80/10/10 split for reproducibility. Results are not directly comparable to papers using a different split.

## Cascade

Tier-2 fires when both conditions hold:

1. Tier-1 confidence for the top clause type clears a configurable threshold.
2. The predicted clause type maps to one of the CUAD categories.

When tier-2 does not fire, the pipeline returns the tier-1 label and confidence only, with no extracted span.

## How to run

All commands are run from the repo root.

```bash
uv sync --group cuad
uv run python cuad/run.py           # runs full pipeline
uv run python cuad/run.py --from baseline  # resume from specific step
uv run python cuad/run.py --force baseline # rerun specific step
```

Steps in order:

| Step          | Description                                                          |
|---------------|----------------------------------------------------------------------|
| `download`    | Fetch CUAD v1 from HuggingFace                                       |
| `splits`      | Generate deterministic contract-level splits                         |
| `eda`         | Compute per-category span counts, identify low-N categories          |
| `baseline`    | Run BART-MNLI zero-shot on the test set                              |
| `train_base`  | Fine-tune roberta-base-squad2                                        |
| `train_large` | Fine-tune roberta-large-squad2                                       |
| `eval_all`    | Evaluate all models on the test set, generate comparison table and plots |

`--from <step>` resumes from the given step, skipping earlier steps if outputs exist. `--force <step>` reruns the given step even if outputs exist, then continues.

## Artifacts

```
cuad/artifacts/
├── eda/
│   ├── low_n_categories.json
│   └── category_span_counts.json
├── splits/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── eval/
│   ├── baseline_results.json
│   ├── roberta_base_results.json
│   ├── roberta_large_results.json
│   └── plots/
│       └── per_category_f1_grouped.png
└── models/
    ├── roberta-base-squad2-cuad/
    └── roberta-large-squad2-cuad/
```