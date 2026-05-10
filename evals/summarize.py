"""Render an eval report as a Markdown summary, suitable for $GITHUB_STEP_SUMMARY."""
import json
import sys
from pathlib import Path


def render(report: dict) -> str:
    agg = report["aggregate"]

    counts_line = (
        f"Cases: {agg['n_total']} total, {agg['n_scored']} scored,"
        f" {agg['n_errors']} errors"
    )
    lines = [
        "## RAG Eval Summary",
        "",
        f"Judge model: `{report['judge_model']}`  ",
        f"Dataset: `{report['dataset']}`  ",
        counts_line,
        "",
        "### By dimension",
        "",
        "| Dimension | Score | N |",
        "|-----------|-------|---|",
    ]
    for dim, stats in agg["by_dimension"].items():
        lines.append(f"| {dim} | {stats['mean']:.3f} | {stats['n']} |")

    cat_header = (
        "| Category | Faithfulness | Citation accuracy"
        " | Completeness | Refusal correctness |"
    )
    cat_align = (
        "|----------|--------------|-------------------"
        "|--------------|---------------------|"
    )
    lines += ["", "### By category", "", cat_header, cat_align]
    for cat, dims in agg.get("by_category", {}).items():
        row = [
            cat,
            f"{dims.get('faithfulness', float('nan')):.3f}",
            f"{dims.get('citation_accuracy', float('nan')):.3f}",
            f"{dims.get('completeness', float('nan')):.3f}",
            f"{dims.get('refusal_correctness', float('nan')):.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    buckets = agg.get("refusal_buckets") or {}
    if buckets:
        lines += ["", "### Refusal buckets", ""]
        for bucket, count in sorted(buckets.items()):
            lines.append(f"- `{bucket}`: {count}")

    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: summarize.py <path/to/report.json>", file=sys.stderr)
        sys.exit(2)
    report = json.loads(Path(sys.argv[1]).read_text())
    print(render(report))


if __name__ == "__main__":
    main()
