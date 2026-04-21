"""Run the full CUAD pipeline in sequence.

Usage:
    uv run python cuad/run.py                        # run all, skip completed steps
    uv run python cuad/run.py --from baseline        # resume from a specific step
    uv run python cuad/run.py --force                # rerun everything
    uv run python cuad/run.py --force baseline       # rerun one step only
    uv run python cuad/run.py --force baseline eval_all  # rerun specific steps
"""

import argparse
import os
import sys

# Must be set before torch is imported by any downstream module
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from cuad.scripts.baseline import main as baseline  # noqa: E402
from cuad.scripts.download import main as download  # noqa: E402
from cuad.scripts.eda import main as eda  # noqa: E402
from cuad.scripts.eval_all import main as eval_all  # noqa: E402
from cuad.scripts.splits import main as splits  # noqa: E402
from cuad.scripts.train_base import main as train_base  # noqa: E402
from cuad.scripts.train_large import main as train_large  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts: str) -> str:
    return os.path.join(_ROOT, *parts)


# Each step is (name, fn, done_artifact).
# A step is considered complete when its done_artifact exists on disk.
STEPS = [
    (
        "download",
        download,
        _p("cuad", "data", "cuad_v1", "qa_examples.parquet"),
    ),
    (
        "splits",
        splits,
        _p("cuad", "artifacts", "splits", "train_contracts.json"),
    ),
    (
        "eda",
        eda,
        _p("cuad", "artifacts", "eda", "per_category_counts.json"),
    ),
    (
        "baseline",
        baseline,
        _p(
            "cuad",
            "artifacts",
            "baselines",
            "bart_mnli_zero_shot_results.json",
        ),
    ),
    (
        "train_base",
        train_base,
        _p(
            "cuad",
            "artifacts",
            "models",
            "roberta_base_squad2_cuad",
            "training_log.json",
        ),
    ),
    (
        "train_large",
        train_large,
        _p(
            "cuad",
            "artifacts",
            "models",
            "roberta_large_squad2_cuad",
            "training_log.json",
        ),
    ),
    (
        "eval_all",
        eval_all,
        _p("cuad", "artifacts", "eval", "results_all_models.json"),
    ),
]

STEP_NAMES = [name for name, _, _ in STEPS]


def _banner(name: str, skipped: bool = False) -> None:
    bar = "=" * 60
    suffix = "  [skipping - already done]" if skipped else ""
    print(f"\n{bar}")
    print(f"  {name}{suffix}")
    print(f"{bar}\n")


def _is_done(artifact: str) -> bool:
    return os.path.exists(artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CUAD pipeline.")
    parser.add_argument(
        "--from",
        dest="start_from",
        metavar="STEP",
        help="Start from this step (skips earlier steps entirely).",
    )
    parser.add_argument(
        "--force",
        nargs="*",
        metavar="STEP",
        help=(
            "Force rerun even if the step is already done. "
            "Pass no arguments to force all steps, "
            "or pass step names to force specific ones."
        ),
    )
    args = parser.parse_args()

    # Validate step names
    if args.start_from and args.start_from not in STEP_NAMES:
        print(
            f"Unknown step '{args.start_from}'. "
            f"Valid steps: {', '.join(STEP_NAMES)}"
        )
        sys.exit(1)

    forced: set[str] = set()
    force_all = False
    if args.force is not None:
        if len(args.force) == 0:
            force_all = True
        else:
            for name in args.force:
                if name not in STEP_NAMES:
                    print(
                        f"Unknown step '{name}'. "
                        f"Valid steps: {', '.join(STEP_NAMES)}"
                    )
                    sys.exit(1)
            forced = set(args.force)

    start_idx = 0
    if args.start_from:
        start_idx = STEP_NAMES.index(args.start_from)

    for name, fn, artifact in STEPS[start_idx:]:
        should_force = force_all or name in forced
        done = _is_done(artifact) and not should_force

        _banner(name, skipped=done)
        if done:
            continue

        fn()
        _release_gpu()


def _release_gpu() -> None:
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
