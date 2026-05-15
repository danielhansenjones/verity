"""Hand-rolled RAG eval runner.

Reads evals/dataset.jsonl, ingests the referenced fixture PDFs into the local
Postgres (idempotent via a sidecar cache), runs each question through the
in-process retrieval and LLM pipeline, scores the response on four dimensions,
and writes a JSON report.

Cost guard: print the estimate up front and require --confirm-cost above the
threshold. Judge model defaults to claude-haiku-4-5; generator is whatever
shared.settings.anthropic_model points at (claude-sonnet-4-6 by default).
"""
import argparse
import json
import logging
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from api.llm import ask as llm_ask
from api.rag import embed_query, retrieve
from evals import judges
from shared.logging_config import configure_logging
from shared.models import Chunk, Job, JobStage, JobStatus, get_session
from worker.processors import ingestion
from worker.processors.embeddings import EmbeddingModel


_DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
_FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "test_documents"
_CACHE_PATH = Path(__file__).parent / ".fixture_cache.json"
_RESULTS_DIR = Path(__file__).parent / "results"

# Rough per-case price floor. 1 generation (sonnet) + 3 judge calls (haiku).
# Used only to decide whether to require --confirm-cost.
_EST_COST_PER_CASE = 0.10


def _load_cases(path: Path, limit: int | None) -> list[dict]:
    cases = []
    with open(path) as f:
        for line in f:
            cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.write_text(json.dumps(cache, indent=2))


def _ensure_fixture_job(
    pdf_name: str, embedding_model: EmbeddingModel, db
) -> str:
    cache = _load_cache()
    cached_id = cache.get(pdf_name)
    if cached_id:
        job = db.get(Job, cached_id)
        if job is not None and job.status == JobStatus.COMPLETED:
            # Require *all* chunks to be embedded, not just the first.
            # A partial prior run could otherwise be reused and silently
            # exclude un-embedded chunks from retrieval.
            total = (
                db.query(Chunk).filter(Chunk.job_id == cached_id).count()
            )
            embedded = (
                db.query(Chunk)
                .filter(
                    Chunk.job_id == cached_id, Chunk.embedding.is_not(None)
                )
                .count()
            )
            if total > 0 and embedded == total:
                return cached_id

    pdf_path = _FIXTURE_DIR / pdf_name
    pdf_bytes = pdf_path.read_bytes()
    storage = MagicMock()
    storage.download_bytes.return_value = pdf_bytes

    job = Job(
        id=str(uuid.uuid4()),
        status=JobStatus.RUNNING,
        stage=JobStage.INGESTION,
        object_key=f"eval-fixtures/{pdf_name}",
        filename=f"eval-{pdf_name}",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(job)
    db.commit()

    ingestion.run(job, db, storage, embedding_model)

    # Skip classify/score/assemble; the eval only exercises retrieval + LLM,
    # not the deterministic risk-flag path.
    job.status = JobStatus.COMPLETED
    job.stage = JobStage.DONE
    db.commit()

    cache[pdf_name] = job.id
    _save_cache(cache)
    return job.id


def _run_case(case: dict, fixture_jobs: dict[str, str], db, judge_model: str) -> dict:
    case_id = case["id"]
    t0 = time.time()

    try:
        query_vec = embed_query(case["question"])
        chunks = retrieve(db, fixture_jobs[case["fixture_pdf"]], query_vec, k=8)
    except Exception as exc:
        return {"case_id": case_id, "error": f"retrieval failed: {exc}"}

    try:
        response, usage = llm_ask(case["question"], chunks)
    except Exception as exc:
        return {"case_id": case_id, "error": f"generation failed: {exc}"}

    chunks_by_id = {c.id: c for c in chunks}
    citations_with_text = [
        {
            "chunk_id": cite.chunk_id,
            "quote": cite.quote,
            "chunk_text": chunks_by_id[cite.chunk_id].text
            if cite.chunk_id in chunks_by_id
            else "(chunk missing from retrieved set)",
        }
        for cite in response.citations
    ]

    faith = judges.judge_faithfulness(
        case["question"], response.answer, citations_with_text, model=judge_model
    )
    citacc = judges.judge_citation_accuracy(
        case["question"], response.answer, citations_with_text, model=judge_model
    )
    comp = judges.judge_completeness(
        case["question"],
        response.answer,
        case.get("reference_answer"),
        case.get("reference_keywords", []),
        model=judge_model,
    )
    ref = judges.judge_refusal_correctness(
        case["expected_refusal"], response.refusal_reason is not None
    )

    return {
        "case_id": case_id,
        "category": case["category"],
        "duration_s": round(time.time() - t0, 3),
        "answer": response.answer,
        "refusal_reason": response.refusal_reason,
        "citations": [c.model_dump() for c in response.citations],
        "usage": usage,
        "scores": {
            "faithfulness": faith.score,
            "citation_accuracy": citacc.score,
            "completeness": comp.score,
            "refusal_correctness": ref.score,
        },
        "refusal_bucket": ref.bucket,
    }


def _aggregate(results: list[dict]) -> dict:
    scored = [r for r in results if "scores" in r]
    agg = {
        "n_total": len(results),
        "n_scored": len(scored),
        "n_errors": sum(1 for r in results if "error" in r),
        "by_dimension": {},
        "by_category": {},
        "refusal_buckets": dict(
            Counter(r["refusal_bucket"] for r in scored if "refusal_bucket" in r)
        ),
    }

    dims = ["faithfulness", "citation_accuracy", "completeness", "refusal_correctness"]
    for dim in dims:
        scores = [r["scores"][dim] for r in scored]
        if scores:
            agg["by_dimension"][dim] = {
                "mean": round(sum(scores) / len(scores), 3),
                "n": len(scores),
            }

    by_cat: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in scored:
        for dim, s in r["scores"].items():
            by_cat[r["category"]][dim].append(s)
    agg["by_category"] = {
        cat: {dim: round(sum(ss) / len(ss), 3) for dim, ss in dims_map.items()}
        for cat, dims_map in by_cat.items()
    }

    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=_DATASET_PATH)
    parser.add_argument(
        "--judge", default=judges.DEFAULT_JUDGE_MODEL, help="Primary judge model"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N cases"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the default report path",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=5.0,
        help="Threshold above which --confirm-cost is required",
    )
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"]
    )
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger("evals.run")

    cases = _load_cases(args.dataset, args.limit)
    est_cost = len(cases) * _EST_COST_PER_CASE
    print(
        f"loaded {len(cases)} cases from {args.dataset}"
        f"  est cost ~${est_cost:.2f}"
    )

    if est_cost > args.max_cost and not args.confirm_cost:
        print(
            f"estimated cost ${est_cost:.2f} exceeds --max-cost ${args.max_cost:.2f}."
            f" re-run with --confirm-cost to proceed."
        )
        return

    embedding_model = EmbeddingModel(device=args.device)

    with get_session() as db:
        pdfs = sorted({c["fixture_pdf"] for c in cases})
        logger.info("ensuring %d fixture jobs: %s", len(pdfs), pdfs)
        fixture_jobs = {
            pdf: _ensure_fixture_job(pdf, embedding_model, db) for pdf in pdfs
        }

        results: list[dict] = []
        for case in cases:
            logger.info("running case %s", case["id"])
            results.append(_run_case(case, fixture_jobs, db, args.judge))

        agg = _aggregate(results)

        report = {
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "judge_model": args.judge,
            "dataset": str(args.dataset),
            "aggregate": agg,
            "results": results,
        }

        if args.output:
            out_path = args.output
        else:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = _RESULTS_DIR / f"{ts}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str))

        logger.info("wrote %s", out_path)
        print(json.dumps(agg, indent=2, default=str))


if __name__ == "__main__":
    main()
