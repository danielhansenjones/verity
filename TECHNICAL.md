# Technical Reference

Depth document for the Contract Risk Pipeline. The [README](README.md) is the entry point and quick start; this file is the implementation reference.

## Auth

Protected endpoints require the `X-API-Key` header. The expected value is read from `CONTRACT_API_KEY`. `GET /health` is public.

```bash
curl -H "X-API-Key: $CONTRACT_API_KEY" http://localhost:8000/jobs
```

If `CONTRACT_API_KEY` is unset, auth is disabled and a warning is logged at startup. Production deploys must set the key. Constant-time comparison is used to reject wrong keys without timing leaks.

## Upload limits

`POST /jobs` caps the raw request body via `MAX_UPLOAD_BYTES`. Oversized uploads are rejected early by middleware with `413 Payload Too Large` before any body is parsed or written to MinIO. Requests without a `Content-Length` header are rejected with `411 Length Required`. A defensive second check on actual bytes catches forged headers.

## Rate limits

Per-IP via slowapi. `POST /jobs` and the read endpoints (`GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/report`) have independent budgets configurable via `RATE_LIMIT_SUBMIT` and `RATE_LIMIT_READ`. `GET /health` is exempt. Breach returns `429 Too Many Requests` with a `Retry-After` header.

Per-IP keying assumes direct ingress (see [Deployment topology](#deployment-topology)). Counters live in-process; a single API pod is the supported configuration.

## Idempotency

Duplicate submissions to `POST /jobs` return the existing job rather than creating a new one.

- Clients may supply an `Idempotency-Key` header to control the dedup bucket.
- If the header is absent, the fallback key is `SHA-256` of the PDF body.
- Keys are namespaced (`client:*` vs `content:*`) so a client key that happens to match a content hash cannot collide.
- Replays return `200 OK` (not `201`) with an `Idempotent-Replay: true` header and the current state of the existing job.
- Concurrent first-time submissions with the same key are resolved at the database layer via a unique constraint; the losing request returns the winner's job id.

```bash
curl -X POST http://localhost:8000/jobs \
  -H "X-API-Key: $CONTRACT_API_KEY" \
  -H "Idempotency-Key: order-12345" \
  -F "file=@contract.pdf"
```

## Metrics

Prometheus exposition on two scrape targets: the API process and the worker process. Tracked series include job submissions and completions, per-stage duration histograms, per-stage error counts, queue depth, and RAG-specific question/latency/token counters.

## Pipeline

Processing runs in four sequential stages. On retry, the worker resumes from the last successful stage; a transient failure during scoring does not re-run ingestion.

### Stage 1. Ingestion
Downloads the PDF from MinIO, extracts text with `pypdf`, and chunks by structural markers (numbered clauses, ALL CAPS headings, paragraph breaks) with a token-length fallback for oversized sections.

### Stage 2. Classification
Two-tier cascade per chunk. Tier-1 is zero-shot over the product's clause-label taxonomy in `worker/processors/clause_labels.py`, mapped to CUAD categories by `category_mapping.json`. Editing the labels is a config change, not a retrain. The legal-domain work happens at tier-2: a RoBERTa span extractor fine-tuned on CUAD.

Tier-2 fires only when tier-1 confidence meets a configurable threshold and the predicted label has CUAD coverage. It returns a verbatim extracted span alongside the classification result. A per-chunk timeout falls back to tier-1 output without failing the job.

Full benchmarks and training pipeline: [cuad/README.md](cuad/README.md).

### Stage 3. Scoring
Two-pass scoring per chunk:
- **Tone model** (`distilbert-base-uncased-finetuned-sst-2-english`): a HuggingFace text-classification pipeline contributes to the risk signal. Swappable at this slot.
- **Rule-based flags**: spaCy `Matcher` applied to token sequences declared as YAML in `worker/processors/rules/`. Rules fire on lexical variants ("sole discretion", "sole and absolute discretion", "sole and unfettered discretion" all match one rule) and return exact character offsets so the assembler can quote the matched span. Rule set is data, not code - adding a pattern does not require a Python change.

Document score is a weighted average of chunk scores, bucketed into `low` / `medium` / `high` risk levels.

### Stage 4. Assembly
Aggregates chunk scores and flags into a `RiskResult`, writes the full JSON report to MinIO, and marks the job `completed`.

### Report output

`GET /jobs/{id}/report` returns the full structured result. A trimmed example with both a rule-based flag and a tier-2 extracted span:

```json
{
  "job_id": "a3f1c2d4-5e6f-7890-abcd-ef1234567890",
  "filename": "services_agreement.pdf",
  "overall_score": 72,
  "risk_level": "high",
  "clause_summary": {
    "termination": 3,
    "indemnification": 2,
    "governing law": 1
  },
  "flags": [
    {
      "chunk_index": 4,
      "clause_type": "termination",
      "extracted_span_category": "Termination For Convenience",
      "rule_id": "unilateral_discretion",
      "matched_text": "sole discretion",
      "reason": "One party holds unchecked decision-making authority",
      "severity": "high",
      "excerpt": "Either party may terminate at its sole discretion without cause.",
      "evidence_source": "extracted_span"
    }
  ],
  "chunks": [
    {
      "index": 4,
      "clause_type": "termination",
      "confidence": 0.94,
      "extracted_span": "Either party may terminate this Agreement for convenience upon thirty (30) days written notice",
      "extracted_span_category": "Termination For Convenience",
      "score": 78
    }
  ]
}
```

`evidence_source: "extracted_span"` means the flag excerpt is the verbatim span returned by the tier-2 model. When tier-2 does not fire or times out, `evidence_source` is `"chunk_text"` and the excerpt is taken from a window around the matched token span.

## RAG query layer

The deterministic pipeline produces a structured `RiskResult` per job. The RAG layer is a separate capability that lets a caller ask free-text questions against the contract's chunks and get back a typed answer with verified citations. It is bolted onto the same `chunks` table; the deterministic path runs unchanged whether or not the RAG endpoint is configured.

### Architecture

```
chunks (Postgres) --[local embedding @ ingestion]--> chunks.embedding (vector)
                                                            |
question --[query-prefix embedding]--> vector --> cosine ANN (HNSW) --> top-k chunks
                                                                              |
                                                                              v
                                                      Claude via Anthropic SDK
                                                      with tool_use forcing the
                                                      AnswerResponse schema
                                                                              |
                                                                              v
                                                          citation grounding check
                                                          (cited chunk_ids in the
                                                          retrieved set; quotes are
                                                          verbatim substrings)
                                                                              |
                                                                pass: 200 with answer
                                                                fail: 502
```

### Chunking strategy

Reuses the existing clause-aware chunker in `worker/processors/ingestion.py`. The same chunks the classifier and span extractor already saw are the chunks RAG cites from. No re-chunking pass.

### Embedding choice

Local sentence-transformer embeddings running on the worker. No per-call vendor cost, no PII leaves the box. Documents and queries are encoded asymmetrically per the model's training contract; embeddings are L2-normalised so pgvector's cosine distance reduces to scaled L2.

### Retrieval

`pgvector` HNSW index over `chunks.embedding` with `vector_cosine_ops`. The retrieval query filters by `job_id` and `embedding IS NOT NULL` before ordering, so questions only see chunks from the single contract under review.

### Threat model

The `/ask` endpoint feeds user-supplied questions and arbitrary uploaded-PDF chunks into Claude. Both inputs can carry injected instructions. Mitigations:

1. `tool_use` with a forced tool choice: the model can only respond by calling `submit_answer`, so a "ignore previous instructions and output X" attack has nowhere to land in the response shape.
2. Post-generation citation grounding (`grounding_error` in `api/llm.py`): every cited quote must be a verbatim substring of a retrieved chunk. A fabricated quote fails this check and the response 502s.
3. Refusals are a typed field on the schema; the model cannot break out of the structured response shape.

### Prompt and structured outputs

The Anthropic SDK enforces an `AnswerResponse` Pydantic schema via `tool_use` + `tool_choice` forcing. The system prompt sets the cite-or-refuse contract and is cached with `cache_control: {"type": "ephemeral"}` so bursts of questions amortise the input cost. The user message contains the question and the retrieved chunks labelled with `[chunk_id=... chunk_index=...]` markers.

### Citation grounding

Every `Citation` returned by the model is verified server-side:
- `chunk_id` must appear in the retrieved set. A fabricated id returns `502`.
- `quote` must be a substring of the named chunk's text under whitespace-normalised, case-insensitive comparison. A paraphrased or invented quote returns `502`.

Refusals must have an empty answer and no citations; otherwise also `502`.

### Backfill for prior jobs

Chunks created before the embedding column existed have `embedding IS NULL`. Run `scripts/backfill_embeddings.py` once to populate them. Idempotent and safe to re-run.

```bash
uv run python scripts/backfill_embeddings.py --device cpu
```

### Example request

```bash
curl -X POST http://localhost:8000/jobs/$JOB_ID/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $CONTRACT_API_KEY" \
  -d '{"question":"What is the governing law?","top_k":8}'
```

```json
{
  "answer": "The agreement is governed by Delaware law.",
  "citations": [
    {
      "chunk_id": "a3f1c2d4-...",
      "chunk_index": 14,
      "quote": "governed by the laws of the State of Delaware"
    }
  ],
  "refusal_reason": null
}
```

### Eval harness

Hand-rolled harness (`evals/run.py`) scoring faithfulness, citation accuracy, completeness, and refusal correctness with an LLM-as-judge. Judge model is deliberately different from the generator to reduce self-grading bias. Inter-judge agreement is optionally checked via a second judge.

CI workflow `.github/workflows/evals.yml` re-runs a frozen subset on PRs that touch the RAG surface, posting per-dimension deltas as the job summary.

## Architecture and scaling

Four tiers deliberately separated: API, queue, worker, storage. Each scales, fails, and deploys independently.

- **API tier**: stateless. Horizontal scale behind a load balancer. Idempotency makes retried submissions safe.
- **Worker tier**: one consumer per job via Redis Streams consumer group (`XREADGROUP`). Add workers with `docker compose up --scale worker=N`. Each worker's consumer name is `hostname:pid` so `XAUTOCLAIM` can distinguish live and dead consumers.
- **Queue**: Redis Streams. Entries stay in the pending list until `XACK`. Crashed-consumer entries are reclaimed by `XAUTOCLAIM` after a configurable idle threshold.
- **State**: Postgres. Read replicas are the scale-out path for dashboards and audits.
- **Blobs**: MinIO. S3-compatible, so production can swap to S3/GCS without code changes.

### Deployment topology

Direct ingress: clients connect straight to the API container. Behind a reverse proxy, two things change:

- `slowapi` keys on `remote_addr`, which becomes the proxy's IP. Switch the limiter key to something the proxy populates (e.g. `X-API-Key`), and parse `X-Forwarded-For` only when the immediate peer is in a known trusted-proxy CIDR (untrusted XFF is spoofable).
- Rate-limit counters live in-process. Multiple API pods need shared storage (`slowapi` accepts `storage_uri="redis://..."`).

### Current deployment shape vs. designed shape

The repo ships with one worker. The architecture supports N. The bottleneck at N=1 is GPU inference on classification; scale-out is linear in worker count until Redis becomes the constraint.

### Load test summary

Single-worker Locust runs at light and 20 VU spike loads completed with zero 5xx and sub-100ms submit p50. Submit latency stays low while the worker is buried in ML inference because the API writes to Redis and returns; the queue absorbs concurrent submissions without losing jobs. Adding a second worker halves completion time for a burst back to single-job processing time.

## Project Structure

```
contract-risk-pipeline/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .pre-commit-config.yaml
├── .github/workflows/
├── scripts/
│   ├── run.py
│   ├── test.py
│   └── backfill_embeddings.py
├── shared/        # settings, models, queue and storage clients
├── api/           # FastAPI routes, RAG retrieval, LLM client
├── worker/        # worker loop and processors
└── tests/
```

## Fault Tolerance

- Worker retries up to `max_retries` before marking a job `failed`.
- `job.stage` is preserved on failure; retry resumes from the last completed stage.
- Redis Streams + consumer group: entries stay pending until `XACK`. Crashed-consumer entries are reclaimed by `XAUTOCLAIM` after a configurable idle threshold. No in-flight job is silently lost.
- Postgres is the source of truth for state; Redis holds only the queue.

## Design Decisions

**Redis Streams for the queue.** `XREADGROUP` claims an entry and delivers it to exactly one consumer. The worker calls `XACK` on success. If a worker crashes before acking, the entry stays in the pending-entries list and `XAUTOCLAIM` reassigns it after a configurable idle threshold. Each worker registers as `hostname:pid` so the reclaim logic can tell a slow live consumer from a dead one. The pending-entries list is a first-class structure, directly inspectable and exposed as the `queue_depth` Prometheus gauge.

**Stage checkpoints.** The worker writes `job.stage` to Postgres after each stage completes. On retry, it reads that pointer and resumes from the last successful stage. Checkpointing costs one extra DB write per stage and bounds retry cost to the remaining stages only.

**Zero-shot classification as tier 1.** Zero-shot requires no labeled training data and generalises immediately to contract types not seen at development time. The confidence threshold is tunable without retraining. The accuracy ceiling is lower than a supervised model; tier-2 improves precision where labeled data exists.

**Cascade to fine-tuned span extraction as tier 2.** Zero-shot classification assigns a label; it cannot extract the exact clause text. Fine-tuning a span extractor on CUAD raises trimmed macro F1 from 0.41 to 0.73 and returns a verbatim span the reviewer can read directly in the report. Tier-2 fires only when tier-1 confidence is above threshold and the label maps to a CUAD category. A per-chunk timeout ensures slow GPU inference falls back to tier-1 labels rather than blocking the worker.

**Rule layer on top of ML.** ML assigns a clause type. Rules produce specific, defensible signals: matched text, exact character offsets, a rule ID, and a human-readable rationale. That output is what makes a report reviewable. Rules are declared as YAML and loaded into a spaCy Matcher; adding a pattern is a data change.

**MinIO for artifacts.** Report payloads can reach several megabytes on long contracts. MinIO keeps blobs out of the database and exposes S3-compatible APIs so production can swap to S3 or GCS without changing application code.

**Models loaded once at startup.** Loading a transformer model from disk takes several seconds. Workers load all models at process start and hold them for the process lifetime. Cold-start cost is paid once per worker restart.