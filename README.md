<div align="center">

# Verity

**Fault-Tolerant Contract-Risk Pipeline with Grounded RAG and LLM-as-Judge Evals**

[![CI](https://github.com/danielhansenjones/verity/actions/workflows/ci.yml/badge.svg)](https://github.com/danielhansenjones/verity/actions/workflows/ci.yml)
[![faithfulness](https://img.shields.io/badge/RAG%20faithfulness-0.97-success)](#results)
[![citations](https://img.shields.io/badge/citation%20accuracy-1.00-success)](#results)
[![F1](https://img.shields.io/badge/span%20macro%20F1-0.41%20to%200.73-success)](#results)
[![tests](https://img.shields.io/badge/tests-169%20passing-success)](#results)

[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Redis](https://img.shields.io/badge/Redis%207-Streams-DC382D?logo=redis&logoColor=white)](https://redis.io/docs/latest/develop/data-types/streams/)
[![Postgres](https://img.shields.io/badge/Postgres%2016-pgvector-4169E1?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Sonnet%20%2B%20Haiku-D97757?logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![MinIO](https://img.shields.io/badge/MinIO-S3-C72E49?logo=minio&logoColor=white)](https://min.io/)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)](docker-compose.yml)

</div>

A distributed document processing pipeline built on Redis Streams, PostgreSQL, and MinIO.
Accepts PDF uploads over a hardened REST API, queues jobs for async worker processing, and returns structured risk reports from a two-tier ML cascade:
BART-MNLI zero-shot clause classification feeding into fine-tuned RoBERTa span extraction on flagged clauses, with rule-based risk flags and verbatim evidence spans.

Legal contract review is slow and expensive.
Lawyers spend time locating clauses that a well-built system can classify and flag in seconds.
This pipeline handles that pre-screening layer: classify every clause, match risk patterns with exact character offsets, and extract verbatim spans from flagged sections using a model trained on 500 real contracts.
The goal is to automate the repetitive part and surface structured evidence so that review is faster and more consistent.
The judgment on what to negotiate stays with the lawyer.

## Highlights

A few flagship properties up front; grouped detail below is collapsed to keep this scannable.

- **At-least-once delivery** with crash recovery: Redis Streams consumer groups plus `XAUTOCLAIM` reclaim of dead workers. No in-flight job is silently lost.
- **Two-tier ML cascade** lifts span-extraction trimmed macro F1 from 0.41 (zero-shot) to 0.73 (fine-tuned base).
- **Grounded RAG**: a post-generation citation check rejects fabricated quotes or chunk ids with `502`.

<details>
<summary><b>Reliability and delivery</b> - idempotency, stage-checkpointed retries</summary>

- **At-least-once delivery** via Redis Streams consumer groups. Jobs stay in the pending-entries list until `XACK`. Crashed workers are reclaimed by `XAUTOCLAIM` after a configurable idle threshold; no in-flight job is silently lost.
- **Idempotent `POST /jobs`** with client-supplied or content-hash dedup keys. Concurrent first-time submissions race at the DB layer via a unique constraint; the loser returns the winner's `job_id` with `Idempotent-Replay: true`.
- **Stage-checkpointed retries.** The worker persists which pipeline stage completed before a failure. A retry resumes from the last successful stage; a transient scoring error does not re-run ingestion or classification.

</details>

<details>
<summary><b>API hardening</b> - timing-safe auth, pre-parse upload caps, rate limits</summary>

- **Hardened API layer.** Timing-safe key auth (`hmac.compare_digest`), pre-parse upload cap via middleware (411 and 413 before multipart parsing is attempted), and per-IP rate limits with `Retry-After`.

</details>

<details>
<summary><b>Observability and load</b> - Prometheus on both tiers, Locust results</summary>

- **Prometheus metrics on both tiers.** API and worker each expose a scrape target. Tracked series: submissions, completions, per-stage duration histograms, per-stage error counts, and queue depth.
- **Empirically validated under Locust.** 9.6 jobs/min sustained at one worker, submit p50 14ms under load and 62ms under a 20 VU spike, zero server errors across both. Rate limiting fires with `Retry-After`; span extractor timeout degrades to tier-1 labels without failing the job.

</details>

<details>
<summary><b>ML cascade</b> - BART-MNLI zero-shot into fine-tuned RoBERTa span extraction</summary>

- **Cascade ML pipeline.** BART-MNLI zero-shot classifies a clause type across 10 labels. A fine-tuned RoBERTa model trained on CUAD v1 extracts verbatim spans when tier-1 confidence meets the threshold and the clause maps to a CUAD category. Trimmed macro F1 goes from 0.41 (zero-shot) to 0.73 (fine-tuned base). A spaCy `Matcher` and YAML rule DSL layer produces explainable risk flags with exact character offsets. Per-chunk extraction timeout falls back to tier-1 without failing the job.

</details>

<details>
<summary><b>RAG and evals</b> - grounded citations, audit log, LLM-as-judge harness</summary>

- **RAG query endpoint with verified citations.** `POST /jobs/{id}/ask` runs free-text questions against the contract's chunks. Local BGE-small embeddings stored in a pgvector column on Postgres, cosine-similarity retrieval, Claude generation with structured outputs, and a post-generation grounding check that rejects fabricated quotes or chunk ids with `502`. The deterministic pipeline remains fully functional without this endpoint.
- **Auditable RAG calls.** Every `/ask` is persisted to a `rag_queries` table: the question, ordered retrieved chunk ids, answer, citations, token usage, retrieval and generation latency, and the terminal outcome (answered, refused, or error with the grounding failure). Storing chunk ids rather than text reconstructs the exact prompt for any past call, since chunks are immutable. Writes are best-effort and never fail a good answer.
- **Hand-rolled RAG eval harness with LLM-as-judge.** 30-case dataset scored across faithfulness, citation accuracy, completeness, and refusal correctness. Judge model (`claude-haiku-4-5`) is deliberately different from the generator (`claude-sonnet-4-6`) to reduce self-grading bias. First-run aggregate 0.967 / 1.000 / 0.967 / 1.000 with multi-clause synthesis as the documented soft spot at 0.833 faithfulness. CI re-runs a frozen subset on PRs touching the RAG surface.

</details>

<details>
<summary><b>Testing and ops</b> - 169 tests, single-command stack</summary>

- **169 tests** covering API contracts, pipeline stages, queue semantics, auth, rate limits, idempotency, upload sizing, span extraction, timeout fallback, embeddings, citation grounding, and the RAG endpoint.
- **Single-command local stack** via Docker Compose (Postgres+pgvector, Redis, MinIO, API, worker).

</details>

## Architecture

Four tiers, deliberately separated. Each scales, fails, and deploys independently.

```mermaid
flowchart LR
    C(["Client"])
    C -->|"POST /jobs (PDF)"| API
    C -->|"GET status / report"| API
    C -->|"POST /ask"| API

    subgraph api_tier["API tier - stateless, scales horizontally"]
        API["FastAPI<br/>auth - rate limit - idempotency"]
    end

    API -->|enqueue| REDIS[("Redis Streams<br/>consumer group")]
    REDIS -->|"XREADGROUP / XAUTOCLAIM"| WORKER

    subgraph worker_tier["Worker tier - scales to N"]
        WORKER["Worker<br/>ingestion - classification - scoring - assembly"]
    end

    API <--> PG[("Postgres 16<br/>+ pgvector")]
    WORKER <--> PG
    API <--> MINIO[("MinIO<br/>S3-compatible")]
    WORKER <--> MINIO
    API -.->|"/ask: retrieve + generate"| ANTH(["Claude<br/>Anthropic SDK"])

    classDef store fill:#eef2ff,stroke:#6366f1,color:#1e1b4b;
    classDef ext fill:#fff7ed,stroke:#fb923c,color:#7c2d12;
    class REDIS,PG,MINIO store;
    class ANTH ext;
```

### Two-tier ML cascade

A cheap zero-shot classifier gates an expensive fine-tuned span extractor. Tier-2 only runs when tier-1 is confident and the clause maps to a CUAD category, and a per-chunk timeout degrades to the tier-1 label instead of failing the job.

```mermaid
flowchart TD
    CH["Clause chunk"] --> T1["BART-MNLI zero-shot<br/>10 clause labels"]
    T1 --> Q{"confidence meets threshold<br/>and maps to CUAD?"}
    Q -->|yes| T2["RoBERTa span extractor<br/>fine-tuned on CUAD v1"]
    Q -->|no| L1["Tier-1 label only"]
    T2 --> TO{"extraction timeout?"}
    TO -->|no| SPAN["Verbatim span + char offsets"]
    TO -->|yes| L1
    SPAN --> RULES["spaCy Matcher + YAML rule DSL<br/>explainable risk flags"]
    L1 --> RULES
    RULES --> OUT(["RiskResult"])

    classDef ok fill:#ecfdf5,stroke:#10b981,color:#064e3b;
    class OUT ok;
```

### RAG query flow (POST /ask)

Every answer passes a grounding gate before it leaves the box: cited quotes must be verbatim substrings of retrieved chunks and cited ids must be in the retrieved set, or the request 502s. Each call is persisted to an audit table.

```mermaid
flowchart LR
    Q(["Question"]) --> EMB["BGE-small<br/>query embedding"]
    EMB --> RET["pgvector cosine ANN<br/>HNSW, filtered by job_id"]
    RET --> TOPK["Top-k chunks"]
    TOPK --> GEN["Claude Sonnet<br/>forced tool_use: AnswerResponse"]
    GEN --> GROUND{"citations verbatim<br/>and in retrieved set?"}
    GROUND -->|pass| OK(["200<br/>answer + citations"])
    GROUND -->|fail| ERR(["502<br/>grounding error"])
    OK --> AUDIT[("rag_queries<br/>audit row")]
    ERR --> AUDIT

    classDef store fill:#eef2ff,stroke:#6366f1,color:#1e1b4b;
    classDef good fill:#ecfdf5,stroke:#10b981,color:#064e3b;
    classDef bad fill:#fef2f2,stroke:#ef4444,color:#7f1d1d;
    class RET,AUDIT store;
    class OK good;
    class ERR bad;
```

## Results

| Metric                              | Value                                                       |
|-------------------------------------|-------------------------------------------------------------|
| Span-extraction macro F1            | 0.41 zero-shot to 0.73 fine-tuned (RoBERTa-base on CUAD v1) |
| RAG faithfulness (LLM-as-judge)     | 0.967                                                       |
| Citation accuracy                   | 1.000                                                       |
| Completeness / refusal correctness  | 0.967 / 1.000                                               |
| Throughput, single worker           | 9.6 jobs/min                                                |
| Submit latency p50                  | 14 ms (62 ms under a 20 VU spike)                           |
| Server errors under load            | 0                                                           |
| Tests                               | 169                                                         |

![CUAD per-category F1: BART-MNLI zero-shot vs fine-tuned RoBERTa](assets/cuad_per_category_f1.png)

*Per-category F1 across all 41 CUAD categories. Blue is BART-MNLI zero-shot; orange and green are fine-tuned RoBERTa (base and large). Fine-tuning closes the largest gaps on categories zero-shot barely handles (Anti-Assignment, Effective Date, Governing Law, Expiration Date).*

RAG eval is a 30-case dataset scored by an LLM judge (`claude-haiku-4-5`) that is deliberately different from the generator (`claude-sonnet-4-6`) to reduce self-grading bias. Multi-clause synthesis is the documented soft spot at 0.833 faithfulness. Full methodology is in [cuad/README.md](cuad/README.md).


## Stack

| Layer      | Technology                                | Purpose                                    |
|------------|-------------------------------------------|--------------------------------------------|
| API        | FastAPI                                   | Job submission, status, results, /ask      |
| Queue      | Redis Streams + consumer group            | At-least-once dispatch with crash recovery |
| Worker     | Python process                            | Pipeline execution                         |
| Database   | PostgreSQL + pgvector + SQLAlchemy        | Job state, chunks, embeddings, results     |
| Storage    | MinIO (S3-compatible)                     | Raw PDFs, report artifacts                 |
| ML         | HuggingFace Transformers                  | Clause classification + scoring            |
| Embeddings | sentence-transformers (BGE-small-en-v1.5) | 384-dim chunk + query embeddings           |
| RAG LLM    | Anthropic SDK (claude-sonnet-4-6)         | Structured answer + citation generation    |
| Container  | Docker Compose                            | Single-command local stack                 |

## Quick Start

```bash
cp .env.example .env
python scripts/run.py
```

Or directly via Docker:
```bash
docker compose up --build
```

| Service        | URL                        |
|----------------|----------------------------|
| API            | http://localhost:8000      |
| API docs       | http://localhost:8000/docs |
| MinIO console  | http://localhost:9001      |
| Postgres       | localhost:5432             |

**Set up the RAG endpoint backend (optional, only needed for `/ask`):**

Set `ANTHROPIC_API_KEY=sk-ant-...` in `.env`. Key from `console.anthropic.com` (separate from a Claude.ai subscription).

**Submit a job:**
```bash
curl -X POST http://localhost:8000/jobs \
  -F "file=@sample_contract.pdf"
```

**Check status:**
```bash
curl http://localhost:8000/jobs/{job_id}
```

**Get report:**
```bash
curl http://localhost:8000/jobs/{job_id}/report
```

**Seed sample jobs:**
```bash
docker compose exec api python tests/seed.py
```

## API

| Method | Route                   | Description                                                 |
|--------|-------------------------|-------------------------------------------------------------|
| POST   | `/jobs`                 | Upload PDF, enqueue job - returns `job_id`                  |
| GET    | `/jobs/{job_id}`        | Job status, stage, retry count, and error if any            |
| GET    | `/jobs/{job_id}/report` | Risk result + presigned MinIO URL for full report           |
| POST   | `/jobs/{job_id}/ask`    | Free-text question, returns answer + grounded citations     |
| GET    | `/jobs`                 | List recent jobs, optional `?status=` filter                |
| GET    | `/health`               | Postgres and Redis connectivity check                       |
| GET    | `/metrics`              | Prometheus exposition (public, unauthenticated)             |

`POST /jobs/{id}/ask` requires `ANTHROPIC_API_KEY` to be set; otherwise the endpoint returns `503`. The rest of the API works without it.

Auth, rate limits, idempotency semantics, full request/response shapes, and the pipeline internals are in [TECHNICAL.md](TECHNICAL.md).

### Audit log retention and sensitive data

The `rag_queries` table grows without bound: every `/ask` appends a row and nothing prunes them. Production deployments should set a retention window, 90 days is a reasonable default, and delete older rows on a schedule. That pruner is not part of this change.

Questions and stored answers can carry PII or commercially sensitive terms, written verbatim to `rag_queries.question` and `rag_queries.answer`. Deployments handling real contracts should enable Postgres encryption at rest, restrict the database role that can read this table to the principals that actually need audit access, and consider field-level redaction at write time if the exposure scope warrants it.

## CI

Three workflows run on push and pull request to `main`:

| Workflow     | Jobs                                                       |
|--------------|------------------------------------------------------------|
| `ci.yml`     | Flake8 lint, pytest                                        |
| `docker.yml` | Docker image build (validates the Dockerfile)              |
| `evals.yml`  | Frozen RAG eval subset, on PRs touching the RAG surface    |

## Pre-commit

```bash
pip install pre-commit
pre-commit install
```

Runs on every commit: Flake8 lint, trailing whitespace, EOF, YAML and TOML validation.

## Scripts

| Script              | Purpose                                    |
|---------------------|--------------------------------------------|
| `scripts/run.py`    | Start the full stack via Docker Compose    |
| `scripts/test.py`   | Run the pytest suite (no services needed)  |

Both are plain Python files. Point PyCharm's Run/Debug buttons directly at them.

## Deeper reading

- [TECHNICAL.md](TECHNICAL.md) - auth, limits, idempotency, metrics, pipeline stages, RAG layer, architecture and scaling numbers, CUAD v2 cascade, project structure, fault tolerance, design decisions.
- [cuad/README.md](cuad/README.md) - CUAD training hyperparameters, data split methodology, per-category F1 results.
