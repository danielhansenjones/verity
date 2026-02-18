# Contract Risk Pipeline

[![CI](https://github.com/danielhansenjones/contract_risk_pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/danielhansenjones/contract_risk_pipeline/actions/workflows/ci.yml)

A production-minded document processing pipeline that ingests contract PDFs, classifies clauses using zero-shot ML, applies rule-based risk flags, scores the document, and exposes results via a REST API.

Built to demonstrate: distributed job processing, ML inference in production, fault-tolerant worker design, and structured artifact storage.

## Stack

| Layer     | Technology               | Purpose                         |
|-----------|--------------------------|---------------------------------|
| API       | FastAPI                  | Job submission, status, results |
| Queue     | Redis (BRPOP/LPUSH)      | FIFO job dispatch, no busy-wait |
| Worker    | Python process           | Pipeline execution              |
| Database  | PostgreSQL + SQLAlchemy  | Job state, chunks, results      |
| Storage   | MinIO (S3-compatible)    | Raw PDFs, report artifacts      |
| ML        | HuggingFace Transformers | Clause classification + scoring |
| Container | Docker Compose           | Single-command local stack      |

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

| Method | Route                   | Description                                       |
|--------|-------------------------|---------------------------------------------------|
| POST   | `/jobs`                 | Upload PDF, enqueue job - returns `job_id`        |
| GET    | `/jobs/{job_id}`        | Job status, stage, retry count, and error if any  |
| GET    | `/jobs/{job_id}/report` | Risk result + presigned MinIO URL for full report |
| GET    | `/jobs`                 | List recent jobs, optional `?status=` filter      |
| GET    | `/health`               | Postgres and Redis connectivity check             |

## Auth

Protected endpoints require the `X-API-Key` header. The expected value is read from the `CONTRACT_API_KEY` env var. `GET /health` is public.

```bash
curl -H "X-API-Key: $CONTRACT_API_KEY" http://localhost:8000/jobs
```

If `CONTRACT_API_KEY` is unset, auth is disabled and a warning is logged at startup. Production deploys must set the key. Constant-time comparison is used to reject wrong keys without timing leaks.

## Upload limits

`POST /jobs` caps the raw request body at `MAX_UPLOAD_BYTES` (default 25 MiB). Oversized uploads are rejected early by middleware with `413 Payload Too Large` before any body is parsed or written to MinIO. Requests without a `Content-Length` header are rejected with `411 Length Required`. A defensive second check on actual bytes catches forged headers.

## Rate limits

Per-IP via slowapi. `POST /jobs` is capped at `RATE_LIMIT_SUBMIT` (default 30/minute); read endpoints (`GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/report`) share `RATE_LIMIT_READ` (default 120/minute). `GET /health` is exempt. Breach returns `429 Too Many Requests` with a `Retry-After` header. Per-subject limits are queued for once JWT auth lands.

## Pipeline

Processing runs in four sequential stages. On retry, the worker resumes from the last successful stage - a transient failure during scoring does not re-run ingestion.

```
ingestion → classification → scoring → assembly
```

### Stage 1 - Ingestion
Downloads the PDF from MinIO, extracts text with `pypdf`, and chunks by structural markers (numbered clauses, ALL CAPS headings, paragraph breaks). Falls back to 400-token splits only when a section is oversized.

### Stage 2 - Classification
Runs `facebook/bart-large-mnli` zero-shot classification against 10 clause labels (indemnification, termination, liability limitation, governing law, payment terms, IP assignment, confidentiality, dispute resolution, warranty, force majeure). Confidence threshold: 0.4; below that, chunk is labeled `general`. Chunks are batched in groups of 8.

### Stage 3 - Scoring
Two-pass scoring per chunk:
- **Tone model** (`distilbert-base-uncased-finetuned-sst-2-english`): negative sentiment raises risk signal.
- **Rule-based flags**: regex/keyword patterns for high-confidence, explainable risks (e.g. `"sole discretion"` → high, `"automatic renewal"` → medium).

Chunk score formula:
```
chunk_score = (tone * 0.3 + flag_severity * 0.4 + clause_type_weight * 0.3) * 100
```

Document score = weighted average of chunk scores. Thresholds: `low < 35`, `35 <= medium < 65`, `high >= 65`.

### Stage 4 - Assembly
Aggregates chunk scores and flags into a `RiskResult`, writes the full JSON report to MinIO, and marks the job `completed`.

## Architecture and scaling

Four tiers are deliberately separated: API, queue, worker, storage. Each scales, fails, and deploys independently. Full design rationale lives in [`docs/DESIGN.md`](docs/DESIGN.md).

### Scaling axes

- **API tier**: stateless. Horizontal scale behind a load balancer. No in-process queue, no session state. Idempotency makes retried submissions safe (see `docs/DESIGN.md` section 3).
- **Worker tier**: one consumer per job via atomic Redis dequeue. Add workers to raise throughput: `docker compose up --scale worker=N`. The worker loop and queue semantics are already correct for concurrent consumers.
- **Queue**: Redis. BRPOP today; migration to Redis Streams with consumer groups is queued so crashed workers have their in-flight jobs reclaimed rather than lost.
- **State**: Postgres. Write throughput is not the bottleneck at realistic ML inference rates, so a single primary is fine. Read replicas are the scale-out path for dashboards and audits.
- **Blobs**: MinIO. S3-compatible so production can swap to S3/GCS without code changes.

### Why not a monolith

A FastAPI + BackgroundTasks + SQLite deployment would be a few hundred lines smaller. It would also:

- Lose job durability across API restarts. In-memory tasks die with the process.
- Couple API latency to ML inference cost. A 30 s classification run blocks the event loop.
- Cap throughput at one machine. No horizontal path.
- Lose crash isolation. A pypdf OOM in the worker takes down the API with it.

The split costs a docker-compose.yml. The payoff is independent failure, scale, and deploy per tier.

### Current deployment shape vs designed shape

The repo ships with one worker. The architecture supports N. The bottleneck at N=1 is GPU inference on classification; scale-out is linear in worker count until Redis becomes the constraint, which it will not at the contract throughput this system targets.

### What the split does not buy you

Multi-tenant isolation, geographic replication, zero-downtime deploys, HA Postgres. Those are deployment-layer concerns and are explicitly out of scope for a single-zone single-tenant operation. Tracked in `docs/DESIGN.md` future work.

## CI

Two workflows run on every push and pull request to `main`:

| Workflow       | Jobs                                          |
|----------------|-----------------------------------------------|
| `ci.yml`       | Flake8 lint, pytest                  |
| `docker.yml`   | Docker image build (validates the Dockerfile) |

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

Both are plain Python files - point PyCharm's Run/Debug buttons directly at them.

## Project Structure

```
contract-risk-pipeline/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .pre-commit-config.yaml
├── .github/workflows/
│   ├── ci.yml
│   └── docker.yml
├── scripts/
│   ├── run.py                     # Start stack (PyCharm run button)
│   └── test.py                    # Run test suite (PyCharm test button)
├── shared/
│   ├── settings.py                # Pydantic settings - single env var source
│   ├── models.py                  # SQLAlchemy: Job, Chunk, RiskResult
│   ├── redis_queue.py             # Queue wrapper (LPUSH enqueue, BRPOP dequeue)
│   └── minio_client.py            # MinIO wrapper (upload, download, presigned URL)
├── api/
│   └── main.py                    # FastAPI routes
├── worker/
│   ├── main.py                    # Worker loop + model loading
│   └── processors/
│       ├── ingestion.py           # PDF download, text extraction, chunking
│       ├── classifier.py          # Zero-shot clause classification
│       ├── scorer.py              # Risk scoring + rule-based flags
│       └── assembler.py           # Report assembly + persistence
└── tests/
    ├── conftest.py                # Fixtures: SQLite engine, mocks, factories
    ├── seed.py                    # Synthetic PDF factory + sample contract data
    ├── test_documents/            # Real contract PDFs used in smoke tests
    ├── test_api.py
    ├── test_assembler.py
    ├── test_classifier.py
    ├── test_ingestion.py
    ├── test_scorer.py
    └── test_seed.py
```

## Fault Tolerance

- Worker retries up to `max_retries` (default 3) before marking a job `failed`
- `job.stage` is preserved on failure - retry resumes from the last completed stage
- Postgres is the source of truth for all states; Redis holds only the queue.

## Design Decisions

**Redis for the queue** - BRPOP gives atomic, blocking dequeue with no busy-wait. No two workers can pop the same job. Scale horizontally by adding worker containers.

**Stage checkpoints** - Prevents redundant reprocessing on transient failures. A network blip during scoring shouldn't re-extract and re-classify a 100-page document.

**Zero-shot classification** - No labeled training data required. Immediately deployable to new contract types without retraining. The confidence threshold (0.4) is tunable.

**Hybrid ML + rules** - ML assigns a clause type; rules produce specific, explainable risk signals that a lawyer or executive can act on. This mirrors production legal AI systems.

**MinIO for artifacts** - Report payloads can be large. Storing JSON blobs in Postgres is an anti-pattern at scale. MinIO is S3-compatible locally and trivially swappable for S3 in production.

**Models loaded once at startup** - No cold-start cost per job. Single memory allocation for keep the worker process lifetime.