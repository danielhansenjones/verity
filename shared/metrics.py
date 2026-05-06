"""Prometheus metric definitions shared between the API and worker.

Metrics are module-level singletons registered against the default registry, so any
import within the same process contributes to the same counters and histograms.
"""

from prometheus_client import Counter, Gauge, Histogram

# Labels:
#   outcome: "created" for a new job, "replayed" for an idempotency hit.
jobs_submitted_total = Counter(
    "jobs_submitted_total",
    "POST /jobs outcomes by whether a new job was created or a prior one replayed.",
    labelnames=("outcome",),
)

# Labels:
#   status: "completed" or "failed".
jobs_completed_total = Counter(
    "jobs_completed_total",
    "Jobs that reached a terminal state.",
    labelnames=("status",),
)

# Wall-clock time spent in each pipeline stage. Buckets target realistic
# inference latencies on CPU and GPU.
job_stage_duration_seconds = Histogram(
    "job_stage_duration_seconds",
    "Wall-clock time per pipeline stage.",
    labelnames=("stage",),
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

job_stage_errors_total = Counter(
    "job_stage_errors_total",
    "Pipeline stage failures by stage.",
    labelnames=("stage",),
)

# Current queue depth. The worker samples this on every dequeue loop tick.
queue_depth = Gauge(
    "queue_depth",
    "Number of jobs waiting in the queue at last sample.",
)

# Labels:
#   outcome: "answered" (model answered with citations), "refused" (model
#            refused on insufficient evidence), "error" (grounding failure
#            or upstream error before a response was returned).
rag_questions_total = Counter(
    "rag_questions_total",
    "Outcomes of /jobs/{id}/ask requests.",
    labelnames=("outcome",),
)

rag_retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Wall-clock time spent embedding the query and pulling top-k chunks.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

rag_generation_latency_seconds = Histogram(
    "rag_generation_latency_seconds",
    "Wall-clock time for the LLM call that produces the structured answer.",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0),
)

# Labels:
#   direction: "in" (request input), "out" (model output),
#              "cache_read" (cached input reused), "cache_creation"
#              (input tokens written to cache for the first time).
rag_tokens_total = Counter(
    "rag_tokens_total",
    "Anthropic API token usage attributed to /ask.",
    labelnames=("direction",),
)
