from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    postgres_user: str = "contractuser"
    postgres_password: str = "contractpass"
    postgres_db: str = "contractdb"
    postgres_host: str = "localhost"

    redis_host: str = "localhost"
    redis_port: int = 6379

    minio_host: str = "localhost"
    minio_port: int = 9000
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin"
    minio_bucket: str = "contracts"

    # Redis Streams: the stream key, consumer group name, and reclaim threshold.
    # Entries idle for longer than this on a dead consumer are reclaimed via XAUTOCLAIM.
    # Idle threshold must exceed realistic p99 stage duration (zero-shot classification
    # over many chunks on CPU can run several minutes); otherwise a healthy worker's
    # in-flight entry gets stolen concurrently.
    job_queue_key: str = "contract_jobs"
    job_queue_group: str = "workers"
    job_queue_idle_ms: int = 600_000

    # Dead-letter stream for entries that have been delivered more than
    # job_queue_max_deliveries times. Protects against poison-pill jobs that
    # crash the worker process (OOM, SIGKILL) before app-layer retry logic runs.
    job_queue_dlq_key: str = "contract_jobs:dlq"
    job_queue_max_deliveries: int = 5

    # XADD maxlen trim target. XACK removes from PEL but not from the stream;
    # without trimming the stream grows unbounded.
    job_queue_maxlen: int = 10_000

    # Unset disables auth with a startup warning; production deploys must set it.
    contract_api_key: Optional[str] = None

    # Cap on the raw HTTP body of POST /jobs. 25 MiB default.
    max_upload_bytes: int = 26_214_400

    # Per-IP rate limits. slowapi syntax: "<count>/<period>".
    rate_limit_submit: str = "30/minute"
    rate_limit_read: str = "120/minute"

    # Port for the worker's /metrics HTTP server. API metrics ride on the API port.
    worker_metrics_port: int = 9100

    # Span extractor (v2 cascade). Disabled by default until the model is trained.
    span_extractor_enabled: bool = False
    span_extractor_model_path: str = ""
    span_extractor_tier1_confidence_threshold: float = 0.7
    span_extractor_timeout_s: float = 30.0

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
