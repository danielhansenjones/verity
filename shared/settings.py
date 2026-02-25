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
    job_queue_key: str = "contract_jobs"
    job_queue_group: str = "workers"
    job_queue_idle_ms: int = 60_000

    # Unset disables auth with a startup warning; production deploys must set it.
    contract_api_key: Optional[str] = None

    # Cap on the raw HTTP body of POST /jobs. 25 MiB default.
    max_upload_bytes: int = 26_214_400

    # Per-IP rate limits. slowapi syntax: "<count>/<period>".
    rate_limit_submit: str = "30/minute"
    rate_limit_read: str = "120/minute"

    # Port for the worker's /metrics HTTP server. API metrics ride on the API port.
    worker_metrics_port: int = 9100

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
