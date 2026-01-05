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

    job_queue_key: str = "contract_jobs"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
