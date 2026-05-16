import io
import json
import logging

from minio import Minio
from minio.error import S3Error

from shared.settings import settings

logger = logging.getLogger(__name__)


class StorageClient:
    def __init__(self):
        self._client = Minio(
            f"{settings.minio_host}:{settings.minio_port}",
            access_key=settings.minio_root_user,
            secret_key=settings.minio_root_password,
            secure=False,
        )
        self._bucket = settings.minio_bucket
        self._ensure_bucket()

    def _ensure_bucket(self):
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)
            logger.info("storage: created bucket %s", self._bucket)

    def upload_bytes(
        self,
        object_key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        try:
            self._client.put_object(
                self._bucket,
                object_key,
                io.BytesIO(data),
                length=len(data),
                content_type=content_type,
            )
        except S3Error as exc:
            logger.error("storage: upload failed key=%s: %s", object_key, exc)
            raise
        return object_key

    def delete_object(self, object_key: str) -> None:
        try:
            self._client.remove_object(self._bucket, object_key)
        except S3Error as exc:
            logger.error("storage: delete failed key=%s: %s", object_key, exc)
            raise

    def download_bytes(self, object_key: str) -> bytes:
        try:
            response = self._client.get_object(self._bucket, object_key)
        except S3Error as exc:
            logger.error("storage: download failed key=%s: %s", object_key, exc)
            raise
        try:
            return response.read()
        finally:
            response.close()

    def upload_json(self, object_key: str, payload: dict) -> str:
        data = json.dumps(payload, indent=2).encode("utf-8")
        return self.upload_bytes(object_key, data, content_type="application/json")

    def presigned_url(self, object_key: str, expires_seconds: int = 3600) -> str:
        from datetime import timedelta

        try:
            return self._client.presigned_get_object(
                self._bucket,
                object_key,
                expires=timedelta(seconds=expires_seconds),
            )
        except S3Error as exc:
            logger.error("storage: presign failed key=%s: %s", object_key, exc)
            raise
