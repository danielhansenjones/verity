import hmac
import logging
from typing import Optional

from fastapi import Header, HTTPException, status

from shared.settings import settings

logger = logging.getLogger(__name__)

_warned_missing_key = False


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    global _warned_missing_key

    expected = settings.contract_api_key
    if not expected:
        if not _warned_missing_key:
            logger.warning(
                "api: CONTRACT_API_KEY is not set; API authentication is disabled"
            )
            _warned_missing_key = True
        return

    # hmac.compare_digest avoids timing side channels on key comparison.
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
