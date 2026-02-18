from slowapi import Limiter
from slowapi.util import get_remote_address

from shared.settings import settings

limiter = Limiter(key_func=get_remote_address)


# Callables so settings can be patched in tests and re-read at request time.
def submit_limit() -> str:
    return settings.rate_limit_submit


def read_limit() -> str:
    return settings.rate_limit_read
