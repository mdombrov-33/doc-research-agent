from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config import get_settings

settings = get_settings()


def _client_ip(request: Request) -> str:
    """Key requests on the client IP. Behind Cloud Run / a proxy the real client is the
    first entry in X-Forwarded-For; otherwise fall back to the socket address."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(
    key_func=_client_ip,
    storage_uri=settings.RATE_LIMIT_STORAGE_URI,
    enabled=settings.RATE_LIMIT_ENABLED,
)


def rate_limit() -> str:
    """Limit value resolved per-request so config/tests can vary it."""
    return get_settings().RATE_LIMIT
