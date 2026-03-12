from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from .config import API_KEY, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS

_RATE_LIMIT_STATE: dict[str, deque[float]] = defaultdict(deque)
_RATE_LIMIT_LOCK = Lock()
api_key_header_scheme = APIKeyHeader(name="x-api-key", auto_error=False)


def reset_rate_limit_state() -> None:
    with _RATE_LIMIT_LOCK:
        _RATE_LIMIT_STATE.clear()


def require_api_key(request: Request, provided_key: str | None = None) -> None:
    if not API_KEY:
        return

    resolved_key = provided_key if provided_key is not None else request.headers.get("x-api-key", "")
    if resolved_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized. Provide x-api-key header or use /docs Authorize.")


def authorize_request(request: Request, api_key: str | None = Security(api_key_header_scheme)) -> None:
    require_api_key(request, provided_key=api_key)
    enforce_rate_limit(request)


def enforce_rate_limit(request: Request) -> None:
    if RATE_LIMIT_REQUESTS <= 0:
        return

    client_id = request.headers.get("x-forwarded-for")
    if not client_id:
        client_id = request.client.host if request.client is not None else "unknown"

    now = monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    with _RATE_LIMIT_LOCK:
        request_times = _RATE_LIMIT_STATE[client_id]
        while request_times and request_times[0] < window_start:
            request_times.popleft()

        if len(request_times) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        request_times.append(now)
