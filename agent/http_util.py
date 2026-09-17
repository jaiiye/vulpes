"""Minimal stdlib HTTP helper with retries.

Deliberately dependency-free so the agent runs before any pip install:
the data layer only needs GET requests against public JSON endpoints.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from typing import Any


class HttpError(RuntimeError):
    """Raised when a request fails after all retries."""


DEFAULT_TIMEOUT = 15
DEFAULT_RETRIES = 3
USER_AGENT = "fox-lite-agent/0.1"

# Reading the body in chunks lets us retry a truncated transfer: a single
# `resp.read()` that dies partway cannot be resumed, and the leaderboard
# response is ~37 MB.
CHUNK_SIZE = 1 << 20  # 1 MB

# Ceiling on a single response body. The largest legitimate response is the
# ~37 MB leaderboard; this turns a misbehaving endpoint (streaming forever)
# into a clear error instead of an unbounded hang or memory growth.
MAX_RESPONSE_BYTES = 256 << 20  # 256 MB


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    method: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff: float = 0.8,
    headers: dict[str, str] | None = None,
) -> Any:
    """Perform a JSON request and return the decoded body.

    Retries on connection errors, truncated transfers and 5xx responses with
    exponential backoff. 4xx responses fail immediately, since retrying will
    not help.

    `headers` adds caller-supplied headers (for example an Authorization
    token). Everything else goes through this one function so that retries,
    the truncated-transfer guard and the response size ceiling apply
    uniformly, instead of being re-implemented per caller.
    """
    method = method or ("POST" if payload is not None else "GET")
    body = json.dumps(payload).encode() if payload is not None else None
    request_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        # Compression on multi-megabyte responses causes truncated transfers
        # in practice; the bandwidth saving is not worth the failure mode.
        "Accept-Encoding": "identity",
    }
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            url, data=body, headers=request_headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > MAX_RESPONSE_BYTES:
                        raise HttpError(
                            f"{method} {url} exceeded {MAX_RESPONSE_BYTES} bytes; "
                            "aborting rather than buffering indefinitely"
                        )
                raw = b"".join(chunks).decode("utf-8")
            return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:300]
            except Exception:  # pragma: no cover - best effort
                pass
            if exc.code < 500:
                raise HttpError(
                    f"{method} {url} -> HTTP {exc.code}: {detail}"
                ) from exc
            last_error = exc
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            http.client.IncompleteRead,
            ConnectionError,
        ) as exc:
            last_error = exc

        if attempt < retries:
            time.sleep(backoff * attempt)

    raise HttpError(f"{method} {url} failed after {retries} attempts: {last_error}")
