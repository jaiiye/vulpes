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

#: 4xx 里这几个是「等一下再来」而不是「请求错了」，所以值得重试。其余 4xx
#: 重试无益（400/401/404 不会因为再发一次而变成 200）。
#:
#: 429 是篮子拉取时遇到的真问题：一次建仓要拉近 90 个币的 K 线，串着发会撞上
#: 限速，而它被当成 4xx 直接抛出时，缺失的币就静默地少了 —— 池从 89 掉到 79，
#: 腿从 17 掉到 15，而且掉哪些币取决于谁先撞上限速，不是随机的。
RETRYABLE_4XX = frozenset({408, 425, 429})

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
            if exc.code < 500 and exc.code not in RETRYABLE_4XX:
                raise HttpError(
                    f"{method} {url} -> HTTP {exc.code}: {detail}"
                ) from exc
            # 限速比服务端错误更需要等：同一个原因（发得太快）会让下一次尝试
            # 也失败，所以 429 的退避加倍。
            if exc.code == 429:
                time.sleep(backoff * 2)
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
