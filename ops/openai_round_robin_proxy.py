from __future__ import annotations

import os
import urllib.error
import urllib.request
from itertools import count

from fastapi import FastAPI, Request, Response


BACKENDS = [
    base_url.rstrip("/")
    for base_url in os.environ["OPENAI_PROXY_BACKENDS"].split(",")
    if base_url.strip()
]
REQUEST_COUNTER = count()

app = FastAPI()


def _backend() -> str:
    if not BACKENDS:
        raise RuntimeError("OPENAI_PROXY_BACKENDS is empty")
    return BACKENDS[next(REQUEST_COUNTER) % len(BACKENDS)]


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request) -> Response:
    target = f"{_backend()}/{path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }
    upstream_request = urllib.request.Request(
        target,
        data=await request.body(),
        headers=headers,
        method=request.method,
    )
    try:
        upstream = urllib.request.urlopen(upstream_request, timeout=None)  # noqa: S310
        status_code = upstream.status
        response_headers = dict(upstream.headers.items())
        content = upstream.read()
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        response_headers = dict(exc.headers.items())
        content = exc.read()
    return Response(
        content=content,
        status_code=status_code,
        headers={
            key: value
            for key, value in response_headers.items()
            if key.lower() not in {"content-encoding", "transfer-encoding", "connection"}
        },
    )
