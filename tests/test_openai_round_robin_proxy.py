import json
import os
import socket

from fastapi.testclient import TestClient

os.environ.setdefault("OPENAI_PROXY_BACKENDS", "http://backend-a")

import ops.openai_round_robin_proxy as proxy_module


def test_proxy_logs_vlm_request_metadata_and_uses_hard_timeout(monkeypatch, capsys):
    proxy_module.BACKENDS[:] = ["http://backend-a"]
    monkeypatch.setenv("OPENAI_PROXY_TIMEOUT_SEC", "12")
    captured = {}

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/json"}

        def read(self) -> bytes:
            return b'{"ok":true}'

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        captured["url"] = request.full_url
        return FakeResponse()

    monkeypatch.setattr(proxy_module.urllib.request, "urlopen", fake_urlopen)
    client = TestClient(proxy_module.app)

    response = client.post(
        "/v1/chat/completions",
        headers={
            "x-accident-vlm-image-count": "4",
            "x-accident-vlm-prompt-chars": "1234",
            "x-accident-vlm-max-tokens": "128",
            "x-accident-vlm-chunk": "chunk 1/2",
        },
        json={"messages": []},
    )

    assert response.status_code == 200
    assert captured["timeout"] == 12.0
    assert captured["url"] == "http://backend-a/v1/chat/completions"
    log = capsys.readouterr().out
    assert "images=4" in log
    assert "prompt_chars=1234" in log
    assert "max_tokens=128" in log
    assert "chunk=chunk 1/2" in log


def test_proxy_returns_502_when_upstream_disconnects(monkeypatch):
    proxy_module.BACKENDS[:] = ["http://backend-a"]

    def fake_urlopen(_request, timeout):
        raise socket.timeout("timed out")

    monkeypatch.setattr(proxy_module.urllib.request, "urlopen", fake_urlopen)
    client = TestClient(proxy_module.app)

    response = client.post("/v1/chat/completions", json={"messages": []})

    assert response.status_code == 502
    assert "timed out" in response.text
