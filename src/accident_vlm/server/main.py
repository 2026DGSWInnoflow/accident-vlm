from __future__ import annotations

import os

import uvicorn


def run() -> None:
    host = os.getenv("ACCIDENT_VLM_HOST", "0.0.0.0")
    port = int(os.getenv("ACCIDENT_VLM_PORT", "8000"))
    uvicorn.run("accident_vlm.server.app:app", host=host, port=port)


if __name__ == "__main__":
    run()
