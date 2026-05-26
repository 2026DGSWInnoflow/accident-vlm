# Local Qwen 35B Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the project a focused blackbox-video preprocessing -> local Qwen3.6-35B-A3B VLM -> schema JSON pipeline.

**Architecture:** Keep the existing preprocessing pipeline, evidence builder, Transformers-based Qwen backend, JSON parsing, retry, fallback, and schema guard. Remove external OpenAI-compatible/vLLM/SGLang serving code, proxy code, tests, and documentation so the VLM path is local Qwen3.6-35B-A3B only.

**Tech Stack:** Python 3.11+, Pydantic, OpenCV, Transformers, pytest, FastAPI for the project API.

---

### Task 1: Lock Local-Only Backend Behavior

**Files:**
- Modify: `tests/test_vlm_composer.py`
- Modify: `src/accident_vlm/modules/vlm_composer.py`

- [x] **Step 1: Add a failing test**

Add a test showing `ACCIDENT_VLM_BACKEND=openai` no longer selects an HTTP backend.

- [x] **Step 2: Run the focused test**

Run: `pytest tests/test_vlm_composer.py::test_get_qwen_backend_ignores_external_backend_env -q`
Expected before implementation: fail because the current code returns `OpenAICompatibleVLMBackend`.

- [x] **Step 3: Remove external backend selection**

Change `_get_qwen_backend()` so it always returns `TransformersQwenBackend`.

- [x] **Step 4: Run the focused test**

Run: `pytest tests/test_vlm_composer.py::test_get_qwen_backend_ignores_external_backend_env -q`
Expected after implementation: pass.

### Task 2: Remove OpenAI-Compatible Serving Code

**Files:**
- Modify: `src/accident_vlm/modules/vlm_composer.py`
- Modify: `tests/test_vlm_composer.py`
- Delete: `ops/openai_round_robin_proxy.py`
- Delete: `tests/test_openai_round_robin_proxy.py`

- [x] **Step 1: Delete OpenAI-compatible backend implementation**

Remove `OpenAICompatibleVLMBackend`, `build_interleaved_content`, `_openai_headers`, `_read_http_error_detail`, and `_image_data_url`.

- [x] **Step 2: Delete tests for removed HTTP/proxy behavior**

Remove OpenAI-compatible backend tests and proxy tests. Keep chunking/fallback tests for `TransformersQwenBackend`.

- [x] **Step 3: Run VLM tests**

Run: `pytest tests/test_vlm_composer.py -q`
Expected: all remaining VLM composer tests pass.

### Task 3: Update Documentation and Verify

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-05-19-accident-video-facts-design.md`

- [x] **Step 1: Rewrite README VLM section**

Document only local Qwen3.6-35B-A3B usage and remove AWQ/vLLM/SGLang/OpenAI proxy guidance.

- [x] **Step 2: Run targeted tests**

Run: `pytest tests/test_vlm_composer.py tests/test_server_options.py tests/test_config.py -q`
Expected: pass.

- [x] **Step 3: Run full test suite**

Run: `pytest -q`
Expected: pass.

- [x] **Step 4: Commit and push**

Run:
`git add README.md docs/superpowers/specs/2026-05-19-accident-video-facts-design.md src tests ops docs/superpowers/plans/2026-05-26-local-qwen35-cleanup.md`
`git commit -m "refactor: focus pipeline on local qwen35 vlm"`
`git push origin main`
