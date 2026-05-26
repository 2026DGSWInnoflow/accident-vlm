import json
import urllib.error

from accident_vlm.modules.vlm_composer import (
    SYSTEM_PROMPT,
    build_interleaved_content,
    build_qwen_interleaved_content,
    build_vlm_prompt,
    compose_with_backend,
    compose_with_retry,
    compose_final_facts,
    get_qwen_backend,
    normalize_model_id,
    normalize_device,
    disable_transformers_allocator_warmup,
    OpenAICompatibleVLMBackend,
    TransformersQwenBackend,
    parse_json_response,
    render_qwen_chat_template,
    _configure_transformers_loading,
    _collect_evidence_image_paths,
    _format_non_retriable_vlm_error,
    _image_data_url,
    _is_retriable_vlm_error,
    _read_http_error_detail,
    _auto_cuda_max_memory,
    _model_dtype,
    _parse_max_memory,
    _prepare_transformers_runtime_env,
    _attn_implementation,
    _use_generation_cache,
)
from accident_vlm.schemas.preprocessing import PipelineContext


class RecordingBackend:
    def __init__(self) -> None:
        self.prompt = None
        self.image_paths = None
        self.image_records = None

    def generate_json(self, prompt: str, image_paths: list[str], image_records=None) -> dict:
        self.prompt = prompt
        self.image_paths = image_paths
        self.image_records = image_records
        return {"schema_version": "accident_video_facts.v1"}


class RecordingChunkBackend:
    def __init__(self) -> None:
        self.calls = []

    def _generate_text(
        self,
        prompt: str,
        image_paths: list[str],
        max_tokens: int,
        image_records=None,
        chunk_label=None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "image_paths": list(image_paths),
                "max_tokens": max_tokens,
                "image_records": image_records,
                "chunk_label": chunk_label,
            }
        )
        if image_paths:
            return f"{chunk_label} saw {len(image_paths)} images"
        return '{"schema_version":"accident_video_facts.v1","objective_summary":"ok"}'


class FlakyBackend:
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, prompt: str, image_paths: list[str], image_records=None) -> dict:
        self.calls += 1
        if self.calls == 1:
            raise ValueError("bad json")
        return {"schema_version": "accident_video_facts.v1", "objective_summary": "ok"}


class OomThenRecordingBackend:
    def __init__(self) -> None:
        self.image_counts: list[int] = []

    def generate_json(self, prompt: str, image_paths: list[str], image_records=None) -> dict:
        self.image_counts.append(len(image_paths))
        if len(self.image_counts) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.35 GiB.")
        return {"schema_version": "accident_video_facts.v1", "objective_summary": "ok"}


class OomUntilCompactBackend:
    def __init__(self) -> None:
        self.image_counts: list[int] = []
        self.prompts: list[str] = []

    def generate_json(self, prompt: str, image_paths: list[str], image_records=None) -> dict:
        self.image_counts.append(len(image_paths))
        self.prompts.append(prompt)
        if len(self.image_counts) < 3:
            raise RuntimeError("CUDA out of memory. Tried to allocate 13.37 GiB.")
        return {"schema_version": "accident_video_facts.v1", "objective_summary": "compact ok"}


class WeightPackedBackend:
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, prompt: str, image_paths: list[str], image_records=None) -> dict:
        self.calls += 1
        raise KeyError("weight_packed")


class InvalidJsonBackend:
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, prompt: str, image_paths: list[str], image_records=None) -> dict:
        self.calls += 1
        raise SyntaxError("'{'' was never closed")


def test_build_vlm_prompt_includes_system_prompt_schema_and_evidence_package() -> None:
    context = PipelineContext(
        video_path="sample.mp4",
        evidence_package={
            "frames": [{"id": "f001", "path": "/tmp/frame.jpg"}],
            "precomputed_facts": {"metadata": {"duration_sec": 6.0}},
        },
    )

    prompt = build_vlm_prompt(context)

    assert SYSTEM_PROMPT in prompt
    assert "accident_video_facts.v1" in prompt
    assert '"duration_sec": 6.0' in prompt
    assert "확인불가" in prompt


def test_compose_with_backend_sends_prompt_and_truthy_evidence_image_paths(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_FORCE_COMPACT_PROMPT", "0")
    context = PipelineContext(
        video_path="sample.mp4",
        evidence_package={
            "frames": [
                {"id": "f001", "path": "/tmp/frame-1.jpg"},
                {"id": "f002", "path": None},
                {"id": "f003", "path": ""},
                {"id": "f004", "path": "/tmp/frame-4.jpg"},
            ],
            "overlays": [
                {"id": "o001", "path": "/tmp/overlay-1.jpg"},
            ],
            "crops": [
                {"id": "c001", "path": "/tmp/crop-1.jpg"},
                {"id": "c002", "path": ""},
            ],
            "evidence_images": [
                {"id": "lane_overlay", "path": "/tmp/lane-overlay.jpg"},
                {"id": "signal_crop", "path": "/tmp/signal-crop.jpg", "purpose": "traffic_light_crop"},
            ],
            "precomputed_facts": {},
        },
    )
    backend = RecordingBackend()

    result = compose_with_backend(context, backend)

    assert result == {"schema_version": "accident_video_facts.v1"}
    assert backend.prompt == build_vlm_prompt(context)
    assert backend.image_paths == [
        "/tmp/signal-crop.jpg",
        "/tmp/crop-1.jpg",
        "/tmp/overlay-1.jpg",
        "/tmp/lane-overlay.jpg",
        "/tmp/frame-1.jpg",
        "/tmp/frame-4.jpg",
    ]


def test_parse_json_response_accepts_python_style_object() -> None:
    assert parse_json_response("{'schema_version': 'accident_video_facts.v1'}") == {
        "schema_version": "accident_video_facts.v1"
    }


def test_get_qwen_backend_reuses_instances(monkeypatch) -> None:
    created = []

    class FakeQwenBackend:
        def __init__(self, model_id: str, device: str = "auto") -> None:
            created.append((model_id, device))

    monkeypatch.setattr(
        "accident_vlm.modules.vlm_composer.TransformersQwenBackend",
        FakeQwenBackend,
    )
    get_qwen_backend.cache_clear()

    first = get_qwen_backend("Qwen/test", "auto")
    second = get_qwen_backend("Qwen/test", "auto")

    assert first is second
    assert created == [("Qwen/test", "auto")]


def test_build_vlm_prompt_includes_output_template_contract() -> None:
    context = PipelineContext(
        video_path="sample.mp4",
        evidence_package={"frames": [], "precomputed_facts": {}},
    )

    prompt = build_vlm_prompt(context)

    assert '"scene_type"' in prompt
    assert '"rag_hints"' in prompt
    assert '"objective_summary"' in prompt
    assert "Do not include markdown, prose, or reasoning" in prompt


def test_build_vlm_prompt_includes_storyboard_and_generic_candidate_checks() -> None:
    context = PipelineContext(
        video_path="sample.mp4",
        evidence_package={
            "vlm_storyboard": [
                {
                    "slot": 1,
                    "id": "frame_000001",
                    "path": "/tmp/frame.jpg",
                    "time": "00:01.000",
                    "phase": "impact_candidate",
                    "role": "충돌 후보 확인",
                    "why_selected": "event candidate",
                }
            ],
            "precomputed_facts": {},
        },
    )

    prompt = build_vlm_prompt(context)

    assert "VLM evidence storyboard" in prompt
    assert "차대보행자" in prompt
    assert "차대차" in prompt
    assert "차대이륜차" in prompt
    assert "단독사고" in prompt
    assert "If a person/pedestrian is visible" in prompt
    assert "insurance_claim_fields" in prompt
    assert "accident_datetime" in prompt
    assert "road_shape" in prompt
    assert "lane_count" in prompt
    assert "damage_parts" in prompt
    assert "accident_type_candidates" in prompt


def test_build_interleaved_content_places_storyboard_text_before_each_image():
    evidence_package = {
        "vlm_storyboard": [
            {
                "slot": 1,
                "path": "/tmp/a.jpg",
                "time": "00:01.000",
                "phase": "scene_context",
                "role": "도로 구조 확인",
                "why_selected": "regular context",
                "linked_actor_ids": ["T1"],
                "precomputed_hints": {"visible_actor_candidates": ["승용차"]},
            }
        ]
    }

    content = build_interleaved_content("prompt", ["/tmp/a.jpg"], evidence_package)

    assert content[0]["type"] == "text"
    assert "Frame 01" in content[0]["text"]
    assert "00:01.000" in content[0]["text"]
    assert "scene_context" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[-1] == {"type": "text", "text": "prompt"}


def test_build_qwen_interleaved_content_places_caption_before_each_image():
    content = build_qwen_interleaved_content(
        "prompt",
        ["image-a"],
        ["a.jpg"],
        [{"path": "a.jpg", "slot": 7, "time": "00:07.000", "phase": "insurance_context"}],
    )

    assert content[0]["type"] == "text"
    assert "Frame 07" in content[0]["text"]
    assert "insurance_context" in content[0]["text"]
    assert content[1] == {"type": "image", "image": "image-a"}
    assert content[-1] == {"type": "text", "text": "prompt"}


def test_render_qwen_chat_template_disables_thinking() -> None:
    class FakeProcessor:
        def __init__(self) -> None:
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return "rendered"

    processor = FakeProcessor()

    assert render_qwen_chat_template(processor, [{"role": "user", "content": []}]) == "rendered"
    assert processor.kwargs["enable_thinking"] is False
    assert processor.kwargs["add_generation_prompt"] is True
    assert processor.kwargs["tokenize"] is False


def test_compose_with_retry_retries_after_json_failure_when_fallback_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_DISABLE_FALLBACK_JSON", "1")
    backend = FlakyBackend()

    result = compose_with_retry(
        PipelineContext(video_path="sample.mp4", evidence_package={"frames": []}),
        backend,
        max_attempts=2,
    )

    assert backend.calls == 2
    assert result["objective_summary"] == "ok"


def test_compose_with_retry_reduces_images_after_cuda_oom(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_MAX_IMAGES", "12")
    monkeypatch.setenv("ACCIDENT_VLM_OOM_RETRY_MAX_IMAGES", "4")
    backend = OomThenRecordingBackend()
    context = PipelineContext(
        video_path="sample.mp4",
        evidence_package={
            "frames": [{"path": f"/tmp/frame-{index}.jpg"} for index in range(12)],
        },
    )

    result = compose_with_retry(context, backend, max_attempts=2)

    assert result["objective_summary"] == "ok"
    assert backend.image_counts == [12, 4]


def test_compose_with_retry_reduces_to_twelve_images_by_default_after_cuda_oom(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_MAX_IMAGES", "20")
    monkeypatch.delenv("ACCIDENT_VLM_OOM_RETRY_MAX_IMAGES", raising=False)
    backend = OomThenRecordingBackend()
    context = PipelineContext(
        video_path="sample.mp4",
        evidence_package={
            "frames": [{"path": f"/tmp/frame-{index}.jpg"} for index in range(20)],
        },
    )

    result = compose_with_retry(context, backend)

    assert result["objective_summary"] == "ok"
    assert backend.image_counts == [20, 12]


def test_compose_with_retry_uses_compact_text_only_prompt_after_repeated_cuda_oom(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_MAX_IMAGES", "12")
    monkeypatch.setenv("ACCIDENT_VLM_OOM_RETRY_MAX_IMAGES", "4")
    backend = OomUntilCompactBackend()
    context = PipelineContext(
        video_path="sample.mp4",
        evidence_package={
            "frames": [
                {"id": f"frame-{index}", "path": f"/tmp/frame-{index}.jpg", "verbose": "x" * 200}
                for index in range(100)
            ],
            "precomputed_facts": {
                "tracks": [
                    {
                        "track_id": "T1",
                        "type": "car",
                        "positions": [{"frame_id": f"frame-{index}", "bbox": [1, 2, 3, 4]} for index in range(80)],
                    }
                ],
                "event_candidates": [{"event_type": "접촉", "event_score": 80}],
            },
        },
    )

    result = compose_with_retry(context, backend, max_attempts=3)

    assert result["objective_summary"] == "compact ok"
    assert backend.image_counts == [12, 4, 0]
    assert "Compact evidence package" in backend.prompts[-1]
    assert "frame-99" not in backend.prompts[-1]
    assert "x" * 200 not in backend.prompts[-1]


def test_compose_with_retry_does_not_retry_non_retriable_weight_packed_error() -> None:
    backend = WeightPackedBackend()

    try:
        compose_with_retry(
            PipelineContext(video_path="sample.mp4", evidence_package={"frames": []}),
            backend,
            max_attempts=3,
        )
    except RuntimeError as exc:
        assert "ACCIDENT_VLM_BACKEND=openai" in str(exc)
    else:
        raise AssertionError("expected non-retryable AWQ backend error")

    assert backend.calls == 1


def test_non_retriable_weight_packed_error_has_actionable_message() -> None:
    message = _format_non_retriable_vlm_error(KeyError("weight_packed"))

    assert "AWQ/compressed-tensors" in message
    assert "vLLM/SGLang" in message
    assert "Transformers-compatible checkpoint" in message


def test_retriable_vlm_error_only_retries_oom_and_json_errors() -> None:
    assert _is_retriable_vlm_error(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB."))
    assert _is_retriable_vlm_error(ValueError("bad json"))
    assert _is_retriable_vlm_error(json.JSONDecodeError("bad", "{}", 0))
    assert not _is_retriable_vlm_error(KeyError("weight_packed"))


def test_collect_evidence_image_paths_caps_and_prioritizes_images(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_MAX_IMAGES", "3")
    evidence_package = {
        "frames": [{"path": "/tmp/frame.jpg", "purpose": "regular_context"}],
        "overlays": [{"path": "/tmp/lane.jpg", "purpose": "lane_segmentation_overlay"}],
        "crops": [{"path": "/tmp/sign.jpg", "purpose": "sign_crop"}],
        "evidence_images": [
            {"path": "/tmp/signal.jpg", "purpose": "traffic_light_crop"},
            {"path": "/tmp/event.jpg", "purpose": "event_segment"},
        ],
    }

    assert _collect_evidence_image_paths(evidence_package) == [
        "/tmp/signal.jpg",
        "/tmp/sign.jpg",
        "/tmp/event.jpg",
    ]


def test_collect_evidence_image_paths_uses_quality_cap_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ACCIDENT_VLM_MAX_IMAGES", raising=False)
    evidence_package = {
        "frames": [{"path": f"/tmp/frame-{index}.jpg"} for index in range(80)],
    }

    assert len(_collect_evidence_image_paths(evidence_package)) == 20


def test_collect_evidence_image_paths_can_disable_cap(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_MAX_IMAGES", "0")
    evidence_package = {
        "frames": [{"path": f"/tmp/frame-{index}.jpg"} for index in range(80)],
    }

    assert len(_collect_evidence_image_paths(evidence_package)) == 80


def test_qwen_generate_json_chunks_images_without_dropping_evidence(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_IMAGE_CHUNK_SIZE", "2")
    monkeypatch.setenv("ACCIDENT_VLM_CHUNK_MAX_NEW_TOKENS", "128")
    monkeypatch.setenv("ACCIDENT_VLM_FINAL_MAX_NEW_TOKENS", "512")
    backend = object.__new__(TransformersQwenBackend)
    backend._load_image = lambda path: f"image:{path}"
    calls = []

    image_records = [
        {"path": name, "slot": index + 1, "time": f"00:0{index}.000", "phase": "approach"}
        for index, name in enumerate(["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"])
    ]
    reversed_image_records = list(reversed(image_records))

    def fake_generate_text(
        prompt: str,
        images,
        image_paths=None,
        max_new_tokens=None,
        image_records=None,
        chunk_label=None,
    ):
        calls.append((prompt, list(images or []), max_new_tokens, image_records))
        if images:
            return f"chunk saw {len(images)} images"
        return '{"schema_version":"accident_video_facts.v1","objective_summary":"ok"}'

    backend._generate_text = fake_generate_text

    result = backend.generate_json(
        "full prompt with verbose evidence package",
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"],
        reversed_image_records,
    )

    assert result["schema_version"] == "accident_video_facts.v1"
    assert [images for _, images, _, _ in calls] == [
        ["image:a.jpg", "image:b.jpg"],
        ["image:c.jpg", "image:d.jpg"],
        ["image:e.jpg"],
        [],
    ]
    assert [tokens for _, _, tokens, _ in calls] == [128, 128, 128, 512]
    assert calls[0][3] == image_records[:2]
    assert "chunk saw 2 images" in calls[-1][0]
    assert "full prompt with verbose evidence package" not in calls[-1][0]
    assert "insurance_claim_fields" in calls[-1][0]
    assert "Use all chunk observations below as evidence" in calls[-1][0]


def test_image_data_url_encodes_local_image(tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"abc")

    assert _image_data_url(str(image_path)) == "data:image/jpeg;base64,YWJj"


def test_openai_compatible_backend_posts_chat_completion(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"abc")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"schema_version":"accident_video_facts.v1",'
                                    '"objective_summary":"ok"}'
                                )
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = dict(request.header_items())
        return FakeResponse()

    monkeypatch.setattr("accident_vlm.modules.vlm_composer.urllib.request.urlopen", fake_urlopen)

    backend = OpenAICompatibleVLMBackend(
        model_id="qwen-awq",
        base_url="http://localhost:30000/v1/",
        api_key="secret",
        timeout_sec=123,
    )

    result = backend.generate_json("prompt", [str(image_path)])

    assert result["objective_summary"] == "ok"
    assert captured["url"] == "http://localhost:30000/v1/chat/completions"
    assert captured["timeout"] == 123
    assert captured["payload"]["model"] == "qwen-awq"
    assert captured["payload"]["temperature"] == 0
    assert captured["payload"]["max_tokens"] == 1024
    assert captured["payload"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["payload"]["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["payload"]["top_p"] == 1
    assert captured["payload"]["stop"] == ["<think>", "</think>"]
    assert "Frame 01" in captured["payload"]["messages"][0]["content"][0]["text"]
    assert captured["payload"]["messages"][0]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,YWJj"
    assert captured["payload"]["messages"][0]["content"][-1] == {"type": "text", "text": "prompt"}
    headers = {key.lower(): value for key, value in captured["headers"].items()}
    assert headers["authorization"] == "Bearer secret"
    assert headers["x-accident-vlm-image-count"] == "1"
    assert headers["x-accident-vlm-max-tokens"] == "1024"
    assert headers["x-accident-vlm-chunk"] == "direct"


def test_openai_compatible_backend_includes_http_error_body(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"abc")

    def fake_urlopen(_request, timeout):
        assert timeout == 300.0
        raise urllib.error.HTTPError(
            url="http://localhost:30000/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=None,
        )

    monkeypatch.setattr("accident_vlm.modules.vlm_composer.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "accident_vlm.modules.vlm_composer._read_http_error_detail",
        lambda _error: '{"error":"model not found"}',
    )
    backend = OpenAICompatibleVLMBackend("wrong-model", "http://localhost:30000/v1")

    try:
        backend.generate_json("prompt", [str(image_path)])
    except RuntimeError as exc:
        assert "HTTP 400" in str(exc)
        assert "model not found" in str(exc)
    else:
        raise AssertionError("expected HTTP error detail")


def test_read_http_error_detail_reads_body() -> None:
    class FakeBody:
        def read(self) -> bytes:
            return b'{"error":"bad request detail"}'

        def close(self) -> None:
            return None

    error = urllib.error.HTTPError(
        url="http://localhost",
        code=400,
        msg="Bad Request",
        hdrs={},
        fp=FakeBody(),
    )

    assert _read_http_error_detail(error) == '{"error":"bad request detail"}'


def test_openai_compatible_backend_chunks_images_without_dropping_evidence(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_IMAGE_CHUNK_SIZE", "2")
    monkeypatch.setenv("ACCIDENT_VLM_CHUNK_MAX_NEW_TOKENS", "128")
    monkeypatch.setenv("ACCIDENT_VLM_FINAL_MAX_NEW_TOKENS", "512")
    backend = OpenAICompatibleVLMBackend("qwen-awq", "http://localhost:30000/v1")
    calls = []

    image_records = [
        {"path": name, "slot": index + 1, "time": f"00:0{index}.000", "phase": "approach"}
        for index, name in enumerate(["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"])
    ]
    reversed_image_records = list(reversed(image_records))

    def fake_generate_text(
        prompt: str,
        image_paths: list[str],
        max_tokens: int,
        image_records=None,
        chunk_label=None,
    ) -> str:
        calls.append((prompt, list(image_paths), max_tokens, image_records, chunk_label))
        if image_paths:
            return f"chunk saw {len(image_paths)} images"
        return '{"schema_version":"accident_video_facts.v1","objective_summary":"ok"}'

    backend._generate_text = fake_generate_text

    result = backend.generate_json(
        "full prompt with verbose evidence package",
        ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"],
        reversed_image_records,
    )

    assert result["schema_version"] == "accident_video_facts.v1"
    assert [images for _, images, _, _, _ in calls] == [
        ["a.jpg", "b.jpg"],
        ["c.jpg", "d.jpg"],
        ["e.jpg"],
        [],
    ]
    assert [tokens for _, _, tokens, _, _ in calls] == [128, 128, 128, 512]
    assert [label for _, _, _, _, label in calls] == ["chunk 1/3", "chunk 2/3", "chunk 3/3", "final text-only"]
    assert calls[0][3] == image_records[:2]
    assert "chunk saw 2 images" in calls[-1][0]
    assert "full prompt with verbose evidence package" not in calls[-1][0]
    assert "insurance_claim_fields" in calls[-1][0]
    assert "Use all chunk observations below as evidence" in calls[-1][0]


def test_openai_backend_defaults_to_compact_chunked_text_only_final(monkeypatch) -> None:
    monkeypatch.delenv("ACCIDENT_VLM_IMAGE_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("ACCIDENT_VLM_CHUNK_MAX_NEW_TOKENS", raising=False)
    monkeypatch.delenv("ACCIDENT_VLM_FINAL_MAX_NEW_TOKENS", raising=False)
    backend = OpenAICompatibleVLMBackend("qwen-awq", "http://localhost:30000/v1")
    recorder = RecordingChunkBackend()
    backend._generate_text = recorder._generate_text

    result = backend.generate_json(
        "full prompt " + ("x" * 1000),
        [f"{index}.jpg" for index in range(8)],
        [{"path": f"{index}.jpg"} for index in range(8)],
    )

    assert result["objective_summary"] == "ok"
    assert [len(call["image_paths"]) for call in recorder.calls] == [4, 4, 0]
    assert [call["max_tokens"] for call in recorder.calls] == [128, 128, 1024]
    assert recorder.calls[-1]["chunk_label"] == "final text-only"
    assert "full prompt" not in recorder.calls[-1]["prompt"]


def test_compose_final_facts_returns_conservative_fallback_after_invalid_json(monkeypatch) -> None:
    monkeypatch.delenv("ACCIDENT_VLM_DISABLE_FALLBACK_JSON", raising=False)
    backend = InvalidJsonBackend()

    output = compose_final_facts(
        PipelineContext(
            video_path="sample.mp4",
            evidence_package={
                "vlm_storyboard": [{"slot": 1, "id": "frame_000001", "time": "00:01.000"}],
                "precomputed_facts": {"metadata": {"duration_sec": 10.0}},
            },
        ),
        backend,
    )

    assert backend.calls == 1
    assert output.schema_version == "accident_video_facts.v1"
    assert output.objective_summary == "VLM JSON 출력이 불완전하여 보수적 fallback 결과를 반환함."
    assert any("VLM JSON parsing failed" in item for item in output.uncertainties)


def test_openai_chunked_backend_returns_chunk_fallback_when_final_json_is_invalid(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_IMAGE_CHUNK_SIZE", "2")
    backend = OpenAICompatibleVLMBackend("qwen-awq", "http://localhost:30000/v1")
    calls = []

    def fake_generate_text(prompt: str, image_paths: list[str], max_tokens: int, image_records=None, chunk_label=None):
        calls.append((prompt, list(image_paths), chunk_label))
        if image_paths:
            return f"{chunk_label}: vehicle visible"
        return '{"schema_version":"accident_video_facts.v1"'

    backend._generate_text = fake_generate_text

    result = backend.generate_json("prompt", ["a.jpg", "b.jpg", "c.jpg"], [{"path": "a.jpg"}])

    assert result["schema_version"] == "accident_video_facts.v1"
    assert result["evidence_index"]["fallback_reason"] == "chunked_final_json_parse_failure"
    assert len(result["evidence_index"]["chunk_observations"]) == 2
    assert "vehicle visible" in result["evidence_index"]["chunk_observations"][0]["observations"]
    assert [call[2] for call in calls] == ["chunk 1/2", "chunk 2/2", "final text-only"]


def test_get_qwen_backend_can_use_openai_compatible_backend(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_BACKEND", "openai")
    monkeypatch.setenv("ACCIDENT_VLM_OPENAI_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("ACCIDENT_VLM_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("ACCIDENT_VLM_OPENAI_TIMEOUT_SEC", "77")
    get_qwen_backend.cache_clear()

    backend = get_qwen_backend("served-model", "cuda:0")

    assert isinstance(backend, OpenAICompatibleVLMBackend)
    assert backend.model_id == "served-model"
    assert backend.base_url == "http://localhost:8001/v1"
    assert backend.api_key == "secret"
    assert backend.timeout_sec == 77


def test_get_qwen_backend_can_override_openai_served_model(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_BACKEND", "openai")
    monkeypatch.setenv("ACCIDENT_VLM_OPENAI_MODEL", "served-qwen-awq")
    get_qwen_backend.cache_clear()

    backend = get_qwen_backend("/local/path/Qwen3.6-27B-AWQ-INT4", "auto")

    assert isinstance(backend, OpenAICompatibleVLMBackend)
    assert backend.model_id == "served-qwen-awq"


def test_parse_max_memory_accepts_gpu_and_cpu_entries() -> None:
    assert _parse_max_memory("0:22GiB,1:22GiB,2:22GiB,3:22GiB,cpu:64GiB") == {
        0: "22GiB",
        1: "22GiB",
        2: "22GiB",
        3: "22GiB",
        "cpu": "64GiB",
    }


def test_auto_cuda_max_memory_uses_visible_free_memory(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def mem_get_info(index: int) -> tuple[int, int]:
            free_by_index = {
                0: 10 * 1024 * 1024 * 1024,
                1: 20 * 1024 * 1024 * 1024,
            }
            return free_by_index[index], 24 * 1024 * 1024 * 1024

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.delenv("ACCIDENT_VLM_AUTO_MAX_MEMORY", raising=False)
    monkeypatch.setenv("ACCIDENT_VLM_AUTO_MAX_MEMORY_FRACTION", "0.8")

    assert _auto_cuda_max_memory(FakeTorch) == {
        0: "8192MiB",
        1: "16384MiB",
    }


def test_auto_cuda_max_memory_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_AUTO_MAX_MEMORY", "0")

    assert _auto_cuda_max_memory(object()) == {}


def test_qwen_alias_maps_to_local_awq_int4_model() -> None:
    assert normalize_model_id("Qwen/Qwen3.6-27B") == "/home/minsung0830/accident-vlm/models/Qwen3.6-27B-AWQ-INT4"


def test_prepare_transformers_runtime_env_defaults_to_tmp_cache(monkeypatch) -> None:
    for key in ("TMPDIR", "HF_HOME", "TRANSFORMERS_CACHE", "HF_HUB_CACHE", "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)

    _prepare_transformers_runtime_env()

    assert __import__("os").environ["TMPDIR"] == "/tmp"
    assert __import__("os").environ["HF_HOME"] == "/tmp/accident-vlm-hf"
    assert __import__("os").environ["TRANSFORMERS_CACHE"] == "/tmp/accident-vlm-hf/transformers"
    assert __import__("os").environ["HF_HUB_CACHE"] == "/tmp/accident-vlm-hf/hub"
    assert __import__("os").environ["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/accident-vlm-torchinductor"


def test_attn_implementation_defaults_to_sdpa_without_flash_attn(monkeypatch) -> None:
    monkeypatch.delenv("ACCIDENT_VLM_ATTN_IMPLEMENTATION", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    assert _attn_implementation() == "sdpa"


def test_attn_implementation_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_ATTN_IMPLEMENTATION", "eager")

    assert _attn_implementation() == "eager"


def test_generation_cache_defaults_on_for_decode_speed(monkeypatch) -> None:
    monkeypatch.delenv("ACCIDENT_VLM_USE_CACHE", raising=False)

    assert _use_generation_cache() is True


def test_generation_cache_can_be_disabled_for_memory_pressure(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_USE_CACHE", "0")

    assert _use_generation_cache() is False


def test_model_dtype_defaults_to_bfloat16_for_quantized_models(monkeypatch) -> None:
    monkeypatch.delenv("ACCIDENT_VLM_MODEL_DTYPE", raising=False)

    assert _model_dtype() == "bfloat16"


def test_model_dtype_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("ACCIDENT_VLM_MODEL_DTYPE", "float16")

    assert _model_dtype() == "float16"


def test_configure_transformers_loading_disables_allocator_warmup(monkeypatch) -> None:
    class FakeModelingUtils:
        @staticmethod
        def caching_allocator_warmup() -> str:
            return "original"

    monkeypatch.setenv("ACCIDENT_VLM_DISABLE_ALLOCATOR_WARMUP", "1")

    _configure_transformers_loading(FakeModelingUtils)

    assert FakeModelingUtils.caching_allocator_warmup() is None


def test_configure_transformers_loading_can_keep_allocator_warmup(monkeypatch) -> None:
    class FakeModelingUtils:
        @staticmethod
        def caching_allocator_warmup() -> str:
            return "original"

    monkeypatch.setenv("ACCIDENT_VLM_DISABLE_ALLOCATOR_WARMUP", "0")

    _configure_transformers_loading(FakeModelingUtils)

    assert FakeModelingUtils.caching_allocator_warmup() == "original"


def test_compose_with_retry_clears_cuda_cache_after_failed_attempt(monkeypatch) -> None:
    calls = []
    backend = FlakyBackend()
    monkeypatch.setattr(
        "accident_vlm.modules.vlm_composer._clear_cuda_cache",
        lambda: calls.append("cleared"),
    )

    compose_with_retry(
        PipelineContext(video_path="sample.mp4", evidence_package={"frames": []}),
        backend,
        max_attempts=2,
    )

    assert calls == ["cleared"]


def test_normalize_model_id_maps_qwen_alias_to_local_model() -> None:
    assert normalize_model_id("Qwen/Qwen3.6-27B") == "/home/minsung0830/accident-vlm/models/Qwen3.6-27B-AWQ-INT4"


def test_get_qwen_backend_reuses_local_model_for_qwen_alias(monkeypatch) -> None:
    created = []

    class FakeQwenBackend:
        def __init__(self, model_id: str, device: str = "auto") -> None:
            created.append((model_id, device))

    monkeypatch.setattr(
        "accident_vlm.modules.vlm_composer.TransformersQwenBackend",
        FakeQwenBackend,
    )
    get_qwen_backend.cache_clear()

    first = get_qwen_backend("/home/minsung0830/accident-vlm/models/Qwen3.6-27B-AWQ-INT4", "auto")
    second = get_qwen_backend("Qwen/Qwen3.6-27B", "auto")

    assert first is second
    assert created == [("/home/minsung0830/accident-vlm/models/Qwen3.6-27B-AWQ-INT4", "auto")]


def test_normalize_device_pins_backend_to_auto_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ACCIDENT_VLM_ALLOW_DEVICE_OVERRIDE", raising=False)

    assert normalize_device("cuda:0") == "auto"
    assert normalize_device(" balanced_low_0 ") == "auto"
    assert normalize_device("") == "auto"


def test_get_qwen_backend_reuses_instance_across_device_aliases(monkeypatch) -> None:
    created = []

    class FakeQwenBackend:
        def __init__(self, model_id: str, device: str = "auto") -> None:
            created.append((model_id, device))

    monkeypatch.delenv("ACCIDENT_VLM_ALLOW_DEVICE_OVERRIDE", raising=False)
    monkeypatch.setattr(
        "accident_vlm.modules.vlm_composer.TransformersQwenBackend",
        FakeQwenBackend,
    )
    get_qwen_backend.cache_clear()

    first = get_qwen_backend("Qwen/Qwen3.6-27B", "cuda:0")
    second = get_qwen_backend("/home/minsung0830/accident-vlm/models/Qwen3.6-27B-AWQ-INT4", "balanced_low_0")

    assert first is second
    assert created == [("/home/minsung0830/accident-vlm/models/Qwen3.6-27B-AWQ-INT4", "auto")]


def test_transformers_backend_generation_uses_global_lock(monkeypatch) -> None:
    events = []

    class FakeLock:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, exc_type, exc, tb):
            events.append("exit")

    class FakeBackend:
        _generate_lock = FakeLock()

        def _generate_text(self):
            from accident_vlm.modules.vlm_composer import TransformersQwenBackend

            return TransformersQwenBackend._generate_text(
                self,
                prompt="prompt",
                images=[],
                image_paths=[],
                max_new_tokens=1,
            )

    class FakeInputIds:
        shape = (1, 3)

    class FakeInputs(dict):
        input_ids = FakeInputIds()

        def to(self, device):
            return self

    class FakeProcessor:
        def __call__(self, **kwargs):
            return FakeInputs()

        def batch_decode(self, values, skip_special_tokens=True):
            return ["{}"]

    class FakeGenerated:
        def __getitem__(self, item):
            return [[4]]

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            events.append("generate")
            return FakeGenerated()

    backend = FakeBackend()
    backend._processor = FakeProcessor()
    backend._model = FakeModel()
    monkeypatch.setattr("accident_vlm.modules.vlm_composer.render_qwen_chat_template", lambda processor, messages: "chat")

    backend._generate_text()

    assert events == ["enter", "generate", "exit"]


def test_transformers_backend_reuses_cached_resized_images(tmp_path, monkeypatch) -> None:
    from accident_vlm.modules.vlm_composer import TransformersQwenBackend

    opened = []

    class FakeImage:
        def convert(self, mode):
            return self

        def thumbnail(self, size):
            self.size = size

    class FakeImageModule:
        @staticmethod
        def open(path):
            opened.append(str(path))
            return FakeImage()

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    backend = object.__new__(TransformersQwenBackend)
    backend._image_cls = FakeImageModule
    backend._image_cache = {}
    monkeypatch.setenv("ACCIDENT_VLM_IMAGE_MAX_SIDE", "640")

    first = backend._load_image(str(image_path))
    second = backend._load_image(str(image_path))

    assert first is second
    assert opened == [str(image_path)]


def test_transformers_backend_image_cache_respects_max_side(tmp_path, monkeypatch) -> None:
    from accident_vlm.modules.vlm_composer import TransformersQwenBackend

    opened = []

    class FakeImage:
        def convert(self, mode):
            return self

        def thumbnail(self, size):
            self.size = size

    class FakeImageModule:
        @staticmethod
        def open(path):
            opened.append(str(path))
            return FakeImage()

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image")
    backend = object.__new__(TransformersQwenBackend)
    backend._image_cls = FakeImageModule
    backend._image_cache = {}

    monkeypatch.setenv("ACCIDENT_VLM_IMAGE_MAX_SIDE", "640")
    backend._load_image(str(image_path))
    monkeypatch.setenv("ACCIDENT_VLM_IMAGE_MAX_SIDE", "512")
    backend._load_image(str(image_path))

    assert opened == [str(image_path), str(image_path)]


def test_disable_transformers_allocator_warmup_patches_and_restores(monkeypatch) -> None:
    import types

    calls = []
    module = types.SimpleNamespace(caching_allocator_warmup=lambda *args, **kwargs: calls.append("original"))
    monkeypatch.setitem(__import__("sys").modules, "transformers.modeling_utils", module)

    with disable_transformers_allocator_warmup():
        module.caching_allocator_warmup()
        assert calls == []

    module.caching_allocator_warmup()
    assert calls == ["original"]
