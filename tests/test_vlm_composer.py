from accident_vlm.modules.vlm_composer import (
    SYSTEM_PROMPT,
    build_vlm_prompt,
    compose_with_backend,
    compose_with_retry,
    _clear_cuda_cache,
    get_qwen_backend,
    normalize_model_id,
    normalize_device,
    disable_transformers_allocator_warmup,
    parse_json_response,
    render_qwen_chat_template,
    _configure_transformers_loading,
    _collect_evidence_image_paths,
    _parse_max_memory,
)
from accident_vlm.schemas.preprocessing import PipelineContext


class RecordingBackend:
    def __init__(self) -> None:
        self.prompt = None
        self.image_paths = None

    def generate_json(self, prompt: str, image_paths: list[str]) -> dict:
        self.prompt = prompt
        self.image_paths = image_paths
        return {"schema_version": "accident_video_facts.v1"}


class FlakyBackend:
    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, prompt: str, image_paths: list[str]) -> dict:
        self.calls += 1
        if self.calls == 1:
            raise ValueError("bad json")
        return {"schema_version": "accident_video_facts.v1", "objective_summary": "ok"}


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


def test_compose_with_backend_sends_prompt_and_truthy_evidence_image_paths() -> None:
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


def test_compose_with_retry_retries_after_json_failure() -> None:
    backend = FlakyBackend()

    result = compose_with_retry(
        PipelineContext(video_path="sample.mp4", evidence_package={"frames": []}),
        backend,
        max_attempts=2,
    )

    assert backend.calls == 2
    assert result["objective_summary"] == "ok"


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


def test_collect_evidence_image_paths_does_not_cap_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ACCIDENT_VLM_MAX_IMAGES", raising=False)
    evidence_package = {
        "frames": [{"path": f"/tmp/frame-{index}.jpg"} for index in range(20)],
    }

    assert len(_collect_evidence_image_paths(evidence_package)) == 20


def test_parse_max_memory_accepts_gpu_and_cpu_entries() -> None:
    assert _parse_max_memory("0:22GiB,1:22GiB,2:22GiB,3:22GiB,cpu:64GiB") == {
        0: "22GiB",
        1: "22GiB",
        2: "22GiB",
        3: "22GiB",
        "cpu": "64GiB",
    }


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
    assert normalize_model_id("Qwen/Qwen3.6-27B") == "/home/minsung0830/accident-vlm/models/Qwen3.6-27B"


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

    first = get_qwen_backend("/home/minsung0830/accident-vlm/models/Qwen3.6-27B", "auto")
    second = get_qwen_backend("Qwen/Qwen3.6-27B", "auto")

    assert first is second
    assert created == [("/home/minsung0830/accident-vlm/models/Qwen3.6-27B", "auto")]


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
    second = get_qwen_backend("/home/minsung0830/accident-vlm/models/Qwen3.6-27B", "balanced_low_0")

    assert first is second
    assert created == [("/home/minsung0830/accident-vlm/models/Qwen3.6-27B", "auto")]


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

