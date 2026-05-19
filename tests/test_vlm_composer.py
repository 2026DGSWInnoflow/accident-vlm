from accident_vlm.modules.vlm_composer import (
    SYSTEM_PROMPT,
    build_vlm_prompt,
    compose_with_backend,
    compose_with_retry,
    get_qwen_backend,
    parse_json_response,
    render_qwen_chat_template,
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
            "precomputed_facts": {},
        },
    )
    backend = RecordingBackend()

    result = compose_with_backend(context, backend)

    assert result == {"schema_version": "accident_video_facts.v1"}
    assert backend.prompt == build_vlm_prompt(context)
    assert backend.image_paths == [
        "/tmp/frame-1.jpg",
        "/tmp/frame-4.jpg",
        "/tmp/overlay-1.jpg",
        "/tmp/crop-1.jpg",
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
