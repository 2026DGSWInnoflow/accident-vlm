from accident_vlm.modules.vlm_composer import (
    SYSTEM_PROMPT,
    build_vlm_prompt,
    compose_with_backend,
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


def test_compose_with_backend_sends_prompt_and_truthy_frame_paths() -> None:
    context = PipelineContext(
        video_path="sample.mp4",
        evidence_package={
            "frames": [
                {"id": "f001", "path": "/tmp/frame-1.jpg"},
                {"id": "f002", "path": None},
                {"id": "f003", "path": ""},
                {"id": "f004", "path": "/tmp/frame-4.jpg"},
            ],
            "precomputed_facts": {},
        },
    )
    backend = RecordingBackend()

    result = compose_with_backend(context, backend)

    assert result == {"schema_version": "accident_video_facts.v1"}
    assert backend.prompt == build_vlm_prompt(context)
    assert backend.image_paths == ["/tmp/frame-1.jpg", "/tmp/frame-4.jpg"]
