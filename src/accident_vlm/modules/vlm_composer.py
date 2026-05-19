import json
from pathlib import Path
from typing import Any, Protocol

from accident_vlm.config import PipelineConfig
from accident_vlm.modules.schema_guard import validate_final_output
from accident_vlm.schemas.final_output import AccidentFactOutput
from accident_vlm.schemas.preprocessing import PipelineContext


SYSTEM_PROMPT = """
Compose objective accident facts from the supplied evidence only.
Do not determine fault ratio, legal violation, negligence, offender, or victim.
Mark unsupported fields as 확인불가.
Every important event must include confidence and evidence.
""".strip()


def build_vlm_prompt(context: PipelineContext) -> str:
    evidence_package_json = json.dumps(
        context.evidence_package,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Return only JSON matching schema_version accident_video_facts.v1.\n\n"
        "Evidence package:\n"
        f"{evidence_package_json}"
    )


class VLMBackend(Protocol):
    def generate_json(self, prompt: str, image_paths: list[str]) -> dict[str, Any]:
        ...


def compose_with_backend(context: PipelineContext, backend: VLMBackend) -> dict[str, Any]:
    prompt = build_vlm_prompt(context)
    image_paths = [
        frame["path"]
        for frame in context.evidence_package.get("frames", [])
        if isinstance(frame, dict) and frame.get("path")
    ]
    return backend.generate_json(prompt=prompt, image_paths=image_paths)


class TransformersQwenBackend:
    def __init__(self, model_id: str, device: str = "auto") -> None:
        try:
            from PIL import Image  # type: ignore
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError("install accident-vlm[vlm] on the server to use Qwen backend") from exc

        self._image_cls = Image
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map=device,
            trust_remote_code=True,
        )

    def generate_json(self, prompt: str, image_paths: list[str]) -> dict[str, Any]:
        images = [self._image_cls.open(path).convert("RGB") for path in image_paths]
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(text=[text], images=images or None, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        generated_ids = self._model.generate(**inputs, max_new_tokens=4096)
        generated_text = self._processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]
        return _parse_json_response(generated_text)


def _parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("VLM response did not contain a JSON object")
    return json.loads(stripped[start : end + 1])


def create_qwen_backend(config: PipelineConfig) -> VLMBackend:
    return TransformersQwenBackend(config.qwen_model_id, device=config.device)


def compose_final_facts(
    context: PipelineContext,
    backend: VLMBackend,
) -> AccidentFactOutput:
    payload = compose_with_backend(context, backend)
    return validate_final_output(payload)


def write_final_facts(output: AccidentFactOutput, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        output.model_dump_json(indent=2),
        encoding="utf-8",
    )
