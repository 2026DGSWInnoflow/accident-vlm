import ast
import json
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from accident_vlm.config import PipelineConfig
from accident_vlm.modules.fact_verifier import verify_vlm_payload_against_context
from accident_vlm.modules.schema_guard import validate_final_output
from accident_vlm.schemas.final_output import AccidentFactOutput
from accident_vlm.schemas.preprocessing import PipelineContext


SYSTEM_PROMPT = """
Compose objective accident facts from the supplied evidence only.
Do not determine fault ratio, legal violation, negligence, offender, or victim.
Mark unsupported fields as 확인불가.
Every important event must include confidence and evidence.
Return a single valid JSON object. Do not include markdown, prose, or reasoning.
""".strip()


LOCAL_QWEN_MODEL_ID = "/home/minsung0830/accident-vlm/models/Qwen3.6-27B"
QWEN_MODEL_ALIASES = {
    "Qwen/Qwen3.6-27B": LOCAL_QWEN_MODEL_ID,
}
QWEN_BACKEND_LOCK = Lock()
DEFAULT_VLM_MAX_IMAGES = 12
DEFAULT_VLM_IMAGE_MAX_SIDE = 896
DEFAULT_VLM_MAX_NEW_TOKENS = 2048


OUTPUT_TEMPLATE = {
    "schema_version": "accident_video_facts.v1",
    "input_quality": {},
    "scene_type": {
        "value": "확인불가",
        "status": "unknown",
        "confidence": "unknown",
        "source": [],
        "evidence": [],
        "note": "영상에서 확인되지 않음",
    },
    "road_conditions": {},
    "traffic_control": {},
    "actors": [],
    "timeline": [],
    "collision": {},
    "speed_and_distance": {},
    "uncertainties": [],
    "evidence_index": {},
    "rag_hints": {
        "accident_type": "확인불가",
        "scenario_keywords": [],
    },
    "objective_summary": "확인 가능한 객관 사실이 제한적임.",
}


def build_vlm_prompt(context: PipelineContext) -> str:
    evidence_package_json = json.dumps(
        context.evidence_package,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Return only JSON matching this exact shape. Keep enum-like values in Korean "
        "when they are visible; use 확인불가/unknown when unsupported.\n"
        f"{json.dumps(OUTPUT_TEMPLATE, ensure_ascii=False, indent=2)}\n\n"
        "Evidence package:\n"
        f"{evidence_package_json}"
    )


class VLMBackend(Protocol):
    def generate_json(self, prompt: str, image_paths: list[str]) -> dict[str, Any]:
        ...


def compose_with_backend(context: PipelineContext, backend: VLMBackend) -> dict[str, Any]:
    prompt = build_vlm_prompt(context)
    image_paths = _collect_evidence_image_paths(context.evidence_package)
    return backend.generate_json(prompt=prompt, image_paths=image_paths)


def compose_with_retry(
    context: PipelineContext,
    backend: VLMBackend,
    max_attempts: int = 2,
) -> dict[str, Any]:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return compose_with_backend(context, backend)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            _clear_cuda_cache()
            if attempt == max_attempts - 1:
                break
    raise ValueError(f"VLM JSON composition failed after {max_attempts} attempts: {last_error}")


def _clear_cuda_cache() -> None:
    try:
        import torch  # type: ignore
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _collect_evidence_image_paths(evidence_package: dict[str, Any]) -> list[str]:
    max_images = int(os.getenv("ACCIDENT_VLM_MAX_IMAGES", str(DEFAULT_VLM_MAX_IMAGES)))

    weighted_paths: list[tuple[int, str]] = []
    for section in ("evidence_images", "crops", "overlays", "frames"):
        for item in evidence_package.get(section, []):
            if not isinstance(item, dict) or not item.get("path"):
                continue
            purpose = str(item.get("purpose", ""))
            priority = _image_priority(section, purpose)
            if item.get("importance_score") is not None:
                priority -= int(item["importance_score"])
            weighted_paths.append((priority, item["path"]))

    image_paths: list[str] = []
    for _, path in sorted(weighted_paths, key=lambda entry: entry[0]):
        if path not in image_paths:
            image_paths.append(path)
        if max_images > 0 and len(image_paths) >= max_images:
            break
    return image_paths


def _image_priority(section: str, purpose: str) -> int:
    purpose_weights = {
        "traffic_light_crop": 0,
        "sign_crop": 1,
        "event_segment": 2,
        "actor_crop": 3,
        "track_overlay": 4,
        "tracking_overlay": 4,
        "bev_overlay": 5,
        "lane_segmentation_overlay": 6,
        "motion_keyframe": 7,
        "regular_context": 8,
    }
    section_weights = {
        "crops": 20,
        "overlays": 40,
        "evidence_images": 60,
        "frames": 80,
    }
    return purpose_weights.get(purpose, section_weights.get(section, 100))


def normalize_model_id(model_id: str) -> str:
    stripped = model_id.strip()
    if stripped in QWEN_MODEL_ALIASES:
        return QWEN_MODEL_ALIASES[stripped]
    path = Path(stripped).expanduser()
    if path.exists():
        return str(path.resolve())
    return stripped


def normalize_device(device: str) -> str:
    allow_override = os.getenv("ACCIDENT_VLM_ALLOW_DEVICE_OVERRIDE", "0") in {"1", "true", "True", "yes"}
    normalized = (device or "auto").strip() or "auto"
    if allow_override:
        return normalized
    return "auto"


def render_qwen_chat_template(processor: Any, messages: list[dict[str, Any]]) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@contextmanager
def disable_transformers_allocator_warmup():
    if os.getenv("ACCIDENT_VLM_DISABLE_ALLOCATOR_WARMUP", "1") not in {"1", "true", "True", "yes"}:
        yield
        return
    try:
        import transformers.modeling_utils as modeling_utils  # type: ignore
    except ImportError:
        yield
        return

    original = modeling_utils.caching_allocator_warmup
    modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    try:
        yield
    finally:
        modeling_utils.caching_allocator_warmup = original


class TransformersQwenBackend:
    def __init__(self, model_id: str, device: str = "auto") -> None:
        try:
            from PIL import Image  # type: ignore
            import transformers.modeling_utils as modeling_utils  # type: ignore
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError("install accident-vlm[vlm] on the server to use Qwen backend") from exc

        self._image_cls = Image
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _configure_transformers_loading(modeling_utils)
        model_kwargs = {
            "device_map": device,
            "dtype": "auto",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        max_memory = _parse_max_memory(os.getenv("ACCIDENT_VLM_MAX_MEMORY"))
        if max_memory:
            model_kwargs["max_memory"] = max_memory
        with disable_transformers_allocator_warmup():
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                **model_kwargs,
            )
        self._model.eval()

    def generate_json(self, prompt: str, image_paths: list[str]) -> dict[str, Any]:
        images = [self._load_image(path) for path in image_paths]
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = render_qwen_chat_template(self._processor, messages)
        inputs = self._processor(text=[text], images=images or None, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        max_new_tokens = int(os.getenv("ACCIDENT_VLM_MAX_NEW_TOKENS", str(DEFAULT_VLM_MAX_NEW_TOKENS)))
        try:
            import torch  # type: ignore

            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
        except ImportError:
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated_text = self._processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]
        return parse_json_response(generated_text)

    def _load_image(self, path: str):
        image = self._image_cls.open(path).convert("RGB")
        max_side = int(os.getenv("ACCIDENT_VLM_IMAGE_MAX_SIDE", str(DEFAULT_VLM_IMAGE_MAX_SIDE)))
        if max_side > 0:
            image.thumbnail((max_side, max_side))
        return image


def _parse_max_memory(raw_value: str | None) -> dict[Any, str]:
    if not raw_value:
        return {}
    max_memory: dict[Any, str] = {}
    for item in raw_value.split(","):
        if not item.strip() or ":" not in item:
            continue
        key, value = item.split(":", 1)
        normalized_key: Any = key.strip()
        if normalized_key.isdigit():
            normalized_key = int(normalized_key)
        max_memory[normalized_key] = value.strip()
    return max_memory


def _configure_transformers_loading(modeling_utils: Any) -> None:
    disable_warmup = os.getenv("ACCIDENT_VLM_DISABLE_ALLOCATOR_WARMUP", "1").lower()
    if disable_warmup not in {"1", "true", "yes", "on"}:
        return

    def _noop_allocator_warmup(*_args: Any, **_kwargs: Any) -> None:
        return None

    modeling_utils.caching_allocator_warmup = _noop_allocator_warmup


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("VLM response did not contain a JSON object")
    candidate = stripped[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(candidate)
        if not isinstance(parsed, dict):
            raise ValueError("VLM response JSON root was not an object")
        return parsed


_parse_json_response = parse_json_response


@lru_cache(maxsize=1)
def _get_qwen_backend(normalized_model_id: str, normalized_device: str = "auto") -> VLMBackend:
    return TransformersQwenBackend(normalized_model_id, device=normalized_device)


def get_qwen_backend(model_id: str, device: str = "auto") -> VLMBackend:
    with QWEN_BACKEND_LOCK:
        return _get_qwen_backend(normalize_model_id(model_id), normalize_device(device))


get_qwen_backend.cache_clear = _get_qwen_backend.cache_clear  # type: ignore[attr-defined]


def create_qwen_backend(config: PipelineConfig) -> VLMBackend:
    return get_qwen_backend(config.qwen_model_id, config.device)


def compose_final_facts(
    context: PipelineContext,
    backend: VLMBackend,
) -> AccidentFactOutput:
    payload = compose_with_retry(context, backend)
    payload = verify_vlm_payload_against_context(payload, context)
    return validate_final_output(payload)


def write_final_facts(output: AccidentFactOutput, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        output.model_dump_json(indent=2),
        encoding="utf-8",
    )
