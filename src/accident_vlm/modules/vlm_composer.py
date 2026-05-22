import ast
import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from accident_vlm.config import PipelineConfig
from accident_vlm.modules.fact_verifier import verify_vlm_payload_against_context
from accident_vlm.modules.schema_guard import (
    ACCIDENT_TYPE_CANDIDATE_DEFAULTS,
    INSURANCE_CLAIM_FIELD_DEFAULTS,
    validate_final_output,
)
from accident_vlm.schemas.final_output import AccidentFactOutput
from accident_vlm.schemas.preprocessing import PipelineContext


SYSTEM_PROMPT = """
Compose objective accident facts from the supplied evidence only.
Do not determine fault ratio, legal violation, negligence, offender, or victim.
Mark unsupported fields as 확인불가.
Every important event must include confidence and evidence.
Return a single valid JSON object. Do not include markdown, prose, or reasoning.
""".strip()

STORYBOARD_PROMPT = """
You are analyzing a short accident-video storyboard, not unrelated still images.
Read storyboard images in slot order and compare changes over time.

Required analysis discipline:
1. Build a frame-by-frame observation list before writing final facts.
2. Track whether the same actor appears across frames; do not assume identity if uncertain.
3. Check all accident hypotheses before setting accident_type:
   - 차대보행자
   - 차대차
   - 차대이륜차
   - 차대자전거
   - 차대시설물
   - 단독사고
   - 비접촉사고
   - 확인불가
4. If a person/pedestrian is visible in any storyboard image, crop, overlay, or precomputed hint,
   do not omit that actor. If uncertain, include it as a candidate with low confidence.
5. If a vehicle, motorcycle, bicycle, kickboard, pedestrian, sign, signal, lane, or impact clue is
   visible only in a crop or overlay, cite that crop/overlay id and keep confidence conservative.
6. Use precomputed hints only as candidates. Confirmed facts must cite visible storyboard evidence.
7. Put unsupported but relevant possibilities in uncertainties or rag_hints.scenario_keywords.
8. Every actor, timeline event, collision claim, traffic control claim, and accident_type candidate
   should cite exact storyboard slot/frame_id/crop_id evidence.
9. Always fill insurance_claim_fields and accident_type_candidates. Use 확인불가/unknown when not visible.
   Insurance claim fields must focus on objective claim-writing facts: accident_datetime,
   location, road_shape, lane_count, ego_direction, other_direction, damage_parts,
   and police_report_visible. Do not invent location or time if OCR/GPS/metadata is absent.
""".strip()


LOCAL_QWEN_MODEL_ID = "/home/minsung0830/accident-vlm/models/Qwen3.6-27B"
QWEN_MODEL_ALIASES = {
    "Qwen/Qwen3.6-27B": LOCAL_QWEN_MODEL_ID,
}
QWEN_BACKEND_LOCK = Lock()


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
    "insurance_claim_fields": INSURANCE_CLAIM_FIELD_DEFAULTS,
    "accident_type_candidates": ACCIDENT_TYPE_CANDIDATE_DEFAULTS,
    "uncertainties": [],
    "evidence_index": {},
    "rag_hints": {
        "accident_type": "확인불가",
        "scenario_keywords": [],
    },
    "objective_summary": "확인 가능한 객관 사실이 제한적임.",
}


def build_vlm_prompt(context: PipelineContext, compact: bool = False) -> str:
    evidence_package = _compact_evidence_package(context.evidence_package) if compact else context.evidence_package
    storyboard = evidence_package.get("vlm_storyboard", [])
    evidence_package_json = json.dumps(
        evidence_package,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    evidence_heading = "Compact evidence package" if compact else "Evidence package"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{STORYBOARD_PROMPT}\n\n"
        "Return only JSON matching this exact shape. Keep enum-like values in Korean "
        "when they are visible; use 확인불가/unknown when unsupported.\n"
        "In evidence_index, include storyboard_observations and candidate_checks when useful.\n"
        f"{json.dumps(OUTPUT_TEMPLATE, ensure_ascii=False, indent=2)}\n\n"
        "VLM evidence storyboard:\n"
        f"{json.dumps(_compact_storyboard(storyboard), ensure_ascii=False, indent=2)}\n\n"
        f"{evidence_heading}:\n"
        f"{evidence_package_json}"
    )


class VLMBackend(Protocol):
    def generate_json(
        self,
        prompt: str,
        image_paths: list[str],
        image_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        ...


def compose_with_backend(
    context: PipelineContext,
    backend: VLMBackend,
    image_limit: int | None = None,
    compact_prompt: bool = False,
) -> dict[str, Any]:
    if _force_compact_prompt():
        compact_prompt = True
    prompt = build_vlm_prompt(context, compact=compact_prompt)
    image_records = _collect_evidence_image_records(context.evidence_package, max_images=image_limit)
    image_paths = [str(record["path"]) for record in image_records if record.get("path")]
    return backend.generate_json(prompt=prompt, image_paths=image_paths, image_records=image_records)


def compose_with_retry(
    context: PipelineContext,
    backend: VLMBackend,
    max_attempts: int = 3,
) -> dict[str, Any]:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    last_error: Exception | None = None
    image_limit: int | None = None
    compact_prompt = False
    for attempt in range(max_attempts):
        try:
            return compose_with_backend(context, backend, image_limit=image_limit, compact_prompt=compact_prompt)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            _clear_cuda_cache()
            if not _is_retriable_vlm_error(exc):
                raise RuntimeError(_format_non_retriable_vlm_error(exc)) from exc
            if _is_cuda_oom(exc):
                image_limit, compact_prompt = _next_oom_strategy(image_limit, compact_prompt)
            if attempt == max_attempts - 1:
                break
    if _allow_fallback_json() and last_error is not None and _is_json_error(last_error):
        return _fallback_final_payload(context, last_error)
    raise ValueError(f"VLM JSON composition failed after {max_attempts} attempts: {last_error}")


def _is_cuda_oom(error: Exception) -> bool:
    message = str(error).lower()
    return "cuda out of memory" in message or ("out of memory" in message and "cuda" in message)


def _is_retriable_vlm_error(error: Exception) -> bool:
    if _is_cuda_oom(error):
        return True
    if isinstance(error, (json.JSONDecodeError, SyntaxError)):
        return True
    if isinstance(error, ValueError):
        message = str(error).lower()
        return "vlm response" in message or "json" in message
    return False


def _is_json_error(error: Exception) -> bool:
    if isinstance(error, (json.JSONDecodeError, SyntaxError)):
        return True
    if isinstance(error, ValueError):
        message = str(error).lower()
        return "json" in message or "never closed" in message or "vlm response" in message
    return False


def _allow_fallback_json() -> bool:
    return os.getenv("ACCIDENT_VLM_DISABLE_FALLBACK_JSON", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _fallback_final_payload(context: PipelineContext, error: Exception) -> dict[str, Any]:
    payload = deepcopy(OUTPUT_TEMPLATE)
    payload["input_quality"] = (
        context.input_quality.model_dump() if context.input_quality else {}
    )
    payload["evidence_index"] = {
        "storyboard_observations": _compact_storyboard(
            context.evidence_package.get("vlm_storyboard", [])
        ),
        "fallback_reason": "vlm_json_parse_failure",
    }
    payload["objective_summary"] = "VLM JSON 출력이 불완전하여 보수적 fallback 결과를 반환함."
    payload["uncertainties"] = [
        *payload.get("uncertainties", []),
        f"VLM JSON parsing failed after retries: {error}",
    ]
    return payload


def _format_non_retriable_vlm_error(error: Exception) -> str:
    message = str(error)
    if "weight_packed" in message:
        return (
            "AWQ/compressed-tensors checkpoint failed during Transformers generation "
            "with weight_packed tensors. This is not a retryable VLM output error. "
            "Serve this AWQ model through vLLM/SGLang and set ACCIDENT_VLM_BACKEND=openai, "
            "or use a Transformers-compatible checkpoint such as an FP8 model or a BF16-MTP "
            f"AWQ variant. Original error: {message}"
        )
    return f"Non-retryable VLM backend error: {message}"


def _next_oom_strategy(current_limit: int | None, compact_prompt: bool) -> tuple[int, bool]:
    retry_limit = int(os.getenv("ACCIDENT_VLM_OOM_RETRY_MAX_IMAGES", "12"))
    if current_limit is None:
        if retry_limit <= 0:
            return 0, True
        return retry_limit, compact_prompt
    if current_limit > 0:
        return 0, True
    return current_limit, True


def _compact_evidence_package(evidence_package: dict[str, Any]) -> dict[str, Any]:
    facts = evidence_package.get("precomputed_facts", {})
    compact_facts = {
        "metadata": facts.get("metadata", {}),
        "input_quality": facts.get("input_quality", {}),
        "preprocessing_uncertainties": _limit_list(facts.get("preprocessing_uncertainties", []), 20),
        "ocr_summary": facts.get("ocr_summary", {}),
        "scene_type_candidates": _limit_list(facts.get("scene_type_candidates", []), 5),
        "tracks": [_compact_track(track) for track in _limit_list(facts.get("tracks", []), 8)],
        "tracker_comparison": facts.get("tracker_comparison", {}),
        "road_geometry": facts.get("road_geometry", {}),
        "speed_estimates": facts.get("speed_estimates", {}),
        "traffic_control": facts.get("traffic_control", {}),
        "event_candidates": _limit_list(facts.get("event_candidates", []), 12),
        "evidence_summary": facts.get("evidence_summary", {}),
    }
    return {
        "vlm_storyboard": _compact_storyboard(evidence_package.get("vlm_storyboard", [])),
        "frames": _compact_image_records(evidence_package.get("frames", []), 20),
        "selected_segments": _limit_list(evidence_package.get("selected_segments", []), 8),
        "overlays": _compact_image_records(evidence_package.get("overlays", []), 8),
        "crops": _compact_image_records(evidence_package.get("crops", []), 8),
        "evidence_images": _compact_image_records(evidence_package.get("evidence_images", []), 20),
        "precomputed_facts": compact_facts,
    }


def _compact_storyboard(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in _limit_list(value, 24):
        if not isinstance(item, dict):
            continue
        record = {
            key: item.get(key)
            for key in (
                "slot",
                "id",
                "frame_id",
                "time",
                "phase",
                "role",
                "purpose",
                "source",
                "track_id",
                "linked_actor_ids",
                "why_selected",
                "precomputed_hints",
            )
            if item.get(key) is not None
        }
        records.append(record)
    return records


def _limit_list(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:limit]


def _compact_image_records(value: Any, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in _limit_list(value, limit):
        if not isinstance(item, dict):
            continue
        records.append(
            {
                key: item.get(key)
                for key in (
                    "id",
                    "time",
                    "frame_index",
                    "purpose",
                    "source",
                    "frame_id",
                    "track_id",
                    "type",
                    "confidence",
                    "importance_score",
                    "rank_reason",
                )
                if item.get(key) is not None
            }
        )
    return records


def _compact_track(track: Any) -> dict[str, Any]:
    if not isinstance(track, dict):
        return {}
    compact = {
        key: track.get(key)
        for key in (
            "track_id",
            "type",
            "confidence",
            "movement_candidate",
            "relative_position_start",
            "relative_position_end",
            "tracking_method",
            "source_stage",
        )
        if track.get(key) is not None
    }
    positions = track.get("positions", [])
    if isinstance(positions, list) and positions:
        compact["positions"] = _sample_positions(positions)
    return compact


def _sample_positions(positions: list[Any]) -> list[dict[str, Any]]:
    indexes = sorted({0, len(positions) // 2, len(positions) - 1})
    sampled: list[dict[str, Any]] = []
    for index in indexes:
        item = positions[index]
        if not isinstance(item, dict):
            continue
        sampled.append(
            {
                key: item.get(key)
                for key in ("frame_id", "time", "center", "bbox", "relative_position")
                if item.get(key) is not None
            }
        )
    return sampled


def _chunk_max_new_tokens() -> int:
    return int(os.getenv("ACCIDENT_VLM_CHUNK_MAX_NEW_TOKENS", "128"))


def _final_max_new_tokens() -> int:
    return int(os.getenv("ACCIDENT_VLM_FINAL_MAX_NEW_TOKENS", "1024"))


def _image_chunk_size() -> int:
    return int(os.getenv("ACCIDENT_VLM_IMAGE_CHUNK_SIZE", "4"))


def _force_text_only_final() -> bool:
    return os.getenv("ACCIDENT_VLM_FORCE_TEXT_ONLY_FINAL", "1").lower() in {"1", "true", "yes", "on"}


def _force_compact_prompt() -> bool:
    return os.getenv("ACCIDENT_VLM_FORCE_COMPACT_PROMPT", "1").lower() in {"1", "true", "yes", "on"}


def _clear_cuda_cache() -> None:
    try:
        import torch  # type: ignore
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _collect_evidence_image_paths(evidence_package: dict[str, Any], max_images: int | None = None) -> list[str]:
    return [
        str(record["path"])
        for record in _collect_evidence_image_records(evidence_package, max_images=max_images)
        if record.get("path")
    ]


def _collect_evidence_image_records(
    evidence_package: dict[str, Any],
    max_images: int | None = None,
) -> list[dict[str, Any]]:
    explicit_limit = max_images is not None
    if max_images is None:
        max_images = int(os.getenv("ACCIDENT_VLM_MAX_IMAGES", "20"))
    elif max_images <= 0:
        return []

    storyboard = evidence_package.get("vlm_storyboard", [])
    if isinstance(storyboard, list) and storyboard:
        selected: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for item in sorted(
            [item for item in storyboard if isinstance(item, dict) and item.get("path")],
            key=lambda record: int(record.get("slot", 9999) or 9999),
        ):
            path = str(item["path"])
            if path in seen_paths:
                continue
            seen_paths.add(path)
            selected.append(item)
            if max_images > 0 and len(selected) >= max_images:
                break
        return selected

    weighted_records: list[tuple[int, dict[str, Any]]] = []
    for section in ("evidence_images", "crops", "overlays", "frames"):
        for item in evidence_package.get(section, []):
            if not isinstance(item, dict) or not item.get("path"):
                continue
            purpose = str(item.get("purpose", ""))
            priority = _image_priority(section, purpose)
            if item.get("importance_score") is not None:
                priority -= int(item["importance_score"])
            weighted_records.append((priority, item))

    image_records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for _, record in sorted(weighted_records, key=lambda entry: entry[0]):
        path = str(record["path"])
        if path not in seen_paths:
            seen_paths.add(path)
            image_records.append(record)
        if max_images > 0 and len(image_records) >= max_images:
            break
        if explicit_limit and len(image_records) >= max_images:
            break
    return image_records


def _image_priority(section: str, purpose: str) -> int:
    purpose_weights = {
        "traffic_light_crop": 0,
        "sign_crop": 1,
        "event_segment": 2,
        "impact_candidate": 2,
        "event_scan_impact_candidate": 2,
        "pre_impact": 3,
        "event_scan_pre_impact": 3,
        "actor_crop": 3,
        "track_overlay": 4,
        "tracking_overlay": 4,
        "post_impact": 5,
        "event_scan_post_impact": 5,
        "bev_overlay": 5,
        "lane_segmentation_overlay": 6,
        "motion_keyframe": 7,
        "event_window_context": 7,
        "pre_context": 8,
        "event_scan_pre_context": 8,
        "regular_context": 8,
        "frame_selection_contact_sheet": 9,
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
            "dtype": _model_dtype(),
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

    def generate_json(
        self,
        prompt: str,
        image_paths: list[str],
        image_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        images = [self._load_image(path) for path in image_paths]
        chunk_size = _image_chunk_size()
        if chunk_size > 0 and len(images) > chunk_size:
            return self._generate_chunked_json(prompt, images, image_paths, image_records or [], chunk_size)
        return parse_json_response(
            self._generate_text(
                prompt,
                images,
                image_paths=image_paths,
                max_new_tokens=_final_max_new_tokens(),
                image_records=image_records,
            )
        )

    def _generate_chunked_json(
        self,
        prompt: str,
        images: list[Any],
        image_paths: list[str],
        image_records: list[dict[str, Any]],
        chunk_size: int,
    ) -> dict[str, Any]:
        chunk_summaries: list[dict[str, Any]] = []
        total_chunks = (len(images) + chunk_size - 1) // chunk_size
        for chunk_index, start in enumerate(range(0, len(images), chunk_size), start=1):
            chunk_images = images[start : start + chunk_size]
            chunk_paths = image_paths[start : start + chunk_size]
            chunk_records = _records_for_paths(chunk_paths, image_records)
            chunk_prompt = (
                "Analyze only this evidence image chunk. Do not decide fault or legal liability. "
                "Return concise Korean observations grounded only in visible evidence.\n"
                f"Chunk {chunk_index}/{total_chunks}; image indexes {start + 1}-{start + len(chunk_images)}.\n\n"
                "Focus on visible actors, traffic lights/signs, road type, lane markings, "
                "weather/visibility, impact candidates, and uncertainty."
            )
            chunk_summaries.append(
                {
                    "chunk_index": chunk_index,
                    "image_indexes": list(range(start + 1, start + len(chunk_images) + 1)),
                    "observations": self._generate_text(
                        chunk_prompt,
                        chunk_images,
                        image_paths=chunk_paths,
                        max_new_tokens=_chunk_max_new_tokens(),
                        image_records=chunk_records,
                        chunk_label=f"chunk {chunk_index}/{total_chunks}",
                    ),
                }
            )

        final_prompt = _compact_chunked_final_prompt(chunk_summaries)
        return parse_json_response(
            self._generate_text(final_prompt, [], max_new_tokens=_final_max_new_tokens(), chunk_label="final text-only")
        )

    def _generate_text(
        self,
        prompt: str,
        images: list[Any],
        image_paths: list[str] | None = None,
        max_new_tokens: int | None = None,
        image_records: list[dict[str, Any]] | None = None,
        chunk_label: str | None = None,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": build_qwen_interleaved_content(prompt, images, image_paths, image_records),
            }
        ]
        text = render_qwen_chat_template(self._processor, messages)
        inputs = self._processor(text=[text], images=images or None, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        if max_new_tokens is None:
            max_new_tokens = _final_max_new_tokens()
        use_cache = os.getenv("ACCIDENT_VLM_USE_CACHE", "0").lower() in {"1", "true", "yes", "on"}
        try:
            import torch  # type: ignore

            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=use_cache,
                )
        except ImportError:
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
            )
        return self._processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]

    def _load_image(self, path: str):
        image = self._image_cls.open(path).convert("RGB")
        max_side = int(os.getenv("ACCIDENT_VLM_IMAGE_MAX_SIDE", "640"))
        if max_side > 0:
            image.thumbnail((max_side, max_side))
        return image


def build_interleaved_content(
    prompt: str,
    image_paths: list[str],
    image_records_or_package: list[dict[str, Any]] | dict[str, Any] | None = None,
    *,
    encode_images: bool = False,
) -> list[dict[str, Any]]:
    records = _normalize_image_records_for_content(image_paths, image_records_or_package)
    content: list[dict[str, Any]] = []
    for index, path in enumerate(image_paths, start=1):
        record = records.get(path, {})
        content.append({"type": "text", "text": _storyboard_image_caption(index, path, record)})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(path) if encode_images else path},
            }
        )
    content.append({"type": "text", "text": prompt})
    return content


def build_qwen_interleaved_content(
    prompt: str,
    images: list[Any],
    image_paths: list[str] | None = None,
    image_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if image_paths is None:
        image_paths = [f"image_{index}" for index in range(1, len(images) + 1)]
    records = _normalize_image_records_for_content(image_paths, image_records or [])
    content: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        path = image_paths[index - 1] if index - 1 < len(image_paths) else f"image_{index}"
        record = records.get(path, {})
        content.append({"type": "text", "text": _storyboard_image_caption(index, path, record)})
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    return content


def _normalize_image_records_for_content(
    image_paths: list[str],
    image_records_or_package: list[dict[str, Any]] | dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if isinstance(image_records_or_package, dict):
        records = image_records_or_package.get("vlm_storyboard", [])
    else:
        records = image_records_or_package or []
    by_path = {
        str(record.get("path")): record
        for record in records
        if isinstance(record, dict) and record.get("path")
    }
    return {path: by_path.get(path, {}) for path in image_paths}


def _records_for_paths(image_paths: list[str], image_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_path = {
        str(record.get("path")): record
        for record in image_records
        if isinstance(record, dict) and record.get("path")
    }
    return [by_path[path] for path in image_paths if path in by_path]


def _storyboard_image_caption(index: int, path: str, record: dict[str, Any]) -> str:
    slot = int(record.get("slot", index) or index)
    fields = [
        f"Frame {slot:02d}",
        f"time={record.get('time', 'unknown')}",
        f"phase={record.get('phase', 'unknown')}",
        f"id={record.get('id') or record.get('frame_id') or Path(path).stem}",
        f"role={record.get('role', 'visible evidence')}",
    ]
    if record.get("why_selected"):
        fields.append(f"why_selected={record['why_selected']}")
    if record.get("linked_actor_ids"):
        fields.append(f"linked_actor_ids={record['linked_actor_ids']}")
    if record.get("precomputed_hints"):
        fields.append(
            "precomputed_hints="
            + json.dumps(record["precomputed_hints"], ensure_ascii=False, sort_keys=True)
        )
    return "\n".join(fields)


class OpenAICompatibleVLMBackend:
    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str | None = None,
        timeout_sec: float = 300.0,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def generate_json(
        self,
        prompt: str,
        image_paths: list[str],
        image_records: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        chunk_size = _image_chunk_size()
        if chunk_size > 0 and (len(image_paths) > chunk_size or (_force_text_only_final() and len(image_paths) > 1)):
            return self._generate_chunked_json(prompt, image_paths, image_records or [], chunk_size)
        return parse_json_response(
            self._generate_text(
                prompt,
                image_paths,
                max_tokens=_final_max_new_tokens(),
                image_records=image_records,
                chunk_label="direct",
            )
        )

    def _generate_chunked_json(
        self,
        prompt: str,
        image_paths: list[str],
        image_records: list[dict[str, Any]],
        chunk_size: int,
    ) -> dict[str, Any]:
        chunk_summaries: list[dict[str, Any]] = []
        total_chunks = (len(image_paths) + chunk_size - 1) // chunk_size
        for chunk_index, start in enumerate(range(0, len(image_paths), chunk_size), start=1):
            chunk_paths = image_paths[start : start + chunk_size]
            chunk_records = _records_for_paths(chunk_paths, image_records)
            chunk_prompt = (
                "Analyze only this evidence image chunk. Do not decide fault or legal liability. "
                "Return concise Korean observations grounded only in visible evidence.\n"
                f"Chunk {chunk_index}/{total_chunks}; image indexes {start + 1}-{start + len(chunk_paths)}.\n\n"
                "Focus on visible actors, traffic lights/signs, road type, lane markings, "
                "weather/visibility, impact candidates, and uncertainty."
            )
            chunk_summaries.append(
                {
                    "chunk_index": chunk_index,
                    "image_indexes": list(range(start + 1, start + len(chunk_paths) + 1)),
                    "observations": self._generate_text(
                        chunk_prompt,
                        chunk_paths,
                        max_tokens=_chunk_max_new_tokens(),
                        image_records=chunk_records,
                        chunk_label=f"chunk {chunk_index}/{total_chunks}",
                    ),
                }
            )

        final_prompt = _compact_chunked_final_prompt(chunk_summaries)
        return parse_json_response(
            self._generate_text(
                final_prompt,
                [],
                max_tokens=_final_max_new_tokens(),
                chunk_label="final text-only",
            )
        )

    def _generate_text(
        self,
        prompt: str,
        image_paths: list[str],
        max_tokens: int,
        image_records: list[dict[str, Any]] | None = None,
        chunk_label: str | None = None,
    ) -> str:
        content = build_interleaved_content(
            prompt,
            image_paths,
            image_records or [],
            encode_images=True,
        )
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
            "stop": ["<think>", "</think>"],
            "chat_template_kwargs": {"enable_thinking": False},
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        prompt_chars = sum(len(item.get("text", "")) for item in content if item.get("type") == "text")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                **_openai_headers(self.api_key),
                "X-Accident-Vlm-Image-Count": str(len(image_paths)),
                "X-Accident-Vlm-Prompt-Chars": str(prompt_chars),
                "X-Accident-Vlm-Max-Tokens": str(max_tokens),
                "X-Accident-Vlm-Chunk": chunk_label or "unknown",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:  # noqa: S310
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _read_http_error_detail(exc)
            raise RuntimeError(
                f"OpenAI-compatible VLM request failed with HTTP {exc.code}: {detail}"
            ) from exc
        text = body["choices"][0]["message"]["content"]
        return text


def _build_chunked_final_prompt(
    chunk_summaries: list[dict[str, Any]],
    base_prompt: str | None = None,
) -> str:
    contract = base_prompt or (
        f"{SYSTEM_PROMPT}\n\n"
        "Return only JSON matching this exact shape. Keep enum-like values in Korean "
        "when they are visible; use 확인불가/unknown when unsupported.\n"
        f"{json.dumps(OUTPUT_TEMPLATE, ensure_ascii=False, indent=2)}"
    )
    return (
        f"{contract}\n\n"
        "The evidence images were analyzed in chunks to fit the model context. "
        "Use all chunk observations below as evidence and return the final JSON only.\n"
        f"{json.dumps(chunk_summaries, ensure_ascii=False, indent=2)}"
    )


def _compact_chunked_final_prompt(chunk_summaries: list[dict[str, Any]]) -> str:
    return _build_chunked_final_prompt(chunk_summaries)


def _openai_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _read_http_error_detail(error: urllib.error.HTTPError) -> str:
    try:
        detail = error.read().decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001
        detail = ""
    if not detail:
        detail = error.reason if isinstance(error.reason, str) else str(error)
    return detail[:2000]


def _image_data_url(path: str) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    data = Path(path).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


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


def _model_dtype() -> str:
    return os.getenv("ACCIDENT_VLM_MODEL_DTYPE", "bfloat16")


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
    if os.getenv("ACCIDENT_VLM_BACKEND", "transformers").lower() in {"openai", "openai-compatible", "sglang", "vllm"}:
        base_url = os.getenv("ACCIDENT_VLM_OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")
        api_key = os.getenv("ACCIDENT_VLM_OPENAI_API_KEY")
        timeout_sec = float(os.getenv("ACCIDENT_VLM_OPENAI_TIMEOUT_SEC", "300"))
        served_model_id = os.getenv("ACCIDENT_VLM_OPENAI_MODEL", normalized_model_id)
        return OpenAICompatibleVLMBackend(served_model_id, base_url, api_key, timeout_sec)
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
