from __future__ import annotations

import json
import mimetypes
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from accident_vlm.evaluation import evaluate_result_against_label, load_dataset_labels


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass(frozen=True)
class BenchmarkOptions:
    api_base_url: str
    video_root: Path
    label_root: Path | None = None
    output_path: Path = Path("outputs/api_benchmark.json")
    sample_limit: int = 0
    poll_interval_sec: float = 10.0
    job_timeout_sec: float = 900.0
    mode: str = "full"
    ocr_backend: str = "auto"
    object_detector_backend: str = "bytetrack"
    object_detector_model: str = "yolov8x.pt"
    qwen_model_id: str = "/home/minsung0830/accident-vlm/models/Qwen3.6-27B-AWQ-INT4"
    device: str = "auto"


def discover_videos(video_root: Path, sample_limit: int = 0) -> list[Path]:
    videos = sorted(path for path in video_root.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES)
    if sample_limit > 0:
        return videos[:sample_limit]
    return videos


def run_api_benchmark(options: BenchmarkOptions) -> dict[str, Any]:
    labels = load_dataset_labels(options.label_root) if options.label_root else {}
    videos = discover_videos(options.video_root, options.sample_limit)
    items: list[dict[str, Any]] = []
    started_at = _utc_timestamp()

    for index, video_path in enumerate(videos, start=1):
        item = run_single_api_benchmark_item(video_path, options, labels.get(video_path.stem))
        item["sample_index"] = index
        item["sample_count"] = len(videos)
        items.append(item)

    report = {
        "schema_version": "accident_vlm.api_benchmark.v1",
        "started_at": started_at,
        "finished_at": _utc_timestamp(),
        "config": {
            "api_base_url": options.api_base_url,
            "video_root": str(options.video_root),
            "label_root": str(options.label_root) if options.label_root else None,
            "sample_limit": options.sample_limit,
            "mode": options.mode,
            "ocr_backend": options.ocr_backend,
            "object_detector_backend": options.object_detector_backend,
            "object_detector_model": options.object_detector_model,
            "qwen_model_id": options.qwen_model_id,
            "device": options.device,
        },
        "summary": summarize_benchmark_items(items),
        "items": items,
    }
    options.output_path.parent.mkdir(parents=True, exist_ok=True)
    options.output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_single_api_benchmark_item(
    video_path: Path,
    options: BenchmarkOptions,
    label: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    item: dict[str, Any] = {
        "video_name": video_path.name,
        "video_stem": video_path.stem,
        "video_path": str(video_path),
        "video_size_bytes": video_path.stat().st_size,
        "status": "client_error",
    }
    try:
        created = submit_upload_job(video_path, options)
        job_id = created["job_id"]
        item["job_id"] = job_id
        record = wait_for_job(options.api_base_url, job_id, options.poll_interval_sec, options.job_timeout_sec)
        elapsed_sec = round(time.monotonic() - start, 3)
        item.update(
            {
                "status": record.get("status"),
                "elapsed_sec": elapsed_sec,
                "server_created_at": record.get("created_at"),
                "server_updated_at": record.get("updated_at"),
                "pre_vlm_output_path": record.get("pre_vlm_output_path"),
                "final_output_path": record.get("final_output_path"),
            }
        )
        if record.get("status") == "succeeded":
            result_response = get_job_result(options.api_base_url, job_id)
            result = result_response.get("result", {})
            item["quality"] = inspect_result_quality(result)
            if label:
                item["label_path"] = label.get("path")
                item["label_evaluation"] = evaluate_result_against_label(result, label)
        else:
            item["error"] = str(record.get("error") or "")[:4000]
    except Exception as exc:  # noqa: BLE001
        item["elapsed_sec"] = round(time.monotonic() - start, 3)
        item["error"] = str(exc)[:4000]
    return item


def submit_upload_job(video_path: Path, options: BenchmarkOptions) -> dict[str, Any]:
    fields = {
        "mode": options.mode,
        "ocr_backend": options.ocr_backend,
        "object_detector_backend": options.object_detector_backend,
        "object_detector_model": options.object_detector_model,
        "qwen_model_id": options.qwen_model_id,
        "device": options.device,
    }
    body, content_type = _multipart_body(fields, "file", video_path)
    return _request_json(
        f"{options.api_base_url.rstrip('/')}/v1/jobs/upload",
        method="POST",
        data=body,
        headers={"Content-Type": content_type},
        timeout=180,
    )


def wait_for_job(
    api_base_url: str,
    job_id: str,
    poll_interval_sec: float,
    timeout_sec: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_record: dict[str, Any] | None = None
    while time.monotonic() <= deadline:
        last_record = get_job(api_base_url, job_id)
        if last_record.get("status") in {"succeeded", "failed"}:
            return last_record
        time.sleep(poll_interval_sec)
    if last_record is None:
        last_record = {"job_id": job_id}
    return {**last_record, "status": "timeout", "error": f"job timeout after {timeout_sec:.0f}s"}


def get_job(api_base_url: str, job_id: str) -> dict[str, Any]:
    return _request_json(f"{api_base_url.rstrip('/')}/v1/jobs/{job_id}", timeout=60)


def get_job_result(api_base_url: str, job_id: str) -> dict[str, Any]:
    return _request_json(f"{api_base_url.rstrip('/')}/v1/jobs/{job_id}/result", timeout=120)


def inspect_result_quality(result: dict[str, Any]) -> dict[str, Any]:
    actors = result.get("actors") if isinstance(result.get("actors"), list) else []
    timeline = result.get("timeline") if isinstance(result.get("timeline"), list) else []
    uncertainties = result.get("uncertainties") if isinstance(result.get("uncertainties"), list) else []
    return {
        "has_objective_summary": bool(result.get("objective_summary")),
        "scene_type_value": _nested_value(result, "scene_type", "value"),
        "scene_type_known": _is_known(_nested_value(result, "scene_type", "value")),
        "traffic_signal_known": _is_known(
            _nested_value(result, "traffic_control", "signal", "value")
            or _nested_value(result, "traffic_control", "signals")
        ),
        "actor_count": len(actors),
        "timeline_count": len(timeline),
        "collision_known": any(
            _is_known(value)
            for key, value in (result.get("collision") or {}).items()
            if key not in {"confidence", "source", "evidence", "note"}
        )
        if isinstance(result.get("collision"), dict)
        else False,
        "uncertainty_count": len(uncertainties),
        "objective_summary_chars": len(str(result.get("objective_summary") or "")),
    }


def summarize_benchmark_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    succeeded = [item for item in items if item.get("status") == "succeeded"]
    failed = [item for item in items if item.get("status") == "failed"]
    timed_out = [item for item in items if item.get("status") == "timeout"]
    elapsed = [float(item["elapsed_sec"]) for item in items if item.get("elapsed_sec") is not None]
    label_items = [item["label_evaluation"] for item in succeeded if item.get("label_evaluation")]
    object_matches = [
        item["checks"]["accident_object_match"]
        for item in label_items
        if item.get("checks", {}).get("accident_object_match") is not None
    ]
    return {
        "sample_count": total,
        "success_count": len(succeeded),
        "failure_count": len(failed),
        "timeout_count": len(timed_out),
        "success_rate": _ratio(len(succeeded), total),
        "avg_elapsed_sec": round(mean(elapsed), 3) if elapsed else None,
        "median_elapsed_sec": round(median(elapsed), 3) if elapsed else None,
        "field_fill_rates": _field_fill_rates(succeeded),
        "label_evaluation_count": len(label_items),
        "accident_object_match_rate": _ratio(sum(1 for value in object_matches if value), len(object_matches)),
    }


def _field_fill_rates(succeeded: list[dict[str, Any]]) -> dict[str, float | None]:
    if not succeeded:
        return {
            "objective_summary": None,
            "scene_type_known": None,
            "traffic_signal_known": None,
            "actors_present": None,
            "timeline_present": None,
            "collision_known": None,
        }
    quality = [item.get("quality", {}) for item in succeeded]
    return {
        "objective_summary": _ratio(sum(1 for item in quality if item.get("has_objective_summary")), len(quality)),
        "scene_type_known": _ratio(sum(1 for item in quality if item.get("scene_type_known")), len(quality)),
        "traffic_signal_known": _ratio(sum(1 for item in quality if item.get("traffic_signal_known")), len(quality)),
        "actors_present": _ratio(sum(1 for item in quality if item.get("actor_count", 0) > 0), len(quality)),
        "timeline_present": _ratio(sum(1 for item in quality if item.get("timeline_count", 0) > 0), len(quality)),
        "collision_known": _ratio(sum(1 for item in quality if item.get("collision_known")), len(quality)),
    }


def _multipart_body(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----accident-vlm-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            ]
        )
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode(),
            f"Content-Type: {mime_type}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _request_json(
    url: str,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _nested_value(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _is_known(value: Any) -> bool:
    return value not in {None, "", "확인불가", "unknown", "Unknown", "UNKNOWN"}


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
