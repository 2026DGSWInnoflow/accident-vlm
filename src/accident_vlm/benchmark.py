from __future__ import annotations

import json
import mimetypes
import ssl
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from accident_vlm.evaluation import evaluate_result_against_label, load_dataset_labels
from accident_vlm.utils.timecode import parse_timecode


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
BENCHMARK_MANIFEST_SCHEMA_VERSION = "accident_vlm.benchmark_manifest.v1"


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
    verify_tls: bool = True
    verbose: bool = False


def discover_videos(video_root: Path, sample_limit: int = 0) -> list[Path]:
    videos = sorted(path for path in video_root.rglob("*") if path.suffix.lower() in VIDEO_SUFFIXES)
    if sample_limit > 0:
        return videos[:sample_limit]
    return videos


def load_benchmark_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BENCHMARK_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported benchmark manifest schema: {manifest.get('schema_version')}")
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("benchmark manifest requires an items list")
    normalized_items = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"manifest item {index} must be an object")
        labels = item.get("labels")
        if not isinstance(labels, dict):
            raise ValueError(f"manifest item {index} requires labels")
        video_path = Path(item.get("video_path", ""))
        if not video_path.exists():
            raise ValueError(f"manifest item {index} video_path does not exist: {video_path}")
        normalized_items.append(
            {
                "id": str(item.get("id") or video_path.stem),
                "video_path": str(video_path),
                "labels": labels,
                "metadata": item.get("metadata", {}),
            }
        )
    return {"schema_version": BENCHMARK_MANIFEST_SCHEMA_VERSION, "items": normalized_items}


def compute_preprocessing_metrics(
    precomputed: dict[str, Any],
    final_output: dict[str, Any],
    labels: dict[str, Any],
) -> dict[str, float | None]:
    return {
        "frame_recall": _collision_recall_at_k(precomputed.get("event_candidates", []), labels, k=3),
        "collision_recall_at_3": _collision_recall_at_k(precomputed.get("event_candidates", []), labels, k=3),
        "actor_recall": _actor_recall(precomputed.get("tracks", []), labels.get("actors", [])),
        "actor_miss_rate": _actor_miss_rate(precomputed.get("tracks", []), labels.get("actors", [])),
        "actor_false_positive_rate": _actor_false_positive_rate(
            precomputed.get("tracks", []), labels.get("actors", [])
        ),
        "tracking_continuity": _tracking_continuity(precomputed.get("tracks", [])),
        "track_fragmentation_rate": _track_fragmentation_rate(precomputed.get("tracks", [])),
        "signal_accuracy": _signal_accuracy(precomputed.get("traffic_control", {}), labels.get("traffic_signal")),
        "lane_count_accuracy": _lane_count_accuracy(precomputed.get("road_geometry", {}), labels.get("lane_count")),
        "ocr_speed_accuracy": _ocr_speed_accuracy(precomputed.get("ocr_summary", {}), labels.get("ocr", {})),
        "insurance_known_unknown_appropriateness": _insurance_known_unknown_appropriateness(final_output),
        "uncertainty_presence": 1.0 if final_output.get("uncertainties") else 0.0,
    }


def run_api_benchmark(options: BenchmarkOptions) -> dict[str, Any]:
    labels = load_dataset_labels(options.label_root) if options.label_root else {}
    videos = discover_videos(options.video_root, options.sample_limit)
    items: list[dict[str, Any]] = []
    started_at = _utc_timestamp()

    for index, video_path in enumerate(videos, start=1):
        if options.verbose:
            print(f"[{index}/{len(videos)}] start {video_path.name}", flush=True)
        item = run_single_api_benchmark_item(video_path, options, labels.get(video_path.stem))
        item["sample_index"] = index
        item["sample_count"] = len(videos)
        items.append(item)
        if options.verbose:
            print(
                f"[{index}/{len(videos)}] {item.get('status')} "
                f"{video_path.name} elapsed={item.get('elapsed_sec')}s",
                flush=True,
            )

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
            "verify_tls": options.verify_tls,
            "verbose": options.verbose,
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
        record = wait_for_job(
            options.api_base_url,
            job_id,
            options.poll_interval_sec,
            options.job_timeout_sec,
            verify_tls=options.verify_tls,
        )
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
            result_response = get_job_result(options.api_base_url, job_id, verify_tls=options.verify_tls)
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
        verify_tls=options.verify_tls,
    )


def wait_for_job(
    api_base_url: str,
    job_id: str,
    poll_interval_sec: float,
    timeout_sec: float,
    verify_tls: bool = True,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_record: dict[str, Any] | None = None
    while time.monotonic() <= deadline:
        last_record = get_job(api_base_url, job_id, verify_tls=verify_tls)
        if last_record.get("status") in {"succeeded", "failed"}:
            return last_record
        time.sleep(poll_interval_sec)
    if last_record is None:
        last_record = {"job_id": job_id}
    return {**last_record, "status": "timeout", "error": f"job timeout after {timeout_sec:.0f}s"}


def get_job(api_base_url: str, job_id: str, verify_tls: bool = True) -> dict[str, Any]:
    return _request_json(f"{api_base_url.rstrip('/')}/v1/jobs/{job_id}", timeout=60, verify_tls=verify_tls)


def get_job_result(api_base_url: str, job_id: str, verify_tls: bool = True) -> dict[str, Any]:
    return _request_json(
        f"{api_base_url.rstrip('/')}/v1/jobs/{job_id}/result",
        timeout=120,
        verify_tls=verify_tls,
    )


def inspect_result_quality(result: dict[str, Any]) -> dict[str, Any]:
    actors = result.get("actors") if isinstance(result.get("actors"), list) else []
    timeline = result.get("timeline") if isinstance(result.get("timeline"), list) else []
    uncertainties = result.get("uncertainties") if isinstance(result.get("uncertainties"), list) else []
    insurance_fields = result.get("insurance_claim_fields")
    insurance_known_count, insurance_total = _insurance_field_counts(insurance_fields)
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
        "insurance_fields_known_count": insurance_known_count,
        "insurance_fields_total": insurance_total,
        "accident_type_candidate_known": _has_known_accident_type_candidate(
            result.get("accident_type_candidates")
        ),
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
            "insurance_fields_avg_fill_rate": None,
            "accident_type_candidates_present": None,
        }
    quality = [item.get("quality", {}) for item in succeeded]
    insurance_rates = [
        item.get("insurance_fields_known_count", 0) / item["insurance_fields_total"]
        for item in quality
        if item.get("insurance_fields_total")
    ]
    return {
        "objective_summary": _ratio(sum(1 for item in quality if item.get("has_objective_summary")), len(quality)),
        "scene_type_known": _ratio(sum(1 for item in quality if item.get("scene_type_known")), len(quality)),
        "traffic_signal_known": _ratio(sum(1 for item in quality if item.get("traffic_signal_known")), len(quality)),
        "actors_present": _ratio(sum(1 for item in quality if item.get("actor_count", 0) > 0), len(quality)),
        "timeline_present": _ratio(sum(1 for item in quality if item.get("timeline_count", 0) > 0), len(quality)),
        "collision_known": _ratio(sum(1 for item in quality if item.get("collision_known")), len(quality)),
        "insurance_fields_avg_fill_rate": round(mean(insurance_rates), 4) if insurance_rates else None,
        "accident_type_candidates_present": _ratio(
            sum(1 for item in quality if item.get("accident_type_candidate_known")),
            len(quality),
        ),
    }


def _insurance_field_counts(value: Any) -> tuple[int, int]:
    expected_keys = {
        "accident_datetime",
        "location",
        "road_shape",
        "lane_count",
        "ego_direction",
        "other_direction",
        "damage_parts",
        "police_report_visible",
    }
    if not isinstance(value, dict):
        return 0, len(expected_keys)
    known = 0
    for key in expected_keys:
        item = value.get(key)
        if isinstance(item, dict):
            known += int(_is_known(item.get("value")))
        elif isinstance(item, list):
            known += int(bool(item))
        else:
            known += int(_is_known(item))
    return known, len(expected_keys)


def _has_known_accident_type_candidate(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for candidate in value.values():
        if not isinstance(candidate, dict):
            continue
        status = str(candidate.get("status") or "").lower()
        if status in {"observed", "computed", "inferred", "candidate"} and (
            candidate.get("evidence") or candidate.get("source")
        ):
            return True
    return False


def _collision_recall_at_k(event_candidates: list[dict], labels: dict[str, Any], k: int) -> float | None:
    expected_time = labels.get("collision_time")
    if not expected_time:
        return None
    try:
        expected_seconds = parse_timecode(str(expected_time))
    except ValueError:
        return None
    top_events = sorted(
        [event for event in event_candidates if isinstance(event, dict)],
        key=lambda event: -float(event.get("event_score") or 0.0),
    )[:k]
    for event in top_events:
        try:
            event_seconds = parse_timecode(str(event.get("time", "")))
        except ValueError:
            continue
        if abs(event_seconds - expected_seconds) <= 0.75:
            return 1.0
    return 0.0


def _actor_recall(tracks: list[dict], expected_actors: list[str]) -> float | None:
    if not expected_actors:
        return None
    detected = {str(track.get("track_id")) for track in tracks if isinstance(track, dict)}
    return _ratio(sum(1 for actor in expected_actors if str(actor) in detected), len(expected_actors))


def _actor_miss_rate(tracks: list[dict], expected_actors: list[str]) -> float | None:
    recall = _actor_recall(tracks, expected_actors)
    return round(1.0 - recall, 4) if recall is not None else None


def _actor_false_positive_rate(tracks: list[dict], expected_actors: list[str]) -> float | None:
    detected = {str(track.get("track_id")) for track in tracks if isinstance(track, dict)}
    if not detected:
        return 0.0 if not expected_actors else None
    expected = {str(actor) for actor in expected_actors}
    false_positives = detected - expected
    return _ratio(len(false_positives), len(detected))


def _tracking_continuity(tracks: list[dict]) -> float | None:
    quality_scores = []
    for track in tracks:
        if not isinstance(track, dict):
            continue
        quality = track.get("track_quality", {})
        fragmentation = float(quality.get("fragmentation_score") or 0.0) if isinstance(quality, dict) else 0.0
        quality_scores.append(max(0.0, 1.0 - min(fragmentation, 100.0) / 100.0))
    return round(mean(quality_scores), 4) if quality_scores else None


def _track_fragmentation_rate(tracks: list[dict]) -> float | None:
    fragmentations = []
    for track in tracks:
        if not isinstance(track, dict):
            continue
        quality = track.get("track_quality", {})
        fragmentations.append(min(1.0, float(quality.get("fragmentation_score") or 0.0) / 100.0))
    return round(mean(fragmentations), 4) if fragmentations else None


def _signal_accuracy(traffic_control: dict[str, Any], expected_signal: str | None) -> float | None:
    if not expected_signal:
        return None
    signal = traffic_control.get("signal", {}) if isinstance(traffic_control, dict) else {}
    return 1.0 if signal.get("value") == expected_signal else 0.0


def _lane_count_accuracy(road_geometry: dict[str, Any], expected_lane_count: int | None) -> float | None:
    if expected_lane_count is None:
        return None
    lane_count = _nested_value(road_geometry, "visible_lane_count", "value")
    return 1.0 if lane_count == expected_lane_count else 0.0


def _ocr_speed_accuracy(ocr_summary: dict[str, Any], expected_ocr: dict[str, Any]) -> float | None:
    expected_speed = expected_ocr.get("speed_kmh") if isinstance(expected_ocr, dict) else None
    if expected_speed is None:
        return None
    observed_speed = _nested_value(ocr_summary, "speed", "numeric_kmh")
    if observed_speed is None:
        return 0.0
    return 1.0 if abs(float(observed_speed) - float(expected_speed)) <= 3.0 else 0.0


def _insurance_known_unknown_appropriateness(final_output: dict[str, Any]) -> float | None:
    fields = final_output.get("insurance_claim_fields")
    if not isinstance(fields, dict):
        return None
    total = 0
    appropriate = 0
    for value in fields.values():
        if isinstance(value, list):
            continue
        if not isinstance(value, dict):
            continue
        total += 1
        if _is_known(value.get("value")) or value.get("value") == "확인불가" or value.get("confidence") == "unknown":
            appropriate += 1
    return _ratio(appropriate, total)


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
    verify_tls: bool = True,
) -> dict[str, Any]:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "curl/8.7.1",
        **(headers or {}),
    }
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    context = None if verify_tls else ssl._create_unverified_context()  # noqa: S323
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:  # noqa: S310
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
    if isinstance(value, (list, dict, set, tuple)):
        return bool(value)
    return value not in {None, "", "확인불가", "unknown", "Unknown", "UNKNOWN"}


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
