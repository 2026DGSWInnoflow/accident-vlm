from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ACCIDENT_OBJECT_LABELS = {
    1: "차대보행자",
    2: "차대차",
    3: "차대이륜차",
    4: "차대자전거",
}


def load_dataset_labels(label_root: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for path in label_root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        video = data.get("video", {})
        video_name = video.get("video_name") or path.stem
        labels[str(video_name)] = {"path": str(path), "video": video}
    return labels


def evaluate_result_against_label(result: dict[str, Any], label: dict[str, Any]) -> dict[str, Any]:
    video_label = label.get("video", {})
    expected_object = ACCIDENT_OBJECT_LABELS.get(_to_int(video_label.get("accident_object")))
    predicted_object = _extract_predicted_accident_type(result)
    checks = {
        "has_objective_summary": bool(result.get("objective_summary")),
        "has_timeline": bool(result.get("timeline")),
        "has_traffic_control": bool(result.get("traffic_control")),
        "has_uncertainties": bool(result.get("uncertainties")),
        "accident_object_match": (
            expected_object is not None
            and predicted_object is not None
            and expected_object == predicted_object
        ),
    }
    score = sum(1 for value in checks.values() if value)
    return {
        "expected_accident_object": expected_object,
        "predicted_accident_object": predicted_object,
        "checks": checks,
        "score": score,
        "max_score": len(checks),
        "quality_bucket": "high" if score >= 4 else "medium" if score >= 3 else "low",
    }


def summarize_evaluation(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "sample_count": 0,
            "estimated_usable_count_per_100": 0,
            "note": "평가할 항목이 없음",
        }
    usable = sum(1 for item in items if item.get("quality_bucket") in {"high", "medium"})
    high = sum(1 for item in items if item.get("quality_bucket") == "high")
    return {
        "sample_count": len(items),
        "high_quality_count": high,
        "usable_count": usable,
        "estimated_usable_count_per_100": round(usable / len(items) * 100),
        "estimated_high_quality_count_per_100": round(high / len(items) * 100),
    }


def _extract_predicted_accident_type(result: dict[str, Any]) -> str | None:
    rag_hints = result.get("rag_hints", {})
    accident_type = rag_hints.get("accident_type") if isinstance(rag_hints, dict) else None
    if accident_type and accident_type != "확인불가":
        return str(accident_type)
    actor_types = {
        str(actor.get("type", ""))
        for actor in result.get("actors", [])
        if isinstance(actor, dict)
    }
    if any("보행" in actor_type or actor_type == "person" for actor_type in actor_types):
        return "차대보행자"
    if any("자전거" in actor_type for actor_type in actor_types):
        return "차대자전거"
    if any("이륜" in actor_type or "오토바이" in actor_type for actor_type in actor_types):
        return "차대이륜차"
    if len(actor_types) >= 2:
        return "차대차"
    return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
