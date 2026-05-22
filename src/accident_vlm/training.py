from __future__ import annotations

from pathlib import Path
from typing import Any

from accident_vlm.modules.actor_tracking import ACCIDENT_ACTOR_TAXONOMY


def write_yolo_dataset_yaml(dataset_root: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [key for key in ACCIDENT_ACTOR_TAXONOMY]
    lines = [
        f"path: {dataset_root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(names)],
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def train_custom_yolo(
    base_model: str,
    dataset_yaml: Path,
    epochs: int = 100,
    imgsz: int = 960,
    device: str = "auto",
    project: Path | None = None,
    run_name: str = "accident-traffic-custom",
) -> dict[str, Any]:
    yolo = _load_yolo_class()
    model = yolo(base_model)
    kwargs = {
        "data": str(dataset_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "device": device,
        "name": run_name,
    }
    if project:
        kwargs["project"] = str(project)
    result = model.train(**kwargs)
    return {
        "training_status": "submitted",
        "base_model": base_model,
        "dataset_yaml": str(dataset_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "device": device,
        "result": _to_serializable(result),
    }


def evaluate_actor_detection_map(
    model_path: Path,
    dataset_yaml: Path,
    split: str = "test",
    imgsz: int = 960,
    device: str = "auto",
) -> dict[str, Any]:
    yolo = _load_yolo_class()
    model = yolo(str(model_path))
    results = model.val(data=str(dataset_yaml), split=split, imgsz=imgsz, device=device)
    box = getattr(results, "box", None)
    return {
        "model_path": str(model_path),
        "dataset_yaml": str(dataset_yaml),
        "split": split,
        "actor_detection_mAP": round(float(getattr(box, "map", 0.0) or 0.0), 4),
        "actor_detection_mAP50": round(float(getattr(box, "map50", 0.0) or 0.0), 4),
        "metric_source": "ultralytics.val",
    }


def _load_yolo_class():
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for custom YOLO training/evaluation") from exc
    return YOLO


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if hasattr(value, "save_dir"):
        return {"save_dir": str(value.save_dir)}
    return str(value)
