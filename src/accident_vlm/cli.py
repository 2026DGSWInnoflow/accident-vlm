from __future__ import annotations

import json
import sys
from pathlib import Path


def analyze_video_pre_vlm(video_path: Path, config):
    from accident_vlm.pipeline import analyze_video_pre_vlm as implementation

    return implementation(video_path=video_path, config=config)


def _build_analyze_config(
    *,
    output_dir: Path,
    ocr_backend: str,
    object_detector_backend: str,
    object_detector_model: str,
    speed_mode: str = "quality",
):
    from accident_vlm.config import PipelineConfig

    normalized_speed_mode = speed_mode.strip().lower()
    if normalized_speed_mode == "fast":
        return PipelineConfig(
            output_dir=output_dir,
            regular_frame_interval_sec=1.0,
            max_selected_frames=8,
            enable_ocr=False,
            enable_motion_keyframes=False,
            enable_scene_analysis=False,
            enable_actor_tracking=False,
            enable_segment_tracking=False,
            enable_road_geometry=False,
            enable_speed_distance=False,
            enable_traffic_control=False,
            enable_event_detection=False,
            enable_event_scan=False,
            ocr_backend="none",
            object_detector_backend="none",
            object_detector_model=object_detector_model,
            vlm_frame_budget=8,
        )
    if normalized_speed_mode != "quality":
        raise ValueError("speed_mode must be 'quality' or 'fast'")
    return PipelineConfig(
        output_dir=output_dir,
        ocr_backend=ocr_backend,
        object_detector_backend=object_detector_backend,
        object_detector_model=object_detector_model,
    )


def _try_fast_analyze(argv: list[str]) -> bool:
    if len(argv) < 2 or argv[1] != "analyze":
        return False

    from accident_vlm.config import QUALITY_OBJECT_DETECTOR_BACKEND

    options = {
        "ocr_backend": "auto",
        "object_detector_backend": QUALITY_OBJECT_DETECTOR_BACKEND,
        "object_detector_model": "yolov8x.pt",
        "output_dir": Path("outputs"),
        "speed_mode": "quality",
    }
    option_names = {
        "--ocr-backend": ("ocr_backend", str),
        "--detector": ("object_detector_backend", str),
        "--detector-model": ("object_detector_model", str),
        "--output-dir": ("output_dir", Path),
        "--speed-mode": ("speed_mode", str),
    }
    positional: list[str] = []
    index = 2
    while index < len(argv):
        item = argv[index]
        if item.startswith("-"):
            option = option_names.get(item)
            if option is None or index + 1 >= len(argv):
                return False
            key, converter = option
            options[key] = converter(argv[index + 1])
            index += 2
            continue
        positional.append(item)
        index += 1

    if not positional or len(positional) > 2:
        return False

    video_path = Path(positional[0])
    output_path = Path(positional[1]) if len(positional) == 2 else Path("outputs/pre_vlm_context.json")
    config = _build_analyze_config(
        output_dir=options["output_dir"],
        ocr_backend=options["ocr_backend"],
        object_detector_backend=options["object_detector_backend"],
        object_detector_model=options["object_detector_model"],
        speed_mode=options["speed_mode"],
    )
    context = analyze_video_pre_vlm(video_path=video_path, config=config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(context.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Pre-VLM context written to {output_path}")
    print(str(output_path))
    return True


if __name__ == "__main__" and _try_fast_analyze(sys.argv):
    raise SystemExit(0)


from typing import Annotated

import typer

from accident_vlm import __version__
from accident_vlm.config import DEFAULT_QWEN_MODEL_ID, PipelineConfig, QUALITY_OBJECT_DETECTOR_BACKEND

app = typer.Typer(
    help="Accident video fact extraction tools.",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the installed version and exit."),
    ] = False,
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def check() -> None:
    typer.echo("accident-vlm is installed")


@app.command()
def analyze(
    video_path: Path,
    output_path: Annotated[
        Path,
        typer.Argument(),
    ] = Path("outputs/pre_vlm_context.json"),
    ocr_backend: Annotated[
        str,
        typer.Option("--ocr-backend", help="OCR backend: auto, easyocr, pytesseract, none."),
    ] = "auto",
    object_detector_backend: Annotated[
        str,
        typer.Option("--detector", help="Object detector/tracker: none, ultralytics, bytetrack, botsort."),
    ] = QUALITY_OBJECT_DETECTOR_BACKEND,
    object_detector_model: Annotated[
        str,
        typer.Option("--detector-model", help="Detector model name/path for server execution."),
    ] = "yolov8x.pt",
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", help="Directory for extracted frames and module artifacts."),
    ] = Path("outputs"),
    speed_mode: Annotated[
        str,
        typer.Option("--speed-mode", help="Preprocessing mode: quality or fast."),
    ] = "quality",
) -> None:
    config = _build_analyze_config(
        output_dir=output_dir,
        ocr_backend=ocr_backend,
        object_detector_backend=object_detector_backend,
        object_detector_model=object_detector_model,
        speed_mode=speed_mode,
    )
    context = analyze_video_pre_vlm(video_path=video_path, config=config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(context.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(f"Pre-VLM context written to {output_path}")
    typer.echo(str(output_path))


@app.command()
def analyze_full(
    video_path: Path,
    pre_vlm_output_path: Annotated[
        Path,
        typer.Option("--pre-vlm-output", help="Path for intermediate evidence package JSON."),
    ] = Path("outputs/pre_vlm_context.json"),
    final_output_path: Annotated[
        Path,
        typer.Option("--final-output", help="Path for final accident fact JSON."),
    ] = Path("outputs/accident_facts.json"),
    ocr_backend: Annotated[str, typer.Option("--ocr-backend")] = "auto",
    object_detector_backend: Annotated[str, typer.Option("--detector")] = QUALITY_OBJECT_DETECTOR_BACKEND,
    object_detector_model: Annotated[str, typer.Option("--detector-model")] = "yolov8x.pt",
    qwen_model_id: Annotated[str, typer.Option("--qwen-model")] = DEFAULT_QWEN_MODEL_ID,
    device: Annotated[str, typer.Option("--device")] = "auto",
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("outputs"),
) -> None:
    config = PipelineConfig(
        output_dir=output_dir,
        ocr_backend=ocr_backend,
        object_detector_backend=object_detector_backend,
        object_detector_model=object_detector_model,
        qwen_model_id=qwen_model_id,
        device=device,
        enable_vlm=True,
    )
    from accident_vlm.modules.vlm_composer import compose_final_facts, create_qwen_backend, write_final_facts
    context = analyze_video_pre_vlm(video_path=video_path, config=config)
    pre_vlm_output_path.parent.mkdir(parents=True, exist_ok=True)
    pre_vlm_output_path.write_text(
        json.dumps(context.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    backend = create_qwen_backend(config)
    final_facts = compose_final_facts(context, backend)
    write_final_facts(final_facts, final_output_path)
    typer.echo(f"Pre-VLM context written to {pre_vlm_output_path}")
    typer.echo(f"Final accident facts written to {final_output_path}")


@app.command()
def evaluate_dataset(
    result_dir: Annotated[Path, typer.Argument(help="Directory containing accident_facts JSON files.")],
    label_root: Annotated[Path, typer.Argument(help="AI Hub label root containing dataset JSON labels.")],
    output_path: Annotated[Path, typer.Option("--output", help="Evaluation report path.")] = Path(
        "outputs/dataset_evaluation.json"
    ),
) -> None:
    from accident_vlm.evaluation import evaluate_result_against_label, load_dataset_labels, summarize_evaluation

    labels = load_dataset_labels(label_root)
    items = []
    for result_path in sorted(result_dir.rglob("*.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        label = labels.get(result_path.stem)
        if label is None:
            continue
        item = evaluate_result_against_label(result, label)
        item["result_path"] = str(result_path)
        item["label_path"] = label["path"]
        items.append(item)
    report = {"summary": summarize_evaluation(items), "items": items}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo(f"Dataset evaluation written to {output_path}")


@app.command()
def benchmark_api(
    video_root: Annotated[Path, typer.Argument(help="Directory containing benchmark videos.")],
    label_root: Annotated[
        Path | None,
        typer.Option("--label-root", help="Optional AI Hub label root for accuracy checks."),
    ] = None,
    output_path: Annotated[
        Path,
        typer.Option("--output", help="Benchmark report JSON path."),
    ] = Path("outputs/api_benchmark.json"),
    api_base_url: Annotated[
        str,
        typer.Option("--api-base-url", help="Accident VLM API base URL."),
    ] = "https://vlm.qen.kr",
    sample_limit: Annotated[
        int,
        typer.Option("--sample-limit", help="Maximum number of videos to run; 0 means all."),
    ] = 0,
    poll_interval_sec: Annotated[float, typer.Option("--poll-interval-sec")] = 10.0,
    job_timeout_sec: Annotated[float, typer.Option("--job-timeout-sec")] = 900.0,
    qwen_model_id: Annotated[str, typer.Option("--qwen-model")] = DEFAULT_QWEN_MODEL_ID,
    object_detector_backend: Annotated[str, typer.Option("--detector")] = QUALITY_OBJECT_DETECTOR_BACKEND,
    object_detector_model: Annotated[str, typer.Option("--detector-model")] = "yolov8x.pt",
    verify_tls: Annotated[
        bool,
        typer.Option("--verify-tls/--no-verify-tls", help="Verify HTTPS certificates."),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option("--verbose/--quiet", help="Print per-video benchmark progress."),
    ] = True,
) -> None:
    from accident_vlm.benchmark import BenchmarkOptions, run_api_benchmark

    report = run_api_benchmark(
        BenchmarkOptions(
            api_base_url=api_base_url,
            video_root=video_root,
            label_root=label_root,
            output_path=output_path,
            sample_limit=sample_limit,
            poll_interval_sec=poll_interval_sec,
            job_timeout_sec=job_timeout_sec,
            qwen_model_id=qwen_model_id,
            object_detector_backend=object_detector_backend,
            object_detector_model=object_detector_model,
            verify_tls=verify_tls,
            verbose=verbose,
        )
    )
    typer.echo(f"API benchmark written to {output_path}")
    typer.echo(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
