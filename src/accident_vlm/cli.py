import json
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from accident_vlm import __version__
from accident_vlm.config import DEFAULT_QWEN_MODEL_ID, PipelineConfig, QUALITY_OBJECT_DETECTOR_BACKEND

app = typer.Typer(
    help="Accident video fact extraction tools.",
    no_args_is_help=True,
)


def analyze_video_pre_vlm(video_path: Path, config: PipelineConfig):
    from accident_vlm.pipeline import analyze_video_pre_vlm as implementation

    return implementation(video_path=video_path, config=config)


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
) -> None:
    config = PipelineConfig(
        output_dir=output_dir,
        ocr_backend=ocr_backend,
        object_detector_backend=object_detector_backend,
        object_detector_model=object_detector_model,
    )
    context = analyze_video_pre_vlm(video_path=video_path, config=config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(context.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Pre-VLM context written to {output_path}")
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
    print(f"Pre-VLM context written to {pre_vlm_output_path}")
    print(f"Final accident facts written to {final_output_path}")


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
    print(f"Dataset evaluation written to {output_path}")


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
    print(f"API benchmark written to {output_path}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
