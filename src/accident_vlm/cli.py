import json
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from accident_vlm import __version__
from accident_vlm.config import PipelineConfig
from accident_vlm.modules.vlm_composer import compose_final_facts, create_qwen_backend, write_final_facts
from accident_vlm.pipeline import analyze_video_pre_vlm

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
    ] = "none",
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
    object_detector_backend: Annotated[str, typer.Option("--detector")] = "none",
    object_detector_model: Annotated[str, typer.Option("--detector-model")] = "yolov8x.pt",
    qwen_model_id: Annotated[str, typer.Option("--qwen-model")] = "Qwen/Qwen3.6-27B",
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


if __name__ == "__main__":
    app()
