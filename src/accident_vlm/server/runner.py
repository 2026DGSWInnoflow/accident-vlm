from __future__ import annotations

import json
import traceback
from pathlib import Path

from accident_vlm.config import PipelineConfig
from accident_vlm.modules.vlm_composer import compose_final_facts, create_qwen_backend, write_final_facts
from accident_vlm.pipeline import analyze_video_pre_vlm
from accident_vlm.server.job_store import JobStore
from accident_vlm.server.schemas import AnalysisMode, AnalysisOptions


def config_from_options(options: AnalysisOptions, output_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        output_dir=output_dir,
        regular_frame_interval_sec=options.regular_frame_interval_sec,
        ocr_backend=options.ocr_backend,
        object_detector_backend=options.object_detector_backend,
        object_detector_model=options.object_detector_model,
        qwen_model_id=options.qwen_model_id,
        device=options.device,
        enable_vlm=options.mode == AnalysisMode.FULL,
    )


def run_analysis_job(
    job_store: JobStore,
    job_id: str,
    video_path: Path,
    options: AnalysisOptions,
) -> None:
    record = job_store.set_running(job_id)
    output_dir = Path(record.output_dir)
    pre_vlm_output_path = output_dir / "pre_vlm_context.json"
    final_output_path = output_dir / "accident_facts.json"

    try:
        config = config_from_options(options, output_dir)
        context = analyze_video_pre_vlm(video_path=video_path, config=config)
        pre_vlm_output_path.write_text(
            json.dumps(context.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        final_path: Path | None = None
        if options.mode == AnalysisMode.FULL:
            backend = create_qwen_backend(config)
            final_facts = compose_final_facts(context, backend)
            write_final_facts(final_facts, final_output_path)
            final_path = final_output_path

        job_store.set_succeeded(job_id, pre_vlm_output_path, final_path)
    except Exception as exc:  # noqa: BLE001
        job_store.set_failed(job_id, f"{exc}\n{traceback.format_exc()}")
