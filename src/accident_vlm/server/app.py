from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile

from accident_vlm.server.job_store import JobStore
from accident_vlm.server.runner import run_analysis_job
from accident_vlm.server.schemas import (
    AnalysisMode,
    AnalysisOptions,
    JobCreatedResponse,
    JobRecord,
    JobStatus,
    PathAnalysisRequest,
    ResultResponse,
)


def create_app(job_root: Path = Path("outputs/api_jobs")) -> FastAPI:
    app = FastAPI(
        title="Accident VLM API",
        version="0.1.0",
        description="Evidence-constrained dashcam accident video analysis API.",
    )
    job_store = JobStore(job_root)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/jobs/from-path", response_model=JobCreatedResponse)
    def create_job_from_path(
        request: PathAnalysisRequest,
        background_tasks: BackgroundTasks,
    ) -> JobCreatedResponse:
        video_path = request.video_path.expanduser().resolve()
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"video not found: {video_path}")
        record = job_store.create(request.options.mode, video_path)
        background_tasks.add_task(run_analysis_job, job_store, record.job_id, video_path, request.options)
        return _created_response(record)

    @app.post("/v1/jobs/upload", response_model=JobCreatedResponse)
    def create_job_from_upload(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        mode: AnalysisMode = Form(AnalysisMode.PRE_VLM),
        ocr_backend: str = Form("auto"),
        object_detector_backend: str = Form("none"),
        object_detector_model: str = Form("yolov8x.pt"),
        qwen_model_id: str = Form("/home/minsung0830/accident-vlm/models/Qwen3.6-27B"),
        device: str = Form("auto"),
        regular_frame_interval_sec: float = Form(1.0),
        max_selected_frames: int = Form(16),
    ) -> JobCreatedResponse:
        options = AnalysisOptions(
            mode=mode,
            ocr_backend=ocr_backend,
            object_detector_backend=object_detector_backend,
            object_detector_model=object_detector_model,
            qwen_model_id=qwen_model_id,
            device=device,
            regular_frame_interval_sec=regular_frame_interval_sec,
            max_selected_frames=max_selected_frames,
        )
        record = job_store.create(options.mode, Path(file.filename or "upload.mp4"))
        input_path = job_store.input_path(record.job_id, file.filename or "upload.mp4")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with input_path.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        background_tasks.add_task(run_analysis_job, job_store, record.job_id, input_path, options)
        return _created_response(record)

    @app.get("/v1/jobs/{job_id}", response_model=JobRecord)
    def get_job(job_id: str) -> JobRecord:
        try:
            return job_store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/v1/jobs/{job_id}/result", response_model=ResultResponse)
    def get_job_result(job_id: str) -> ResultResponse:
        try:
            record = job_store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        if record.status != JobStatus.SUCCEEDED:
            raise HTTPException(status_code=409, detail=f"job is {record.status}")
        return ResultResponse(job=record, result=job_store.read_result(record))

    return app


def _created_response(record: JobRecord) -> JobCreatedResponse:
    return JobCreatedResponse(
        job_id=record.job_id,
        status=record.status,
        status_url=f"/v1/jobs/{record.job_id}",
        result_url=f"/v1/jobs/{record.job_id}/result",
    )


app = create_app()
