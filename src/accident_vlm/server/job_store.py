from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from accident_vlm.server.schemas import AnalysisMode, JobRecord, JobStatus


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class JobStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create(self, mode: AnalysisMode, video_path: Path) -> JobRecord:
        job_id = uuid4().hex
        output_dir = self.root_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        now = utc_now_iso()
        record = JobRecord(
            job_id=job_id,
            status=JobStatus.QUEUED,
            mode=mode,
            video_path=str(video_path),
            created_at=now,
            updated_at=now,
            output_dir=str(output_dir),
        )
        self.save(record)
        return record

    def get(self, job_id: str) -> JobRecord:
        path = self._record_path(job_id)
        if not path.exists():
            raise KeyError(job_id)
        return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, record: JobRecord) -> None:
        record.updated_at = utc_now_iso()
        self._record_path(record.job_id).write_text(
            record.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def set_running(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        record.status = JobStatus.RUNNING
        self.save(record)
        return record

    def set_succeeded(
        self,
        job_id: str,
        pre_vlm_output_path: Path,
        final_output_path: Path | None,
    ) -> JobRecord:
        record = self.get(job_id)
        record.status = JobStatus.SUCCEEDED
        record.pre_vlm_output_path = str(pre_vlm_output_path)
        record.final_output_path = str(final_output_path) if final_output_path else None
        record.error = None
        self.save(record)
        return record

    def set_failed(self, job_id: str, error: str) -> JobRecord:
        record = self.get(job_id)
        record.status = JobStatus.FAILED
        record.error = error
        self.save(record)
        return record

    def read_result(self, record: JobRecord) -> dict:
        result_path = record.final_output_path or record.pre_vlm_output_path
        if not result_path:
            raise FileNotFoundError("result is not available")
        return json.loads(Path(result_path).read_text(encoding="utf-8"))

    def input_path(self, job_id: str, filename: str) -> Path:
        safe_name = Path(filename).name or "upload.mp4"
        return self.root_dir / job_id / "input" / safe_name

    def _record_path(self, job_id: str) -> Path:
        return self.root_dir / job_id / "job.json"
