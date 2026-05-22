from pathlib import Path

from accident_vlm.server.job_store import JobStore
from accident_vlm.server.schemas import AnalysisMode, JobStatus


def test_job_store_updates_running_progress_and_intermediate_paths(tmp_path: Path) -> None:
    store = JobStore(tmp_path)
    record = store.create(AnalysisMode.FULL, Path("sample.mp4"))
    store.set_running(record.job_id)
    running = store.get(record.job_id)

    updated = store.set_progress(
        record.job_id,
        stage="vlm_composition",
        progress_message="pre-vlm context ready",
        pre_vlm_output_path=tmp_path / "pre_vlm_context.json",
    )

    assert updated.status == JobStatus.RUNNING
    assert updated.stage == "vlm_composition"
    assert updated.progress_message == "pre-vlm context ready"
    assert updated.pre_vlm_output_path == str(tmp_path / "pre_vlm_context.json")
    assert updated.updated_at > running.updated_at
