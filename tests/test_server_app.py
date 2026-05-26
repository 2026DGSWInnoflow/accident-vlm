from fastapi.testclient import TestClient

import accident_vlm.server.app as server_app
from accident_vlm.server.app import create_app
from accident_vlm.server.schemas import AnalysisSpeedMode


def test_upload_job_accepts_fast_speed_mode_form(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_analysis_job(job_store, job_id, video_path, options):
        captured["speed_mode"] = options.speed_mode

    monkeypatch.setattr(server_app, "run_analysis_job", fake_run_analysis_job)
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/v1/jobs/upload",
        files={"file": ("sample.mp4", b"not a real video", "video/mp4")},
        data={"speed_mode": "fast"},
    )

    assert response.status_code == 200
    assert captured["speed_mode"] == AnalysisSpeedMode.FAST
