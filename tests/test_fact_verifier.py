from accident_vlm.modules.fact_verifier import verify_vlm_payload_against_context
from accident_vlm.schemas.preprocessing import PipelineContext, SelectedFrame


def test_verify_vlm_payload_downgrades_unsupported_collision() -> None:
    context = PipelineContext(
        video_path="sample.mp4",
        selected_frames=[
            SelectedFrame(
                id="frame_000000",
                time="00:00.000",
                frame_index=0,
                purpose="regular_context",
            )
        ],
        event_candidates=[],
    )
    payload = {"collision": {"detected": True, "confidence": "high", "evidence": ["missing"]}}

    verified = verify_vlm_payload_against_context(payload, context)

    assert verified["collision"]["detected"] is False
    assert verified["collision"]["confidence"] == "unknown"
    assert "collision detected 값이 전처리 근거 부족으로 강등됨" in verified["uncertainties"]


def test_verify_vlm_payload_aligns_signal_with_preprocessing() -> None:
    context = PipelineContext(
        video_path="sample.mp4",
        selected_frames=[
            SelectedFrame(
                id="frame_000000",
                time="00:00.000",
                frame_index=0,
                purpose="regular_context",
            )
        ],
        traffic_control={
            "signal": {
                "value": "적색",
                "confidence": "medium",
                "evidence": ["frame_000000"],
            }
        },
    )
    payload = {"traffic_control": {"traffic_light": {"value": "녹색", "confidence": "high"}}}

    verified = verify_vlm_payload_against_context(payload, context)

    assert verified["traffic_control"]["traffic_light"]["value"] == "적색"
    assert verified["traffic_control"]["traffic_light"]["confidence"] == "medium"
