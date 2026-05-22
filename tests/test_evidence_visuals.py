import cv2
import numpy as np

from accident_vlm.modules.evidence_visuals import build_event_evidence_overlays, build_visual_evidence
from accident_vlm.schemas.preprocessing import SelectedFrame


def test_build_visual_evidence_writes_tracking_overlay_and_actor_crop(tmp_path) -> None:
    frame_path = tmp_path / "frame.jpg"
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[20:60, 30:90] = 255
    cv2.imwrite(str(frame_path), image)

    overlays, crops = build_visual_evidence(
        selected_frames=[
            SelectedFrame(
                id="frame_000001",
                frame_index=1,
                time="00:00.500",
                purpose="motion",
                path=str(frame_path),
            )
        ],
        tracks=[
            {
                "track_id": "vehicle_1",
                "type": "car",
                "movement_candidate": "straight",
                "positions": [
                    {
                        "frame_id": "frame_000001",
                        "time": 0.5,
                        "bbox": [30, 20, 90, 60],
                    }
                ],
            }
        ],
        output_dir=tmp_path / "evidence",
    )

    assert len(overlays) == 1
    assert len(crops) == 1
    assert overlays[0]["purpose"] == "tracking_overlay"
    assert crops[0]["purpose"] == "actor_crop"
    assert crops[0]["bbox"] == [30, 20, 90, 60]
    assert crops[0]["quality"]["area_ratio"] > 0
    assert crops[0]["context_padding_px"] > 0
    assert crops[0]["why_selected"]
    assert crops[0]["expected_use"]
    assert overlays[0]["paired_original_frame_id"] == "frame_000001"
    assert "report_path" in overlays[0]
    assert cv2.imread(overlays[0]["path"]) is not None
    assert cv2.imread(crops[0]["path"]) is not None
    assert (tmp_path / "evidence" / "evidence_report.html").exists()


def test_build_visual_evidence_upscales_tiny_crops_and_records_risk(tmp_path) -> None:
    frame_path = tmp_path / "tiny.jpg"
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[10:18, 10:18] = 255
    cv2.imwrite(str(frame_path), image)

    _overlays, crops = build_visual_evidence(
        selected_frames=[
            SelectedFrame(
                id="frame_000002",
                frame_index=2,
                time="00:00.067",
                purpose="impact_candidate",
                path=str(frame_path),
            )
        ],
        tracks=[
            {
                "track_id": "person_1",
                "type": "person",
                "confidence": "low",
                "positions": [{"frame_id": "frame_000002", "time": 0.067, "bbox": [10, 10, 18, 18]}],
            }
        ],
        output_dir=tmp_path / "evidence",
    )

    crop_image = cv2.imread(crops[0]["path"])
    assert crop_image.shape[0] >= 48
    assert crops[0]["quality"]["too_small"] is True
    assert "small_crop" in crops[0]["risk"]


def test_build_event_evidence_overlays_creates_frame_pair_overlay(tmp_path) -> None:
    frame_paths = []
    selected_frames = []
    for index in (1, 2):
        frame_path = tmp_path / f"frame_{index}.jpg"
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[20:60, 30 + index : 70 + index] = 255
        cv2.imwrite(str(frame_path), image)
        frame_paths.append(frame_path)
        selected_frames.append(
            SelectedFrame(
                id=f"frame_00000{index}",
                frame_index=index,
                time=f"00:00.0{index}0",
                purpose="impact_candidate",
                path=str(frame_path),
            )
        )

    events = [
        {
            "id": "collision_1",
            "event_type": "접촉",
            "event_score": 91,
            "candidate_class": "direct_contact_candidate",
            "evidence": ["frame_000001", "frame_000002"],
            "contradicting_signals": [],
        }
    ]

    overlays = build_event_evidence_overlays(
        selected_frames,
        events,
        [
            {
                "track_id": "T1",
                "positions": [
                    {"frame_id": "frame_000001", "bbox": [30, 20, 70, 60]},
                    {"frame_id": "frame_000002", "bbox": [32, 20, 72, 60]},
                ],
            }
        ],
        tmp_path / "evidence",
    )

    assert overlays[0]["purpose"] == "event_candidate_overlay"
    assert overlays[0]["evidence"] == ["frame_000001", "frame_000002"]
    assert events[0]["overlay_evidence"] == ["event_overlay_00"]
    assert cv2.imread(overlays[0]["path"]) is not None
