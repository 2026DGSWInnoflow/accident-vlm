import cv2
import numpy as np

from accident_vlm.modules.traffic_control import analyze_traffic_control
from accident_vlm.schemas.preprocessing import SelectedFrame


def test_analyze_traffic_control_creates_signal_crop_and_speed_limit_sign(tmp_path) -> None:
    frame_path = tmp_path / "traffic.jpg"
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    cv2.circle(image, (100, 25), 8, (0, 0, 255), -1)
    cv2.imwrite(str(frame_path), image)

    result = analyze_traffic_control(
        [
            SelectedFrame(
                id="frame_000001",
                time="00:00.033",
                frame_index=1,
                purpose="regular_context",
                path=str(frame_path),
            )
        ],
        [
            {
                "frame_id": "frame_000001",
                "text": "제한속도 30",
                "confidence": 0.85,
                "bbox": [10, 20, 50, 60],
                "source": "ocr",
            }
        ],
        output_dir=tmp_path / "traffic",
    )

    assert result["signal"]["value"] == "적색"
    assert result["signal"]["method"] == "traffic_light_hsv_crop_temporal_vote"
    assert result["signal"]["crops"]
    assert cv2.imread(result["signal"]["crops"][0]["path"]) is not None
    assert result["signs"][0]["value"] == "제한속도 30"
    assert result["signs"][0]["evidence"] == ["frame_000001"]


def test_analyze_traffic_control_votes_signs_and_saves_failure_cases(tmp_path) -> None:
    frame_path = tmp_path / "no_signal.jpg"
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(frame_path), image)

    result = analyze_traffic_control(
        [
            SelectedFrame(
                id="frame_000010",
                time="00:00.333",
                frame_index=10,
                purpose="regular_context",
                path=str(frame_path),
            )
        ],
        [
            {"frame_id": "frame_000010", "text": "제한속도 50", "confidence": 0.70, "source": "ocr"},
            {"frame_id": "frame_000011", "text": "속도제한 50", "confidence": 0.80, "source": "ocr"},
            {"frame_id": "frame_000012", "text": "제한속도 30", "confidence": 0.55, "source": "ocr"},
        ],
        output_dir=tmp_path / "traffic",
    )

    assert result["sign_votes"][0]["value"] == "제한속도 50"
    assert result["sign_votes"][0]["vote_count"] == 2
    assert result["failure_cases"][0]["reason"] == "traffic_light_not_detected"
    assert cv2.imread(result["failure_cases"][0]["path"]) is not None
