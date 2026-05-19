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
