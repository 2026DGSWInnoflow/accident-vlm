import cv2
import numpy as np

from accident_vlm.modules.evidence_visuals import build_visual_evidence
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
    assert cv2.imread(overlays[0]["path"]) is not None
    assert cv2.imread(crops[0]["path"]) is not None
