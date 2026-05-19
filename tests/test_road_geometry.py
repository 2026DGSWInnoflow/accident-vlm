import cv2
import numpy as np

from accident_vlm.modules.road_geometry import analyze_road_geometry
from accident_vlm.schemas.preprocessing import SelectedFrame


def test_analyze_road_geometry_outputs_lane_overlay_mask_and_bev_scale(tmp_path) -> None:
    frame_path = tmp_path / "lane.jpg"
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    cv2.line(image, (80, 159), (115, 70), (255, 255, 255), 3)
    cv2.line(image, (160, 159), (125, 70), (255, 255, 255), 3)
    cv2.imwrite(str(frame_path), image)

    result = analyze_road_geometry(
        [
            SelectedFrame(
                id="frame_000001",
                time="00:00.033",
                frame_index=1,
                purpose="regular_context",
                path=str(frame_path),
            )
        ],
        lane_width_m=3.2,
        output_dir=tmp_path / "road",
    )

    assert result["visible_lane_count"]["value"] >= 1
    assert result["lane_segmentation"]["method"] == "canny_hough_lane_mask"
    assert result["lane_segmentation"]["overlays"]
    assert result["homography"]["available"] is True
    assert result["homography"]["pixels_per_meter"] is not None
    assert cv2.imread(result["lane_segmentation"]["overlays"][0]["path"]) is not None
