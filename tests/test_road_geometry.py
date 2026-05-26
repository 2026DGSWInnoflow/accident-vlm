import cv2
import numpy as np

from accident_vlm.modules.road_geometry import _detect_road_markings, analyze_road_geometry
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
    assert result["homography"]["matrix"] is not None
    assert result["homography"]["pixels_per_meter"] is not None
    assert result["bev"]["overlays"]
    assert cv2.imread(result["bev"]["overlays"][0]["path"]) is not None
    assert cv2.imread(result["lane_segmentation"]["overlays"][0]["path"]) is not None


def test_analyze_road_geometry_votes_lane_and_road_marking_candidates(tmp_path) -> None:
    frame_path = tmp_path / "markings.jpg"
    image = np.zeros((220, 320, 3), dtype=np.uint8)
    cv2.line(image, (100, 219), (145, 95), (255, 255, 255), 3)
    cv2.line(image, (220, 219), (175, 95), (255, 255, 255), 3)
    cv2.line(image, (35, 170), (285, 170), (255, 255, 255), 3)
    for y in range(120, 152, 10):
        cv2.line(image, (60, y), (260, y), (255, 255, 255), 2)
    cv2.line(image, (160, 219), (160, 105), (0, 220, 220), 3)
    cv2.imwrite(str(frame_path), image)

    result = analyze_road_geometry(
        [
            SelectedFrame(
                id="frame_000010",
                time="00:00.333",
                frame_index=10,
                purpose="event_scan_pre_impact",
                path=str(frame_path),
            )
        ],
        lane_width_m=3.2,
        output_dir=tmp_path / "road",
    )

    markings = {item["type"] for item in result["road_marking_candidates"]}
    assert {"stop_line", "crosswalk", "centerline_yellow"} <= markings
    assert result["lane_detection_vote"]["final_lane_count"] == result["visible_lane_count"]["value"]
    assert result["bev"]["confidence_score"] > 0
    assert "failure_reasons" in result["bev"]
    assert result["homography"]["lane_width_prior_m"] == 3.2


def test_analyze_road_geometry_uses_onnx_lane_segmentation_backend(monkeypatch, tmp_path) -> None:
    frame_path = tmp_path / "lane_model.jpg"
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.line(image, (55, 119), (75, 50), (255, 255, 255), 3)
    cv2.line(image, (105, 119), (85, 50), (255, 255, 255), 3)
    cv2.imwrite(str(frame_path), image)
    model_path = tmp_path / "lane.onnx"
    model_path.write_bytes(b"onnx")

    class FakeNet:
        def setInput(self, blob):
            self.blob = blob

        def forward(self):
            mask = np.zeros((1, 1, 64, 64), dtype=np.float32)
            mask[:, :, :, 20:23] = 1.0
            mask[:, :, :, 42:45] = 1.0
            return mask

    monkeypatch.setattr(cv2.dnn, "readNetFromONNX", lambda path: FakeNet())

    result = analyze_road_geometry(
        [
            SelectedFrame(
                id="frame_000100",
                time="00:03.333",
                frame_index=100,
                purpose="regular_context",
                path=str(frame_path),
            )
        ],
        output_dir=tmp_path / "road",
        lane_segmentation_model_path=model_path,
    )

    assert result["lane_segmentation"]["segmentation_backend_available"] is True
    assert result["lane_segmentation"]["model_path"] == str(model_path)
    assert result["lane_detection_vote"]["segmentation_lane_count"] >= 1
    assert result["lane_segmentation"]["segmentation_overlays"]


def test_detect_road_markings_uses_downscaled_hough_input(monkeypatch) -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(image, (80, 360), (560, 360), (255, 255, 255), 8)
    shapes = []
    original_hough = cv2.HoughLinesP

    def record_hough(edges, *args, **kwargs):
        shapes.append(edges.shape)
        return original_hough(edges, *args, **kwargs)

    monkeypatch.setattr(cv2, "HoughLinesP", record_hough)

    candidates = _detect_road_markings(image, "frame_000001")

    assert any(item["type"] == "stop_line" for item in candidates)
    assert shapes
    assert all(width <= 240 for _height, width in shapes)
