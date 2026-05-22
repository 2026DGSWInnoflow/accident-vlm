from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from accident_vlm.schemas.preprocessing import SelectedFrame


def _line_angle(line: np.ndarray) -> float:
    x1, y1, x2, y2 = line
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def analyze_road_geometry(
    selected_frames: list[SelectedFrame],
    lane_width_m: float = 3.2,
    output_dir: Path | None = None,
    lane_segmentation_model_path: Path | None = None,
) -> dict:
    lane_counts: list[int] = []
    lane_markings: set[str] = set()
    evidence: list[str] = []
    overlays: list[dict] = []
    bev_overlays: list[dict] = []
    lane_width_pixels: list[float] = []
    vanishing_points: list[dict] = []
    homography_matrices: list[list[list[float]]] = []
    road_marking_candidates: list[dict] = []
    failure_reasons: list[str] = []
    segmentation_lane_counts: list[int] = []
    segmentation_overlays: list[dict] = []

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for frame in selected_frames[:20]:
        if not frame.path:
            continue
        image = cv2.imread(frame.path)
        if image is None:
            failure_reasons.append(f"{frame.id}: image_read_failed")
            continue
        segmentation = _run_lane_segmentation_model(
            image=image,
            frame_id=frame.id,
            model_path=lane_segmentation_model_path,
            output_dir=output_dir,
        )
        if segmentation:
            segmentation_lane_counts.append(segmentation["lane_count"])
            segmentation_overlays.extend(segmentation["overlays"])
        road_marking_candidates.extend(_detect_road_markings(image, frame.id))
        height, width = image.shape[:2]
        roi = image[int(height * 0.45) :, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=max(40, width // 12),
            maxLineGap=30,
        )
        if lines is None:
            failure_reasons.append(f"{frame.id}: hough_lines_not_found")
            continue
        angled_lines = [line[0] for line in lines if 20 < abs(_line_angle(line[0])) < 160]
        if not angled_lines:
            failure_reasons.append(f"{frame.id}: lane_angle_candidates_not_found")
            continue
        overlay = image.copy()
        mask = np.zeros_like(image)
        for line in angled_lines:
            x1, y1, x2, y2 = line
            y_offset = int(height * 0.45)
            cv2.line(overlay, (x1, y1 + y_offset), (x2, y2 + y_offset), (0, 255, 255), 2)
            cv2.line(mask, (x1, y1 + y_offset), (x2, y2 + y_offset), (255, 255, 255), 2)

        bottom_intersections = []
        for line in angled_lines:
            x1, y1, x2, y2 = line
            if y2 == y1:
                continue
            slope = (x2 - x1) / (y2 - y1)
            x_at_bottom = x1 + slope * ((height - int(height * 0.45)) - y1)
            if 0 <= x_at_bottom <= width:
                bottom_intersections.append(x_at_bottom)
        vp = _estimate_vanishing_point(angled_lines, y_offset=int(height * 0.45))
        if vp is not None:
            vanishing_points.append(vp)
        if len(bottom_intersections) >= 2:
            unique_positions = sorted(bottom_intersections)
            lane_boundary_count = 1
            previous = unique_positions[0]
            for position in unique_positions[1:]:
                if abs(position - previous) > width * 0.08:
                    lane_boundary_count += 1
                    previous = position
            lane_counts.append(max(1, lane_boundary_count - 1))
            gaps = [
                current - previous
                for previous, current in zip(unique_positions, unique_positions[1:], strict=False)
                if current - previous > width * 0.08
            ]
            if gaps:
                lane_width_pixels.append(float(np.median(gaps)))
            lane_markings.add("차선검출")
            evidence.append(frame.id)
            if output_dir:
                overlay_path = output_dir / f"{frame.id}_lane_overlay.jpg"
                mask_path = output_dir / f"{frame.id}_lane_mask.jpg"
                cv2.imwrite(str(overlay_path), overlay)
                cv2.imwrite(str(mask_path), mask)
                overlays.append(
                    {
                        "id": f"overlay_{frame.id}_lane",
                        "path": str(overlay_path),
                        "mask_path": str(mask_path),
                        "frame_id": frame.id,
                        "purpose": "lane_segmentation_overlay",
                    }
                )
                homography_record = _build_bev_overlay(
                    image=image,
                    frame_id=frame.id,
                    lane_positions=unique_positions,
                    vanishing_point=vp,
                    output_dir=output_dir,
                )
                if homography_record:
                    bev_overlays.append(homography_record["overlay"])
                    homography_matrices.append(homography_record["matrix"])

    if lane_counts:
        visible_lane_count = int(round(float(np.median(lane_counts))))
        confidence = "medium" if len(lane_counts) >= 3 else "low"
    else:
        visible_lane_count = None
        confidence = "unknown"

    pixels_per_meter = None
    if lane_width_pixels:
        pixels_per_meter = round(float(np.median(lane_width_pixels)) / lane_width_m, 3)
    matrix = _median_matrix(homography_matrices)
    bev_confidence_score = _bev_confidence_score(
        lane_count_samples=len(lane_counts),
        has_matrix=matrix is not None,
        lane_width_pixels=lane_width_pixels,
    )
    marking_types = sorted({item["type"] for item in road_marking_candidates})
    segmentation_lane_count = (
        int(round(float(np.median(segmentation_lane_counts)))) if segmentation_lane_counts else None
    )
    final_lane_count = _vote_lane_count(visible_lane_count, segmentation_lane_count)

    return {
        "visible_lane_count": {
            "value": final_lane_count if final_lane_count is not None else "확인불가",
            "confidence": confidence,
            "source": "lane_hough",
            "evidence": evidence,
        },
        "ego_lane": {
            "value": "확인불가",
            "confidence": "unknown",
            "reason": "자차 차로는 카메라 장착 위치와 차선 가림에 따라 별도 BEV 검증 필요",
        },
        "lane_markings": sorted({*lane_markings, *marking_types}) or ["확인불가"],
        "road_marking_candidates": road_marking_candidates,
        "lane_detection_vote": {
            "opencv_hough_lane_count": visible_lane_count if visible_lane_count is not None else None,
            "segmentation_lane_count": segmentation_lane_count,
            "final_lane_count": final_lane_count if final_lane_count is not None else "확인불가",
            "voting_method": "opencv_hough_primary_segmentation_optional",
            "confidence": confidence,
        },
        "lane_segmentation": {
            "method": "canny_hough_lane_mask",
            "backend": "opencv_canny_hough",
            "segmentation_backend_available": bool(segmentation_overlays),
            "model_path": str(lane_segmentation_model_path) if lane_segmentation_model_path else None,
            "overlays": overlays,
            "segmentation_overlays": segmentation_overlays,
            "confidence": confidence,
            "failure_reasons": failure_reasons,
        },
        "bev": {
            "available": bool(bev_overlays),
            "method": "lane_trapezoid_warp",
            "overlays": bev_overlays,
            "confidence": confidence if bev_overlays else "unknown",
            "confidence_score": bev_confidence_score,
            "failure_reasons": [] if bev_overlays else failure_reasons or ["lane_geometry_not_available"],
        },
        "homography": {
            "available": bool(lane_counts),
            "method": "lane_boundary_hough_estimate" if lane_counts else "not_available",
            "assumptions": [f"차선 폭 {lane_width_m}m", "도로 평면"],
            "lane_width_prior_m": lane_width_m,
            "confidence": confidence,
            "pixels_per_meter": pixels_per_meter,
            "vanishing_point": _median_point(vanishing_points),
            "matrix": matrix,
            "failure_reasons": [] if lane_counts else failure_reasons or ["lane_boundaries_not_available"],
        },
    }


def _build_bev_overlay(
    image: np.ndarray,
    frame_id: str,
    lane_positions: list[float],
    vanishing_point: dict | None,
    output_dir: Path,
) -> dict | None:
    if len(lane_positions) < 2 or not vanishing_point:
        return None
    height, width = image.shape[:2]
    left_bottom = float(lane_positions[0])
    right_bottom = float(lane_positions[-1])
    lane_width = max(20.0, right_bottom - left_bottom)
    top_y = max(0.0, min(float(height) * 0.65, float(vanishing_point["y"]) + height * 0.08))
    top_half = max(10.0, lane_width * 0.18)
    top_center = max(0.0, min(float(width), float(vanishing_point["x"])))
    src = np.float32(
        [
            [max(0.0, top_center - top_half), top_y],
            [min(float(width - 1), top_center + top_half), top_y],
            [min(float(width - 1), right_bottom), float(height - 1)],
            [max(0.0, left_bottom), float(height - 1)],
        ]
    )
    dst = np.float32(
        [
            [width * 0.35, 0],
            [width * 0.65, 0],
            [width * 0.65, height - 1],
            [width * 0.35, height - 1],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    bev_path = output_dir / f"{frame_id}_bev_overlay.jpg"
    cv2.imwrite(str(bev_path), warped)
    return {
        "matrix": [[round(float(value), 6) for value in row] for row in matrix.tolist()],
        "overlay": {
            "id": f"overlay_{frame_id}_bev",
            "path": str(bev_path),
            "frame_id": frame_id,
            "purpose": "bev_overlay",
            "source_points": src.tolist(),
        },
    }


def _estimate_vanishing_point(
    lines: list[np.ndarray],
    y_offset: int,
) -> dict | None:
    candidates: list[tuple[float, float]] = []
    for first_index, first in enumerate(lines):
        for second in lines[first_index + 1 :]:
            point = _line_intersection(first, second, y_offset)
            if point is not None:
                candidates.append(point)
    if not candidates:
        return None
    return _median_point(candidates)


def _line_intersection(
    first: np.ndarray,
    second: np.ndarray,
    y_offset: int,
) -> tuple[float, float] | None:
    x1, y1, x2, y2 = [float(value) for value in first]
    x3, y3, x4, y4 = [float(value) for value in second]
    y1 += y_offset
    y2 += y_offset
    y3 += y_offset
    y4 += y_offset
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return (float(px), float(py))


def _median_point(points: list[tuple[float, float]] | list[dict]) -> dict | None:
    if not points:
        return None
    if isinstance(points[0], dict):
        return {
            "x": round(float(np.median([point["x"] for point in points])), 3),
            "y": round(float(np.median([point["y"] for point in points])), 3),
        }
    return {
        "x": round(float(np.median([point[0] for point in points])), 3),
        "y": round(float(np.median([point[1] for point in points])), 3),
    }


def _median_matrix(matrices: list[list[list[float]]]) -> list[list[float]] | None:
    if not matrices:
        return None
    array = np.array(matrices, dtype=float)
    return [[round(float(value), 6) for value in row] for row in np.median(array, axis=0)]


def _run_lane_segmentation_model(
    image: np.ndarray,
    frame_id: str,
    model_path: Path | None,
    output_dir: Path | None,
) -> dict | None:
    if model_path is None:
        return None
    if not model_path.exists():
        return None
    net = cv2.dnn.readNetFromONNX(str(model_path))
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=(512, 288), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()
    mask = _segmentation_output_to_mask(output, image.shape[1], image.shape[0])
    lane_count = _lane_count_from_mask(mask)
    overlays: list[dict] = []
    if output_dir:
        overlay = image.copy()
        overlay[mask > 0] = (0, 255, 120)
        path = output_dir / f"{frame_id}_lane_model_overlay.jpg"
        mask_path = output_dir / f"{frame_id}_lane_model_mask.jpg"
        cv2.imwrite(str(path), overlay)
        cv2.imwrite(str(mask_path), mask)
        overlays.append(
            {
                "id": f"overlay_{frame_id}_lane_model",
                "path": str(path),
                "mask_path": str(mask_path),
                "frame_id": frame_id,
                "purpose": "lane_segmentation_overlay",
                "source": "onnx_lane_segmentation",
            }
        )
    return {"lane_count": lane_count, "overlays": overlays}


def _segmentation_output_to_mask(output: np.ndarray, width: int, height: int) -> np.ndarray:
    array = np.asarray(output)
    while array.ndim > 2:
        array = array[0]
    array = cv2.resize(array.astype("float32"), (width, height), interpolation=cv2.INTER_LINEAR)
    return np.where(array >= 0.5, 255, 0).astype("uint8")


def _lane_count_from_mask(mask: np.ndarray) -> int:
    lower_half = mask[int(mask.shape[0] * 0.45) :, :]
    contours, _ = cv2.findContours(lower_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in contours:
        x, _y, w, h = cv2.boundingRect(contour)
        if w * h >= 20:
            centers.append(x + w / 2)
    return max(1, len(_cluster_line_positions([int(center) for center in centers], tolerance=12)) - 1)


def _vote_lane_count(opencv_count: int | None, segmentation_count: int | None) -> int | None:
    if opencv_count is None:
        return segmentation_count
    if segmentation_count is None:
        return opencv_count
    if abs(opencv_count - segmentation_count) <= 1:
        return int(round((opencv_count + segmentation_count) / 2))
    return opencv_count


def _detect_road_markings(image: np.ndarray, frame_id: str) -> list[dict]:
    candidates: list[dict] = []
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=max(30, width // 8),
        maxLineGap=20,
    )
    horizontal_lines = []
    vertical_center_lines = []
    if lines is not None:
        for line in lines[:, 0, :]:
            angle = abs(_line_angle(line))
            x1, y1, x2, y2 = [int(value) for value in line]
            length = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            if angle < 12 or angle > 168:
                horizontal_lines.append((x1, y1, x2, y2, length))
            if 75 <= angle <= 105 and abs(((x1 + x2) / 2) - width / 2) < width * 0.12:
                vertical_center_lines.append((x1, y1, x2, y2, length))
    if horizontal_lines:
        longest = max(horizontal_lines, key=lambda item: item[4])
        if longest[4] >= width * 0.45 and longest[1] > height * 0.55:
            candidates.append(
                {
                    "type": "stop_line",
                    "frame_id": frame_id,
                    "confidence": "medium",
                    "bbox": [min(longest[0], longest[2]), min(longest[1], longest[3]), max(longest[0], longest[2]), max(longest[1], longest[3])],
                    "source": "opencv_hough_horizontal_line",
                }
            )
        clustered_y = _cluster_line_positions([line[1] for line in horizontal_lines], tolerance=12)
        if len(clustered_y) >= 3:
            candidates.append(
                {
                    "type": "crosswalk",
                    "frame_id": frame_id,
                    "confidence": "medium",
                    "line_count": len(clustered_y),
                    "source": "opencv_hough_parallel_horizontal_lines",
                }
            )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, (15, 60, 60), (40, 255, 255))
    yellow_ratio = float(np.count_nonzero(yellow_mask[:, int(width * 0.35) : int(width * 0.65)])) / max(
        1,
        yellow_mask[:, int(width * 0.35) : int(width * 0.65)].size,
    )
    if yellow_ratio > 0.003 or vertical_center_lines:
        candidates.append(
            {
                "type": "centerline_yellow",
                "frame_id": frame_id,
                "confidence": "medium" if yellow_ratio > 0.003 else "low",
                "yellow_ratio": round(yellow_ratio, 5),
                "source": "hsv_yellow_centerline",
            }
        )
    return candidates


def _cluster_line_positions(values: list[int], tolerance: int) -> list[float]:
    clusters: list[list[int]] = []
    for value in sorted(values):
        if not clusters or abs(float(np.mean(clusters[-1])) - value) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [float(np.mean(cluster)) for cluster in clusters]


def _bev_confidence_score(
    lane_count_samples: int,
    has_matrix: bool,
    lane_width_pixels: list[float],
) -> float:
    if not has_matrix:
        return 0.0
    sample_score = min(1.0, lane_count_samples / 3)
    width_score = 1.0 if lane_width_pixels else 0.4
    return round(max(0.0, min(1.0, 0.55 * sample_score + 0.45 * width_score)), 3)
