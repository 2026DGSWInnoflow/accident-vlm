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
) -> dict:
    lane_counts: list[int] = []
    lane_markings: set[str] = set()
    evidence: list[str] = []
    overlays: list[dict] = []
    lane_width_pixels: list[float] = []
    vanishing_points: list[dict] = []

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for frame in selected_frames[:20]:
        if not frame.path:
            continue
        image = cv2.imread(frame.path)
        if image is None:
            continue
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
            continue
        angled_lines = [line[0] for line in lines if 20 < abs(_line_angle(line[0])) < 160]
        if not angled_lines:
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

    if lane_counts:
        visible_lane_count = int(round(float(np.median(lane_counts))))
        confidence = "medium" if len(lane_counts) >= 3 else "low"
    else:
        visible_lane_count = None
        confidence = "unknown"

    pixels_per_meter = None
    if lane_width_pixels:
        pixels_per_meter = round(float(np.median(lane_width_pixels)) / lane_width_m, 3)

    return {
        "visible_lane_count": {
            "value": visible_lane_count if visible_lane_count is not None else "확인불가",
            "confidence": confidence,
            "source": "lane_hough",
            "evidence": evidence,
        },
        "ego_lane": {
            "value": "확인불가",
            "confidence": "unknown",
            "reason": "자차 차로는 카메라 장착 위치와 차선 가림에 따라 별도 BEV 검증 필요",
        },
        "lane_markings": sorted(lane_markings) or ["확인불가"],
        "lane_segmentation": {
            "method": "canny_hough_lane_mask",
            "overlays": overlays,
            "confidence": confidence,
        },
        "homography": {
            "available": bool(lane_counts),
            "method": "lane_boundary_hough_estimate" if lane_counts else "not_available",
            "assumptions": [f"차선 폭 {lane_width_m}m", "도로 평면"],
            "confidence": confidence,
            "pixels_per_meter": pixels_per_meter,
            "vanishing_point": _median_point(vanishing_points),
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
