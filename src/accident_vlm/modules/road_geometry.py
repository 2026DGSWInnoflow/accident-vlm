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
) -> dict:
    lane_counts: list[int] = []
    lane_markings: set[str] = set()
    evidence: list[str] = []

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
        bottom_intersections = []
        for line in angled_lines:
            x1, y1, x2, y2 = line
            if y2 == y1:
                continue
            slope = (x2 - x1) / (y2 - y1)
            x_at_bottom = x1 + slope * ((height - int(height * 0.45)) - y1)
            if 0 <= x_at_bottom <= width:
                bottom_intersections.append(x_at_bottom)
        if len(bottom_intersections) >= 2:
            unique_positions = sorted(bottom_intersections)
            lane_boundary_count = 1
            previous = unique_positions[0]
            for position in unique_positions[1:]:
                if abs(position - previous) > width * 0.08:
                    lane_boundary_count += 1
                    previous = position
            lane_counts.append(max(1, lane_boundary_count - 1))
            lane_markings.add("차선검출")
            evidence.append(frame.id)

    if lane_counts:
        visible_lane_count = int(round(float(np.median(lane_counts))))
        confidence = "medium" if len(lane_counts) >= 3 else "low"
    else:
        visible_lane_count = None
        confidence = "unknown"

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
        "homography": {
            "available": bool(lane_counts),
            "method": "lane_boundary_hough_estimate" if lane_counts else "not_available",
            "assumptions": [f"차선 폭 {lane_width_m}m", "도로 평면"],
            "confidence": confidence,
        },
    }
