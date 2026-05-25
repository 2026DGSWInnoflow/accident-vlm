from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from accident_vlm.modules.video_sampling import iter_capture_frames_at_indices
from accident_vlm.schemas.preprocessing import InputQuality, SelectedFrame
from accident_vlm.utils.timecode import parse_timecode


def _bucket(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    if value < low:
        return labels[0]
    if value > high:
        return labels[2]
    return labels[1]


def analyze_input_quality(
    video_path: Path,
    selected_frames: list[SelectedFrame],
    event_windows: list[dict] | None = None,
) -> InputQuality:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return InputQuality(
            blur="high",
            brightness="dark",
            night_noise="high",
            camera_shake="unknown",
            occlusion="unknown",
            analysis_reliability="low",
            visibility_conditions={"video_open_failed": True},
        )

    blur_scores: list[float] = []
    brightness_scores: list[float] = []
    noise_scores: list[float] = []
    motion_scores: list[float] = []
    compensated_motion_scores: list[float] = []
    motion_observations: list[dict] = []
    timeline: list[dict] = []
    previous_gray: np.ndarray | None = None
    previous_frame: SelectedFrame | None = None

    frames_by_index = {frame.frame_index: frame for frame in selected_frames[:30]}
    for frame_index, image in iter_capture_frames_at_indices(
        capture,
        list(frames_by_index),
    ):
        frame = frames_by_index[frame_index]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_score = float(gray.mean())
        noise_score = float(gray.std())
        glare_ratio = float((gray >= 245).mean())
        dark_ratio = float((gray <= 25).mean())
        contrast_score = float(np.percentile(gray, 95) - np.percentile(gray, 5))
        blur_scores.append(blur_score)
        brightness_scores.append(brightness_score)
        noise_scores.append(noise_score)
        motion_score = 0.0
        compensated_motion_score = 0.0
        if previous_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                previous_gray,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            motion_score = float(np.linalg.norm(flow, axis=2).mean())
            compensated_motion_score = _ego_motion_compensated_delta(previous_gray, gray)
            motion_scores.append(motion_score)
            compensated_motion_scores.append(compensated_motion_score)
            motion_observations.append(
                {
                    "value": motion_score,
                    "ego_motion_compensated_value": compensated_motion_score,
                    "time": frame.time,
                    "evidence": [
                        previous_frame.id if previous_frame else frame.id,
                        frame.id,
                    ],
                }
            )
        timeline.append(
            {
                "frame_id": frame.id,
                "time": frame.time,
                "purpose": frame.purpose,
                "blur_score": round(blur_score, 3),
                "brightness_score": round(brightness_score, 3),
                "noise_score": round(noise_score, 3),
                "motion_score": round(motion_score, 3),
                "ego_motion_compensated_motion_score": round(compensated_motion_score, 3),
                "glare_ratio": round(glare_ratio, 5),
                "dark_ratio": round(dark_ratio, 5),
                "contrast_score": round(contrast_score, 3),
                "blur_type": _blur_type(blur_score, motion_score, compensated_motion_score),
                "quality_flags": _frame_quality_flags(
                    blur_score,
                    brightness_score,
                    noise_score,
                    glare_ratio,
                    dark_ratio,
                    contrast_score,
                ),
            }
        )
        previous_gray = gray
        previous_frame = frame
    capture.release()

    if not blur_scores:
        return InputQuality(
            blur="high",
            brightness="dark",
            night_noise="high",
            camera_shake="unknown",
            occlusion="unknown",
            analysis_reliability="low",
            visibility_conditions={"no_readable_frames": True},
        )

    blur_mean = float(np.mean(blur_scores))
    brightness_mean = float(np.mean(brightness_scores))
    noise_mean = float(np.mean(noise_scores))
    motion_mean = float(np.mean(motion_scores)) if motion_scores else 0.0
    compensated_motion_mean = (
        float(np.mean(compensated_motion_scores)) if compensated_motion_scores else 0.0
    )
    peak_motion = (
        max(motion_observations, key=lambda observation: observation["value"])
        if motion_observations
        else {"value": 0.0, "time": "확인불가", "evidence": []}
    )

    blur = _bucket(blur_mean, 80.0, 300.0, ("high", "medium", "low"))
    brightness = _bucket(brightness_mean, 65.0, 205.0, ("dark", "normal", "overexposed"))
    night_noise = _bucket(noise_mean, 35.0, 75.0, ("low", "medium", "high"))
    camera_shake = _bucket(motion_mean, 0.8, 4.0, ("low", "medium", "high"))

    weak_factors = [blur == "high", brightness != "normal", night_noise == "high"]
    reliability = "low" if sum(weak_factors) >= 2 else "medium" if any(weak_factors) else "high"
    visibility_conditions = _visibility_conditions(timeline)
    return InputQuality(
        blur=blur,
        brightness=brightness,
        night_noise=night_noise,
        camera_shake=camera_shake,
        camera_shake_score={
            "value": round(float(peak_motion["value"]), 3),
            "ego_motion_compensated_value": round(
                float(peak_motion.get("ego_motion_compensated_value", 0.0)),
                3,
            ),
            "ego_motion_compensated_mean": round(compensated_motion_mean, 3),
            "time": peak_motion["time"],
            "evidence": peak_motion["evidence"],
            "method": "farneback_optical_flow_peak_with_affine_compensation",
        },
        occlusion="unknown",
        analysis_reliability=reliability,
        timeline=timeline,
        segment_quality=_segment_quality(event_windows or [], timeline),
        visibility_conditions=visibility_conditions,
    )


def _ego_motion_compensated_delta(previous_gray: np.ndarray, gray: np.ndarray) -> float:
    features = cv2.goodFeaturesToTrack(previous_gray, maxCorners=200, qualityLevel=0.01, minDistance=7)
    if features is None or len(features) < 6:
        return float(cv2.absdiff(previous_gray, gray).mean()) / 255.0
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, features, None)
    if next_points is None or status is None:
        return float(cv2.absdiff(previous_gray, gray).mean()) / 255.0
    previous_points = features[status.flatten() == 1]
    matched_next = next_points[status.flatten() == 1]
    if len(previous_points) < 6:
        return float(cv2.absdiff(previous_gray, gray).mean()) / 255.0
    matrix, _ = cv2.estimateAffinePartial2D(previous_points, matched_next, method=cv2.RANSAC)
    if matrix is None:
        return float(cv2.absdiff(previous_gray, gray).mean()) / 255.0
    warped = cv2.warpAffine(previous_gray, matrix, (gray.shape[1], gray.shape[0]))
    return float(cv2.absdiff(warped, gray).mean()) / 255.0


def _blur_type(blur_score: float, motion_score: float, compensated_motion_score: float) -> str:
    if blur_score >= 80:
        return "none"
    if motion_score >= 1.5 and compensated_motion_score >= 0.03:
        return "motion_blur_candidate"
    return "focus_blur_candidate"


def _frame_quality_flags(
    blur_score: float,
    brightness_score: float,
    noise_score: float,
    glare_ratio: float,
    dark_ratio: float,
    contrast_score: float,
) -> list[str]:
    flags: list[str] = []
    if blur_score < 80:
        flags.append("blur")
    if brightness_score < 65:
        flags.append("low_light")
    if brightness_score > 205:
        flags.append("overexposure")
    if noise_score > 75:
        flags.append("night_noise")
    if glare_ratio > 0.08:
        flags.append("glare")
    if dark_ratio > 0.45:
        flags.append("dark_occlusion_candidate")
    if contrast_score < 35:
        flags.append("low_contrast_fog_or_dirty_lens_candidate")
    return flags


def _visibility_conditions(timeline: list[dict]) -> dict:
    flags = [flag for item in timeline for flag in item.get("quality_flags", [])]
    return {
        "glare_candidate": "glare" in flags,
        "overexposure_candidate": "overexposure" in flags,
        "low_light_candidate": "low_light" in flags,
        "rain_snow_fog_candidate": "low_contrast_fog_or_dirty_lens_candidate" in flags,
        "rain_snow_fog_or_dirty_lens_candidate": "low_contrast_fog_or_dirty_lens_candidate" in flags,
        "windshield_occlusion_candidate": "dark_occlusion_candidate" in flags,
        "occlusion_candidate": "dark_occlusion_candidate" in flags,
    }


def _segment_quality(event_windows: list[dict], timeline: list[dict]) -> list[dict]:
    results: list[dict] = []
    for event in event_windows:
        window = event.get("window", {}) if isinstance(event, dict) else {}
        try:
            start = parse_timecode(str(window.get("start")))
            end = parse_timecode(str(window.get("end")))
        except ValueError:
            continue
        segment_items = [
            item
            for item in timeline
            if start <= _safe_time_seconds(item.get("time")) <= end
        ]
        if not segment_items:
            continue
        weak_count = sum(1 for item in segment_items if item.get("quality_flags"))
        reliability = (
            "low"
            if weak_count / len(segment_items) >= 0.5
            else "medium"
            if weak_count
            else "high"
        )
        results.append(
            {
                "event_id": event.get("id"),
                "window": window,
                "frame_count": len(segment_items),
                "analysis_reliability": reliability,
                "weak_frame_count": weak_count,
                "mean_blur_score": round(float(np.mean([item["blur_score"] for item in segment_items])), 3),
                "mean_brightness_score": round(
                    float(np.mean([item["brightness_score"] for item in segment_items])),
                    3,
                ),
                "mean_noise_score": round(float(np.mean([item["noise_score"] for item in segment_items])), 3),
            }
        )
    return results


def _safe_time_seconds(value: object) -> float:
    try:
        return parse_timecode(str(value))
    except ValueError:
        return -1.0
