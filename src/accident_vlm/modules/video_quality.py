from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from accident_vlm.schemas.preprocessing import InputQuality, SelectedFrame


def _bucket(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    if value < low:
        return labels[0]
    if value > high:
        return labels[2]
    return labels[1]


def analyze_input_quality(video_path: Path, selected_frames: list[SelectedFrame]) -> InputQuality:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return InputQuality(
            blur="high",
            brightness="dark",
            night_noise="high",
            camera_shake="unknown",
            occlusion="unknown",
            analysis_reliability="low",
        )

    blur_scores: list[float] = []
    brightness_scores: list[float] = []
    noise_scores: list[float] = []
    motion_scores: list[float] = []
    motion_observations: list[dict] = []
    previous_gray: np.ndarray | None = None
    previous_frame: SelectedFrame | None = None

    for frame in selected_frames[:30]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame.frame_index)
        ok, image = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        brightness_scores.append(float(gray.mean()))
        noise_scores.append(float(gray.std()))
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
            motion_scores.append(motion_score)
            motion_observations.append(
                {
                    "value": motion_score,
                    "time": frame.time,
                    "evidence": [
                        previous_frame.id if previous_frame else frame.id,
                        frame.id,
                    ],
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
        )

    blur_mean = float(np.mean(blur_scores))
    brightness_mean = float(np.mean(brightness_scores))
    noise_mean = float(np.mean(noise_scores))
    motion_mean = float(np.mean(motion_scores)) if motion_scores else 0.0
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
    return InputQuality(
        blur=blur,
        brightness=brightness,
        night_noise=night_noise,
        camera_shake=camera_shake,
        camera_shake_score={
            "value": round(float(peak_motion["value"]), 3),
            "time": peak_motion["time"],
            "evidence": peak_motion["evidence"],
            "method": "farneback_optical_flow_peak",
        },
        occlusion="unknown",
        analysis_reliability=reliability,
    )
