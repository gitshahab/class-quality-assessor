import os
from typing import Any

import cv2
import numpy as np


def analyze_video(
    video_path: str,
    sample_interval_seconds: float = 1.0,
    black_brightness_threshold: float = 15.0,
    dark_brightness_threshold: float = 45.0,
    frozen_difference_threshold: float = 0.35,
    min_frozen_seconds: float = 8.0,
) -> dict[str, Any]:
    """
    Run local, sampled-frame checks on a classroom recording.

    These are screening indicators only. They cannot determine teaching
    quality, prove a camera outage, or reliably distinguish a frozen feed
    from a genuinely static scene.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be greater than zero.")
    if min_frozen_seconds <= 0:
        raise ValueError("min_frozen_seconds must be greater than zero.")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Unable to open video. Check the file and codec support.")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or frame_count <= 0:
            raise ValueError("Video metadata is invalid or no frames are readable.")

        duration_seconds = frame_count / fps
        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        face_detector = cv2.CascadeClassifier(cascade_path)
        face_detector_available = not face_detector.empty()

        black_timestamps: list[float] = []
        dark_timestamps: list[float] = []
        brightness_values: list[float] = []
        face_values: list[bool] = []
        freeze_intervals: list[dict[str, float]] = []

        sampled_frames = 0
        previous_small: np.ndarray | None = None
        freeze_start: float | None = None
        freeze_last_time: float | None = None
        next_sample_time = 0.0
        frame_index = 0

        def close_freeze_interval() -> None:
            nonlocal freeze_start, freeze_last_time
            if freeze_start is not None and freeze_last_time is not None:
                duration = freeze_last_time - freeze_start
                if duration >= min_frozen_seconds:
                    freeze_intervals.append({
                        "start_seconds": round(max(0.0, freeze_start), 2),
                        "end_seconds": round(freeze_last_time, 2),
                        "duration_seconds": round(duration, 2),
                    })
            freeze_start = None
            freeze_last_time = None

        while True:
            success, frame = capture.read()
            if not success:
                break

            timestamp = frame_index / fps
            frame_index += 1
            if timestamp + (0.5 / fps) < next_sample_time:
                continue
            next_sample_time = timestamp + sample_interval_seconds
            sampled_frames += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            dark_pixel_ratio = float(np.count_nonzero(gray < 25) / gray.size)
            brightness_values.append(brightness)

            is_black = (
                brightness <= black_brightness_threshold
                and dark_pixel_ratio >= 0.98
            )
            if is_black:
                black_timestamps.append(timestamp)
            if brightness < dark_brightness_threshold:
                dark_timestamps.append(timestamp)

            if face_detector_available:
                detection_gray = gray
                if gray.shape[1] > 640:
                    resized_height = max(1, int(gray.shape[0] * 640 / gray.shape[1]))
                    detection_gray = cv2.resize(gray, (640, resized_height))
                detection_gray = cv2.equalizeHist(detection_gray)
                faces = face_detector.detectMultiScale(
                    detection_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                )
                face_values.append(len(faces) > 0)

            # Compare small grayscale samples. Require both very low overall
            # change and very few changed pixels; still only a possible freeze.
            if is_black:
                close_freeze_interval()
                previous_small = None
                continue

            current_small = cv2.resize(gray, (160, 90))
            if previous_small is not None:
                difference = cv2.absdiff(current_small, previous_small)
                mean_difference = float(np.mean(difference))
                changed_pixel_ratio = float(np.count_nonzero(difference > 3) / difference.size)
                near_identical = (
                    mean_difference <= frozen_difference_threshold
                    and changed_pixel_ratio <= 0.005
                )

                if near_identical:
                    if freeze_start is None:
                        freeze_start = max(0.0, timestamp - sample_interval_seconds)
                    freeze_last_time = timestamp
                else:
                    close_freeze_interval()

            previous_small = current_small

        close_freeze_interval()

        if sampled_frames == 0:
            raise ValueError("No video frames could be analyzed.")

        black_ratio = len(black_timestamps) / sampled_frames
        dark_ratio = len(dark_timestamps) / sampled_frames
        average_brightness = float(np.mean(brightness_values))
        face_ratio = (
            sum(face_values) / len(face_values) if face_values else None
        )

        return {
            "video_metadata": {
                "duration_seconds": round(duration_seconds, 2),
                "width": width,
                "height": height,
                "fps": round(fps, 2),
                "frame_count": frame_count,
                "sampled_frames": sampled_frames,
                "sample_interval_seconds": sample_interval_seconds,
            },
            "visual_checks": {
                "black_screen": {
                    "status": "flagged" if black_timestamps else "not_detected",
                    "sampled_frame_count": len(black_timestamps),
                    "timestamps_seconds": [round(t, 2) for t in black_timestamps],
                    "sampled_frame_ratio": round(black_ratio, 4),
                    "note": "Potential black-screen samples; inspect timestamps to rule out transitions.",
                },
                "lighting": {
                    "status": "potentially_dark" if dark_ratio >= 0.25 else "no_darkness_flag",
                    "average_frame_brightness": round(average_brightness, 2),
                    "dark_sampled_frame_ratio": round(dark_ratio, 4),
                    "note": "Whole-frame brightness only; it does not measure face-specific lighting.",
                },
                "frozen_frames": {
                    "status": "needs_review" if freeze_intervals else "not_detected",
                    "intervals": freeze_intervals,
                    "note": (
                        "Flags sustained near-identical samples, not confirmed freezes. "
                        "A static scene can still cause false positives."
                    ),
                },
                "face_visibility": {
                    "status": "estimated" if face_values else "not_assessed",
                    "sampled_frames_with_face": sum(face_values),
                    "sampled_frames_checked": len(face_values),
                    "face_detection_ratio": round(face_ratio, 4) if face_ratio is not None else None,
                    "note": (
                        "Haar-cascade face detection is approximate and may miss faces. "
                        "This is not proof that a teacher is absent."
                    ),
                },
            },
            "analysis_metadata": {
                "method": "Local OpenCV sampled-frame analysis",
                "external_video_upload": False,
                "face_detector_available": face_detector_available,
                "interpretation": "Indicators require human review; not a teaching-quality score.",
            },
        }
    finally:
        capture.release()
