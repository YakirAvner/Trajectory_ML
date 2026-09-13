#!/usr/bin/env python3
"""
Offline grayscale-only multi-object 2D trajectory analysis.

Designed for benign, non-sensitive test footage such as:
- balls,
- colored/reflective markers,
- lab test objects,
- sports-training targets.

Features:
- Grayscale thresholding only; no HSV processing.
- Connected blob/contour detection.
- Multiple persistent object tracks with unique track IDs.
- Constant-acceleration Kalman filtering in pixel coordinates.
- Nearest-neighbor, gated association between detections and tracks.
- One combined CSV plus a separate CSV for every individual object.
- An annotated output video.
- A saved plot of each object's filtered trajectory and short visual forecast.

Important:
- All outputs are image-space values: pixels, pixels/s, and pixels/s^2.
- The forecast is a short visual extrapolation for offline analysis only.
- It does not estimate real-world range, altitude, impact points, targeting,
  guidance, interception, geolocation, or any operational weapon use.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Point2D = Tuple[float, float]
Color = Tuple[int, int, int]


@dataclass
class Detection:
    """A connected grayscale blob detected in one frame."""

    center: Point2D
    area: float
    radius: float
    contour: np.ndarray


@dataclass
class Track:
    """
    State and history for one tracked image-space object.

    `track_id` is stable for the life of the track. If an object disappears
    for longer than `max_missed_frames`, its track is retired. A later
    reappearance is intentionally assigned a new ID.
    """

    track_id: int
    kalman: "ConstantAccelerationKalman2D"
    color_bgr: Color
    created_frame: int
    last_seen_frame: int
    missed_frames: int = 0
    age_frames: int = 0
    hit_count: int = 1
    last_detection: Optional[Detection] = None
    history: deque = field(default_factory=lambda: deque(maxlen=150))
    all_rows: List[Dict[str, object]] = field(default_factory=list)

    def position(self) -> Point2D:
        state = self.kalman.state
        return float(state[0]), float(state[1])

    def predicted_position(self) -> Optional[Point2D]:
        return self.kalman.predicted_position

    def state(self) -> np.ndarray:
        return self.kalman.state


class ConstantAccelerationKalman2D:
    """
    Kalman state in image coordinates:

        [x, y, vx, vy, ax, ay]^T

    x, y     : pixels
    vx, vy   : pixels/second
    ax, ay   : pixels/second^2
    """

    def __init__(
        self,
        fps: float,
        measurement_std_px: float = 5.0,
        acceleration_noise_std: float = 70.0,
    ) -> None:
        if fps <= 0:
            raise ValueError("FPS must be positive.")

        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.measurement_std_px = float(measurement_std_px)
        self.acceleration_noise_std = float(acceleration_noise_std)

        self.kf = cv2.KalmanFilter(6, 2, 0, cv2.CV_64F)

        self.kf.transitionMatrix = self.make_transition(self.dt)
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        self.kf.processNoiseCov = self.make_process_noise(
            self.dt,
            self.acceleration_noise_std,
        )
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float64) * (
            self.measurement_std_px ** 2
        )

        self.kf.errorCovPost = np.diag(
            [
                100.0**2,
                100.0**2,
                300.0**2,
                300.0**2,
                500.0**2,
                500.0**2,
            ]
        ).astype(np.float64)

        self.kf.statePost = np.zeros((6, 1), dtype=np.float64)
        self.kf.statePre = np.zeros((6, 1), dtype=np.float64)

        self.initialized = False
        self.last_predicted_state: Optional[np.ndarray] = None

    @staticmethod
    def make_transition(dt: float) -> np.ndarray:
        return np.array(
            [
                [1, 0, dt, 0, 0.5 * dt * dt, 0],
                [0, 1, 0, dt, 0, 0.5 * dt * dt],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def make_process_noise(
        dt: float,
        acceleration_noise_std: float,
    ) -> np.ndarray:
        """
        Conservative image-space process uncertainty.

        Raise `acceleration_noise_std` when the filtered track lags legitimate
        motion changes. Lower it when estimates fluctuate excessively.
        """
        q = acceleration_noise_std ** 2

        return q * np.array(
            [
                [dt**4 / 4, 0, dt**3 / 2, 0, dt**2 / 2, 0],
                [0, dt**4 / 4, 0, dt**3 / 2, 0, dt**2 / 2],
                [dt**3 / 2, 0, dt**2, 0, dt, 0],
                [0, dt**3 / 2, 0, dt**2, 0, dt],
                [dt**2 / 2, 0, dt, 0, 1, 0],
                [0, dt**2 / 2, 0, dt, 0, 1],
            ],
            dtype=np.float64,
        )

    def initialize(self, position: Point2D) -> np.ndarray:
        x, y = position

        self.kf.statePost = np.array(
            [[x], [y], [0.0], [0.0], [0.0], [0.0]],
            dtype=np.float64,
        )
        self.kf.statePre = self.kf.statePost.copy()
        self.last_predicted_state = self.kf.statePost.copy()
        self.initialized = True

        return self.state

    def predict(self) -> Optional[np.ndarray]:
        if not self.initialized:
            return None

        predicted = self.kf.predict()
        self.last_predicted_state = predicted.copy()
        return predicted.ravel().copy()

    def correct(self, position: Point2D) -> np.ndarray:
        if not self.initialized:
            return self.initialize(position)

        measurement = np.array(
            [[position[0]], [position[1]]],
            dtype=np.float64,
        )
        corrected = self.kf.correct(measurement)
        return corrected.ravel().copy()

    @property
    def state(self) -> np.ndarray:
        return self.kf.statePost.ravel().copy()

    @property
    def predicted_position(self) -> Optional[Point2D]:
        if self.last_predicted_state is None:
            return None

        return (
            float(self.last_predicted_state[0, 0]),
            float(self.last_predicted_state[1, 0]),
        )

    def forecast(
        self,
        horizon_seconds: float,
        step_seconds: Optional[float] = None,
    ) -> np.ndarray:
        """
        Return an Nx2 sequence of short image-space forecast positions.

        This method uses a copy of the state, so it does not modify the
        actual tracker.
        """
        if not self.initialized or horizon_seconds <= 0:
            return np.empty((0, 2), dtype=np.float64)

        if step_seconds is None:
            step_seconds = self.dt

        steps = max(1, int(math.ceil(horizon_seconds / step_seconds)))
        transition = self.make_transition(step_seconds)
        state = self.kf.statePost.copy()

        points = []
        for _ in range(steps):
            state = transition @ state
            points.append((state[0, 0], state[1, 0]))

        return np.asarray(points, dtype=np.float64)


class GrayscaleBlobDetector:
    """
    Detects blobs using grayscale thresholding only.

    Modes:
    - bright: keeps pixels with grayscale intensity >= threshold.
    - dark: keeps pixels with grayscale intensity <= threshold.

    No HSV color conversion is performed.
    """

    def __init__(
        self,
        threshold: int,
        polarity: str,
        min_area: float,
        max_area: float,
        morph_kernel_size: int,
    ) -> None:
        if not 0 <= threshold <= 255:
            raise ValueError("threshold must be within [0, 255].")

        if polarity not in {"bright", "dark"}:
            raise ValueError("polarity must be 'bright' or 'dark'.")

        if morph_kernel_size < 1 or morph_kernel_size % 2 == 0:
            raise ValueError(
                "morph_kernel_size must be a positive odd integer."
            )

        self.threshold = int(threshold)
        self.polarity = polarity
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size),
        )

    def make_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        threshold_type = (
            cv2.THRESH_BINARY
            if self.polarity == "bright"
            else cv2.THRESH_BINARY_INV
        )

        _, mask = cv2.threshold(
            gray,
            self.threshold,
            255,
            threshold_type,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel,
            iterations=1,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=2,
        )

        return mask

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        mask = self.make_mask(frame)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        detections: List[Detection] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))

            if area < self.min_area or area > self.max_area:
                continue

            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-9:
                continue

            center_x = float(moments["m10"] / moments["m00"])
            center_y = float(moments["m01"] / moments["m00"])
            _, radius = cv2.minEnclosingCircle(contour)

            detections.append(
                Detection(
                    center=(center_x, center_y),
                    area=area,
                    radius=float(radius),
                    contour=contour,
                )
            )

        return mask, detections


def generate_track_color(track_id: int) -> Color:
    """
    Deterministic bright BGR color for video overlays.

    Uses a golden-angle hue spacing, which distributes track colors reasonably
    well for a modest number of simultaneous objects.
    """
    hue = int((track_id * 137.508) % 180)
    hsv_color = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def bgr_to_rgb(color_bgr: Color) -> Tuple[float, float, float]:
    b, g, r = color_bgr
    return r / 255.0, g / 255.0, b / 255.0


def point_to_int(point: Point2D) -> Tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def within_frame(
    point: Point2D,
    width: int,
    height: int,
    margin: int = 0,
) -> bool:
    x, y = point
    return -margin <= x < width + margin and -margin <= y < height + margin


def euclidean_distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def associate_tracks_and_detections(
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    max_distance_px: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Associate each existing track with at most one current detection.

    A simple globally sorted nearest-neighbor assignment is used:
    1. Calculate all valid track-to-detection distances.
    2. Sort by smallest distance.
    3. Accept a pair only if neither element was already assigned.

    This is lightweight and works well when objects are visually distinct
    and do not cross tightly. For dense crossing targets, replace this
    method with Hungarian assignment plus appearance features.
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    candidate_pairs: List[Tuple[float, int, int]] = []

    for track_index, track in enumerate(tracks):
        predicted = track.predicted_position()
        if predicted is None:
            predicted = track.position()

        for detection_index, detection in enumerate(detections):
            distance = euclidean_distance(predicted, detection.center)

            if distance <= max_distance_px:
                candidate_pairs.append(
                    (distance, track_index, detection_index)
                )

    candidate_pairs.sort(key=lambda pair: pair[0])

    assigned_track_indices = set()
    assigned_detection_indices = set()
    matches: List[Tuple[int, int]] = []

    for _, track_index, detection_index in candidate_pairs:
        if track_index in assigned_track_indices:
            continue

        if detection_index in assigned_detection_indices:
            continue

        matches.append((track_index, detection_index))
        assigned_track_indices.add(track_index)
        assigned_detection_indices.add(detection_index)

    unmatched_tracks = [
        index
        for index in range(len(tracks))
        if index not in assigned_track_indices
    ]
    unmatched_detections = [
        index
        for index in range(len(detections))
        if index not in assigned_detection_indices
    ]

    return matches, unmatched_tracks, unmatched_detections


def draw_polyline(
    image: np.ndarray,
    points: Sequence[Point2D],
    color: Color,
    thickness: int = 2,
) -> None:
    if len(points) < 2:
        return

    points_array = np.array(
        [point_to_int(point) for point in points],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cv2.polylines(
        image,
        [points_array],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def draw_track(
    frame: np.ndarray,
    track: Track,
    width: int,
    height: int,
    forecast_seconds: float,
) -> None:
    """
    Draw one active track:
    - colored filtered history,
    - current filtered center,
    - current raw detection if present,
    - dashed short forecast.
    """
    color = track.color_bgr

    draw_polyline(frame, list(track.history), color, thickness=2)

    state = track.state()
    current_position = (float(state[0]), float(state[1]))

    if within_frame(current_position, width, height, margin=10):
        cv2.circle(
            frame,
            point_to_int(current_position),
            6,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"ID {track.track_id}",
            (
                int(round(current_position[0])) + 8,
                int(round(current_position[1])) - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if track.last_detection is not None:
        raw_position = track.last_detection.center
        cv2.drawMarker(
            frame,
            point_to_int(raw_position),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=1,
            line_type=cv2.LINE_AA,
        )

    if track.missed_frames == 0:
        forecast = track.kalman.forecast(forecast_seconds)

        for index in range(1, len(forecast)):
            if index % 2 != 0:
                continue

            p0 = (float(forecast[index - 1, 0]), float(forecast[index - 1, 1]))
            p1 = (float(forecast[index, 0]), float(forecast[index, 1]))

            if within_frame(p0, width, height, margin=5) and within_frame(
                p1,
                width,
                height,
                margin=5,
            ):
                cv2.line(
                    frame,
                    point_to_int(p0),
                    point_to_int(p1),
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )


def draw_hud(
    frame: np.ndarray,
    frame_index: int,
    time_s: float,
    active_track_count: int,
    detection_count: int,
) -> None:
    cv2.rectangle(frame, (10, 10), (495, 94), (0, 0, 0), thickness=-1)

    lines = [
        f"Frame: {frame_index}   Time: {time_s:.3f} s",
        f"Active tracks: {active_track_count}   Detections: {detection_count}",
        "Colored trails=filtered paths  Yellow dashed=short forecast",
    ]

    for line_index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (20, 33 + line_index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def create_csv_row(
    track: Track,
    frame_index: int,
    time_s: float,
    detected: bool,
    detection: Optional[Detection],
    candidate_count: int,
) -> Dict[str, object]:
    state = track.state()
    x, y, vx, vy, ax, ay = (float(value) for value in state)
    speed = math.hypot(vx, vy)

    if detection is None:
        raw_x = ""
        raw_y = ""
        blob_area = ""
        blob_radius = ""
    else:
        raw_x = detection.center[0]
        raw_y = detection.center[1]
        blob_area = detection.area
        blob_radius = detection.radius

    return {
        "track_id": track.track_id,
        "frame": frame_index,
        "time_s": f"{time_s:.6f}",
        "detected": int(detected),
        "raw_x_px": raw_x,
        "raw_y_px": raw_y,
        "blob_area_px2": blob_area,
        "blob_radius_px": blob_radius,
        "filtered_x_px": x,
        "filtered_y_px": y,
        "vx_px_s": vx,
        "vy_px_s": vy,
        "ax_px_s2": ax,
        "ay_px_s2": ay,
        "speed_px_s": speed,
        "missed_frames": track.missed_frames,
        "track_age_frames": track.age_frames,
        "track_hits": track.hit_count,
        "candidate_count_in_frame": candidate_count,
    }


CSV_FIELDNAMES = [
    "track_id",
    "frame",
    "time_s",
    "detected",
    "raw_x_px",
    "raw_y_px",
    "blob_area_px2",
    "blob_radius_px",
    "filtered_x_px",
    "filtered_y_px",
    "vx_px_s",
    "vy_px_s",
    "ax_px_s2",
    "ay_px_s2",
    "speed_px_s",
    "missed_frames",
    "track_age_frames",
    "track_hits",
    "candidate_count_in_frame",
]


def write_per_track_csv_files(
    csv_dir: Path,
    all_tracks: Sequence[Track],
) -> List[Path]:
    """
    Write a separate CSV file for each track ID.

    Every file contains only rows belonging to that one tracked object.
    """
    output_paths: List[Path] = []

    for track in sorted(all_tracks, key=lambda item: item.track_id):
        output_path = csv_dir / f"track_{track.track_id:03d}.csv"

        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(track.all_rows)

        output_paths.append(output_path)

    return output_paths


def make_trajectory_plot(
    plot_path: str,
    all_tracks: Sequence[Track],
    forecast_seconds: float,
    min_track_hits: int,
    width: int,
    height: int,
) -> None:
    """
    Create a top-down-like image-coordinate plot.

    The y axis is inverted so its orientation matches the input image:
    origin near top-left, positive y downward.
    """
    valid_tracks = [
        track
        for track in all_tracks
        if track.hit_count >= min_track_hits and len(track.all_rows) >= 2
    ]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=160)
    ax.set_title("2D Image-Space Object Trajectories and Short Forecasts")
    ax.set_xlabel("x position (pixels)")
    ax.set_ylabel("y position (pixels)")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)

    if not valid_tracks:
        ax.text(
            0.5,
            0.5,
            "No qualifying tracks to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
    else:
        for track in valid_tracks:
            rows = track.all_rows

            x_values = []
            y_values = []
            observed_x = []
            observed_y = []

            for row in rows:
                x = row["filtered_x_px"]
                y = row["filtered_y_px"]

                if x == "" or y == "":
                    continue

                x_values.append(float(x))
                y_values.append(float(y))

                if int(row["detected"]) == 1:
                    observed_x.append(float(x))
                    observed_y.append(float(y))

            if len(x_values) < 2:
                continue

            color = bgr_to_rgb(track.color_bgr)

            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=2.2,
                label=f"Object {track.track_id} filtered path",
            )

            if observed_x:
                ax.scatter(
                    observed_x,
                    observed_y,
                    color=color,
                    s=12,
                    alpha=0.65,
                    zorder=3,
                )

            final_x = x_values[-1]
            final_y = y_values[-1]

            ax.scatter(
                [final_x],
                [final_y],
                color=color,
                marker="o",
                s=60,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
            )

            forecast = track.kalman.forecast(forecast_seconds)

            if len(forecast) > 0:
                forecast_x = [final_x] + [float(point[0]) for point in forecast]
                forecast_y = [final_y] + [float(point[1]) for point in forecast]

                ax.plot(
                    forecast_x,
                    forecast_y,
                    color=color,
                    linestyle="--",
                    linewidth=2.0,
                    alpha=0.85,
                    label=f"Object {track.track_id} short forecast",
                )

                ax.scatter(
                    [forecast_x[-1]],
                    [forecast_y[-1]],
                    color=color,
                    marker="x",
                    s=55,
                    linewidths=2.0,
                    zorder=5,
                )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            fontsize=8,
        )

    fig.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Grayscale-only multi-object 2D trajectory analysis for benign "
            "offline test videos."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input video.",
    )
    parser.add_argument(
        "--output",
        default="tracked_output.mp4",
        help="Annotated output video path. Default: tracked_output.mp4",
    )
    parser.add_argument(
        "--csv-dir",
        default="track_csv",
        help=(
            "Directory for combined CSV and one individual CSV per object. "
            "Default: track_csv"
        ),
    )
    parser.add_argument(
        "--plot",
        default="predicted_trajectories.png",
        help="Trajectory-plot output path. Default: predicted_trajectories.png",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=220,
        help=(
            "Grayscale threshold, 0-255. Bright mode keeps values >= threshold; "
            "dark mode keeps values <= threshold. Default: 220"
        ),
    )
    parser.add_argument(
        "--polarity",
        choices=("bright", "dark"),
        default="bright",
        help="Whether the test object is bright or dark relative to background.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=12.0,
        help="Minimum contour area in pixels. Default: 12",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=100000.0,
        help="Maximum contour area in pixels. Default: 100000",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=5,
        help="Odd morphology-kernel size. Default: 5",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=10,
        help="Maximum simultaneously active object tracks. Default: 10",
    )
    parser.add_argument(
        "--association-distance",
        type=float,
        default=100.0,
        help=(
            "Maximum distance in pixels between predicted track position and "
            "detection for association. Default: 100"
        ),
    )
    parser.add_argument(
        "--max-missed-frames",
        type=int,
        default=15,
        help=(
            "Retire a track after this many consecutive missing frames. "
            "Default: 15"
        ),
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=150,
        help="Maximum filtered points shown as a video trail. Default: 150",
    )
    parser.add_argument(
        "--measurement-std",
        type=float,
        default=5.0,
        help="Estimated grayscale-centroid measurement noise in pixels. Default: 5",
    )
    parser.add_argument(
        "--acceleration-noise",
        type=float,
        default=70.0,
        help=(
            "Process uncertainty in the motion model. Increase for faster "
            "motion changes; decrease for smoother tracks. Default: 70"
        ),
    )
    parser.add_argument(
        "--forecast-seconds",
        type=float,
        default=0.40,
        help="Short visual forecast duration in seconds. Default: 0.40",
    )
    parser.add_argument(
        "--min-track-hits",
        type=int,
        default=5,
        help=(
            "Minimum accepted detections needed before a track appears in the "
            "final plot. Default: 5"
        ),
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display live tracking output; press q or ESC to stop.",
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Display the grayscale segmentation mask in preview mode.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional maximum number of frames to process; 0 means all.",
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input video does not exist: {args.input}")

    if not 0 <= args.threshold <= 255:
        raise ValueError("--threshold must be in [0, 255].")

    if args.min_area <= 0:
        raise ValueError("--min-area must be positive.")

    if args.max_area <= args.min_area:
        raise ValueError("--max-area must be greater than --min-area.")

    if args.morph_kernel < 1 or args.morph_kernel % 2 == 0:
        raise ValueError("--morph-kernel must be an odd positive integer.")

    if args.max_objects < 1:
        raise ValueError("--max-objects must be at least 1.")

    if args.association_distance <= 0:
        raise ValueError("--association-distance must be positive.")

    if args.max_missed_frames < 0:
        raise ValueError("--max-missed-frames cannot be negative.")

    if args.history_length < 2:
        raise ValueError("--history-length must be at least 2.")

    if args.measurement_std <= 0:
        raise ValueError("--measurement-std must be positive.")

    if args.acceleration_noise <= 0:
        raise ValueError("--acceleration-noise must be positive.")

    if args.forecast_seconds < 0:
        raise ValueError("--forecast-seconds cannot be negative.")

    if args.min_track_hits < 1:
        raise ValueError("--min-track-hits must be at least 1.")

    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative.")


def create_track(
    track_id: int,
    detection: Detection,
    fps: float,
    frame_index: int,
    args: argparse.Namespace,
) -> Track:
    kalman = ConstantAccelerationKalman2D(
        fps=fps,
        measurement_std_px=args.measurement_std,
        acceleration_noise_std=args.acceleration_noise,
    )
    kalman.initialize(detection.center)

    track = Track(
        track_id=track_id,
        kalman=kalman,
        color_bgr=generate_track_color(track_id),
        created_frame=frame_index,
        last_seen_frame=frame_index,
        last_detection=detection,
        history=deque(maxlen=args.history_length),
    )
    track.history.append(detection.center)

    return track


def run(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    csv_dir = Path(args.csv_dir)
    plot_path = Path(args.plot)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-6:
        fps = 30.0
        print("Warning: video FPS unavailable; using 30 FPS.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read valid input-video dimensions.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not create output video: {output_path}. "
            "Try an .avi filename if MP4 encoding is unavailable."
        )

    detector = GrayscaleBlobDetector(
        threshold=args.threshold,
        polarity=args.polarity,
        min_area=args.min_area,
        max_area=args.max_area,
        morph_kernel_size=args.morph_kernel,
    )

    combined_csv_path = csv_dir / "all_tracks.csv"
    active_tracks: List[Track] = []
    completed_tracks: List[Track] = []

    next_track_id = 1
    frame_index = 0
    total_detections = 0

    try:
        with combined_csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as combined_file:
            combined_writer = csv.DictWriter(
                combined_file,
                fieldnames=CSV_FIELDNAMES,
            )
            combined_writer.writeheader()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if args.max_frames > 0 and frame_index >= args.max_frames:
                    break

                time_s = frame_index / fps
                mask, detections = detector.detect(frame)

                for track in active_tracks:
                    track.kalman.predict()
                    track.age_frames += 1
                    track.last_detection = None

                matches, unmatched_track_indices, unmatched_detection_indices = (
                    associate_tracks_and_detections(
                        active_tracks,
                        detections,
                        args.association_distance,
                    )
                )

                matched_track_indices = set()

                for track_index, detection_index in matches:
                    track = active_tracks[track_index]
                    detection = detections[detection_index]

                    track.kalman.correct(detection.center)
                    track.last_detection = detection
                    track.last_seen_frame = frame_index
                    track.missed_frames = 0
                    track.hit_count += 1
                    track.history.append(track.position())
                    matched_track_indices.add(track_index)
                    total_detections += 1

                    row = create_csv_row(
                        track=track,
                        frame_index=frame_index,
                        time_s=time_s,
                        detected=True,
                        detection=detection,
                        candidate_count=len(detections),
                    )
                    track.all_rows.append(row)
                    combined_writer.writerow(row)

                for track_index in unmatched_track_indices:
                    track = active_tracks[track_index]
                    track.missed_frames += 1
                    track.history.append(track.position())

                    row = create_csv_row(
                        track=track,
                        frame_index=frame_index,
                        time_s=time_s,
                        detected=False,
                        detection=None,
                        candidate_count=len(detections),
                    )
                    track.all_rows.append(row)
                    combined_writer.writerow(row)

                available_slots = args.max_objects - len(active_tracks)

                for detection_index in unmatched_detection_indices:
                    if available_slots <= 0:
                        break

                    detection = detections[detection_index]
                    track = create_track(
                        track_id=next_track_id,
                        detection=detection,
                        fps=fps,
                        frame_index=frame_index,
                        args=args,
                    )
                    next_track_id += 1
                    active_tracks.append(track)
                    available_slots -= 1
                    total_detections += 1

                    row = create_csv_row(
                        track=track,
                        frame_index=frame_index,
                        time_s=time_s,
                        detected=True,
                        detection=detection,
                        candidate_count=len(detections),
                    )
                    track.all_rows.append(row)
                    combined_writer.writerow(row)

                still_active_tracks: List[Track] = []

                for track in active_tracks:
                    if track.missed_frames > args.max_missed_frames:
                        completed_tracks.append(track)
                    else:
                        still_active_tracks.append(track)

                active_tracks = still_active_tracks

                annotated = frame.copy()

                for detection in detections:
                    center = point_to_int(detection.center)
                    radius = max(2, int(round(detection.radius)))

                    cv2.circle(
                        annotated,
                        center,
                        radius,
                        (110, 110, 110),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

                for track in active_tracks:
                    draw_track(
                        annotated,
                        track,
                        width,
                        height,
                        args.forecast_seconds,
                    )

                draw_hud(
                    annotated,
                    frame_index=frame_index,
                    time_s=time_s,
                    active_track_count=len(active_tracks),
                    detection_count=len(detections),
                )

                writer.write(annotated)

                if args.preview:
                    cv2.imshow(
                        "Grayscale Multi-Object 2D Tracking",
                        annotated,
                    )

                    if args.show_mask:
                        cv2.imshow("Grayscale Segmentation Mask", mask)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        print("Stopped early by user.")
                        break

                frame_index += 1

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    all_tracks = completed_tracks + active_tracks
    all_tracks.sort(key=lambda track: track.track_id)

    individual_csv_paths = write_per_track_csv_files(csv_dir, all_tracks)

    make_trajectory_plot(
        plot_path=str(plot_path),
        all_tracks=all_tracks,
        forecast_seconds=args.forecast_seconds,
        min_track_hits=args.min_track_hits,
        width=width,
        height=height,
    )

    qualifying_tracks = [
        track
        for track in all_tracks
        if track.hit_count >= args.min_track_hits
    ]

    print()
    print("Processing complete")
    print(f"Processed frames: {frame_index}")
    print(f"Accepted detections: {total_detections}")
    print(f"Total created tracks: {len(all_tracks)}")
    print(f"Tracks plotted: {len(qualifying_tracks)}")
    print(f"Annotated video: {output_path}")
    print(f"Combined CSV: {combined_csv_path}")
    print(f"Individual object CSV files: {len(individual_csv_paths)}")
    print(f"Trajectory plot: {plot_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_arguments(args)
        run(args)
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()