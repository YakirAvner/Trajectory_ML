"""
Example:
Input to terminal -> python bright_multi_object_tracker.py input.mp4 \
        --output tracked_output.mp4 --plot all_tracks.png
"""
# Detection according to trajectory

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    center: tuple[float, float]
    area: int
    mean_brightness: float
    bbox: tuple[int, int, int, int]


@dataclass
class Track:
    id: int
    kf: cv2.KalmanFilter
    created_t: float
    history_pos: list[tuple[float, float]] = field(default_factory=list)
    history_t: list[float] = field(default_factory=list)
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    last_seen_t: float = 0.0
    predicted_pos: tuple[float, float] | None = None

    def predict(self, dt: float) -> tuple[float, float]:
        self.kf.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        state = self.kf.predict()
        self.predicted_pos = (float(state[0, 0]), float(state[1, 0]))
        return self.predicted_pos

    def update(self, center: tuple[float, float], t: float) -> None:
        measurement = np.array([[center[0]], [center[1]]], dtype=np.float32)
        state = self.kf.correct(measurement)
        corrected = (float(state[0, 0]), float(state[1, 0]))
        self.history_pos.append(corrected)
        self.history_t.append(t)
        self.last_seen_t = t
        self.hits += 1
        self.misses = 0
        self.predicted_pos = corrected

    @property
    def position(self) -> tuple[float, float]:
        if self.history_pos:
            return self.history_pos[-1]
        return self.predicted_pos if self.predicted_pos is not None else (0.0, 0.0)

    @property
    def velocity(self) -> tuple[float, float]:
        state = self.kf.statePost
        return float(state[2, 0]), float(state[3, 0])


class MultiObjectTracker:
    """2D constant-velocity Kalman tracking with gated Hungarian assignment."""

    def __init__(
        self,
        dt: float,
        max_match_distance: float = 70.0,
        max_misses: int = 12,
        min_hits_to_confirm: int = 3,
        process_noise: float = 30.0,
        measurement_noise: float = 12.0,
    ) -> None:
        self.dt = dt
        self.max_match_distance = max_match_distance
        self.max_misses = max_misses
        self.min_hits_to_confirm = min_hits_to_confirm
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self._next_id = 0

    def _new_track(self, center: tuple[float, float], t: float) -> Track:
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        kf.processNoiseCov = np.diag(
            [self.process_noise, self.process_noise, self.process_noise, self.process_noise]
        ).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise
        kf.errorCovPost = np.diag([100, 100, 1000, 1000]).astype(np.float32)
        kf.statePost = np.array([[center[0]], [center[1]], [0], [0]], dtype=np.float32)

        track = Track(
            id=self._next_id,
            kf=kf,
            created_t=t,
            history_pos=[center],
            history_t=[t],
            hits=1,
            misses=0,
            confirmed=(self.min_hits_to_confirm <= 1),
            last_seen_t=t,
            predicted_pos=center,
        )
        self._next_id += 1
        return track

    def step(self, detections: list[tuple[float, float]], t: float) -> list[Track]:
        predicted = [track.predict(self.dt) for track in self.tracks]
        unmatched_track_indices = set(range(len(self.tracks)))
        unmatched_detection_indices = set(range(len(detections)))
        matches: list[tuple[int, int]] = []

        if self.tracks and detections:
            cost = np.empty((len(self.tracks), len(detections)), dtype=np.float64)
            for i, (px, py) in enumerate(predicted):
                for j, (dx, dy) in enumerate(detections):
                    cost[i, j] = np.hypot(px - dx, py - dy)

            gated_cost = cost.copy()
            gated_cost[gated_cost > self.max_match_distance] = 1e9
            rows, cols = linear_sum_assignment(gated_cost)

            for i, j in zip(rows, cols):
                if cost[i, j] <= self.max_match_distance:
                    matches.append((int(i), int(j)))
                    unmatched_track_indices.discard(int(i))
                    unmatched_detection_indices.discard(int(j))

        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            track.update(detections[detection_index], t)
            if track.hits >= self.min_hits_to_confirm:
                track.confirmed = True

        for track_index in unmatched_track_indices:
            self.tracks[track_index].misses += 1

        for detection_index in unmatched_detection_indices:
            self.tracks.append(self._new_track(detections[detection_index], t))

        still_active: list[Track] = []
        for track in self.tracks:
            if track.misses > self.max_misses:
                if track.confirmed:
                    self.lost_tracks.append(track)
            else:
                still_active.append(track)
        self.tracks = still_active

        return [track for track in self.tracks if track.confirmed]


def detect_bright_objects(
    frame: np.ndarray,
    threshold: int = 220,
    min_area: int = 8,
    max_area: int = 10000,
    morph_kernel_size: int = 3,
) -> list[Detection]:
    """Return bright connected components using grayscale thresholding."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    detections: list[Detection] = []

    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if not min_area <= area <= max_area:
            continue
        component_pixels = gray[labels == label]
        detections.append(
            Detection(
                center=(float(centroids[label, 0]), float(centroids[label, 1])),
                area=int(area),
                mean_brightness=float(component_pixels.mean()),
                bbox=(int(x), int(y), int(w), int(h)),
            )
        )
    return detections


def color_for_id(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 47) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _pt(point: tuple[float, float]) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def fit_motion_model(
    times: list[float],
    positions: list[tuple[float, float]],
    model: str = "auto",
) -> tuple[str, np.ndarray, np.ndarray, float]:
    """Fit CV or CA pixel-space models and choose the lower penalized residual in auto mode."""
    t = np.asarray(times, dtype=np.float64)
    t = t - t[-1]
    x = np.asarray([p[0] for p in positions], dtype=np.float64)
    y = np.asarray([p[1] for p in positions], dtype=np.float64)

    def fit_cv() -> tuple[np.ndarray, np.ndarray, float]:
        px = np.polyfit(t, x, 1)
        py = np.polyfit(t, y, 1)
        err = np.mean((np.polyval(px, t) - x) ** 2 + (np.polyval(py, t) - y) ** 2)
        return px, py, float(err)

    def fit_ca() -> tuple[np.ndarray, np.ndarray, float]:
        px = np.polyfit(t, x, 2)
        py = np.polyfit(t, y, 2)
        err = np.mean((np.polyval(px, t) - x) ** 2 + (np.polyval(py, t) - y) ** 2)
        return px, py, float(err)

    cv_x, cv_y, cv_error = fit_cv()
    if len(t) < 5:
        return "constant_velocity", cv_x, cv_y, cv_error

    ca_x, ca_y, ca_error = fit_ca()
    if model == "constant_velocity":
        return "constant_velocity", cv_x, cv_y, cv_error
    if model == "constant_acceleration":
        return "constant_acceleration", ca_x, ca_y, ca_error
    if model != "auto":
        raise ValueError("model must be auto, constant_velocity, or constant_acceleration")

    # The acceleration model must improve residual meaningfully to justify extra parameters.
    if ca_error < 0.80 * cv_error:
        return "constant_acceleration", ca_x, ca_y, ca_error
    return "constant_velocity", cv_x, cv_y, cv_error


def predict_future_positions(
    track: Track,
    dt: float,
    n_steps: int,
    fit_window: int = 12,
    model: str = "auto",
) -> tuple[str, list[tuple[float, float]], float]:
    if len(track.history_t) < 2:
        return "insufficient_history", [], float("nan")

    times = track.history_t[-fit_window:]
    positions = track.history_pos[-fit_window:]
    model_name, x_poly, y_poly, residual = fit_motion_model(times, positions, model=model)
    future_t = np.arange(1, n_steps + 1, dtype=np.float64) * dt
    px = np.polyval(x_poly, future_t)
    py = np.polyval(y_poly, future_t)
    return model_name, list(zip(px.tolist(), py.tolist())), residual


def plot_all_tracks(tracks: Iterable[Track], frame_size: tuple[int, int], plot_path: str) -> None:
    width, height = frame_size
    fig, ax = plt.subplots(figsize=(10, max(5, 10 * height / width)))
    plotted = 0

    for track in sorted(tracks, key=lambda tr: tr.id):
        if not track.history_pos:
            continue
        xs = [p[0] for p in track.history_pos]
        ys = [p[1] for p in track.history_pos]
        b, g, r = color_for_id(track.id)
        rgb = (r / 255.0, g / 255.0, b / 255.0)

        ax.plot(xs, ys, color=rgb, marker="o", markersize=2.5, linewidth=1.3, label=f"Object {track.id}")
        ax.scatter(xs[0], ys[0], color="lime", edgecolors="black", linewidths=0.5, s=32, zorder=3)
        ax.scatter(xs[-1], ys[-1], color="red", edgecolors="black", linewidths=0.5, s=34, zorder=3)
        ax.text(xs[-1] + 5, ys[-1] - 5, f"ID {track.id}", color=rgb, fontsize=8)
        plotted += 1

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title("All confirmed bright-object tracks")
    ax.grid(alpha=0.2)
    if plotted:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_video(
    video_path: str | Path,
    output_path: str | Path = "tracked_output.mp4",
    plot_path: str | Path = "all_tracks.png",
    brightness_threshold: int = 220,
    min_area: int = 8,
    max_area: int = 10000,
    max_match_distance: float = 70.0,
    max_misses: int = 12,
    min_hits_to_confirm: int = 3,
    trail_length: int = 30,
    predict_ahead_frames: int = 10,
    fit_window: int = 12,
    motion_model: str = "auto",
) -> list[Track]:
    cap = cv2.VideoCapture(str(video_path))
    writer: cv2.VideoWriter | None = None

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 1.0:
            fps = 30.0
        dt = 1.0 / fps

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Video reports invalid frame dimensions.")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video at {output_path}")

        tracker = MultiObjectTracker(
            dt=dt,
            max_match_distance=max_match_distance,
            max_misses=max_misses,
            min_hits_to_confirm=min_hits_to_confirm,
        )
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx * dt

            detections = detect_bright_objects(
                frame,
                threshold=brightness_threshold,
                min_area=min_area,
                max_area=max_area,
            )
            active_tracks = tracker.step([d.center for d in detections], t)

            for detection in detections:
                x, y, w, h = detection.bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)

            for track in active_tracks:
                color = color_for_id(track.id)
                trail = track.history_pos[-trail_length:]
                if len(trail) >= 2:
                    pts = np.asarray([_pt(p) for p in trail], dtype=np.int32)
                    cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)

                cx, cy = track.position
                cv2.circle(frame, _pt((cx, cy)), 6, color, -1, cv2.LINE_AA)
                vx, vy = track.velocity
                cv2.putText(
                    frame,
                    f"ID {track.id}  v=({vx:.0f},{vy:.0f}) px/s",
                    (int(cx) + 8, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    1,
                    cv2.LINE_AA,
                )

                model_name, future, residual = predict_future_positions(
                    track,
                    dt=dt,
                    n_steps=predict_ahead_frames,
                    fit_window=fit_window,
                    model=motion_model,
                )
                in_frame = [(x, y) for x, y in future if 0 <= x < width and 0 <= y < height]
                if len(in_frame) >= 2:
                    pred_pts = np.asarray([_pt(p) for p in in_frame], dtype=np.int32)
                    cv2.polylines(frame, [pred_pts], False, color, 1, cv2.LINE_AA)
                    ex, ey = in_frame[-1]
                    cv2.drawMarker(frame, _pt((ex, ey)), color, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)

                if model_name != "insufficient_history":
                    cv2.putText(
                        frame,
                        model_name.replace("constant_", ""),
                        (int(cx) + 8, int(cy) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                frame,
                f"frame={frame_idx} detections={len(detections)} confirmed={len(active_tracks)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"frame={frame_idx} detections={len(detections)} confirmed={len(active_tracks)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

            writer.write(frame)
            frame_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    combined = tracker.tracks + tracker.lost_tracks
    unique = {track.id: track for track in combined if track.confirmed and track.history_pos}
    all_tracks = sorted(unique.values(), key=lambda track: track.id)

    print(f"Processed {frame_idx} frames.")
    print(f"Tracked {len(all_tracks)} confirmed object(s).")
    for track in all_tracks:
        duration = track.history_t[-1] - track.history_t[0]
        x0, y0 = track.history_pos[0]
        x1, y1 = track.history_pos[-1]
        print(
            f"  Object {track.id}: {len(track.history_t)} detections over {duration:.2f}s, "
            f"from ({x0:.0f}, {y0:.0f}) to ({x1:.0f}, {y1:.0f})"
        )

    print(f"Annotated video saved to {output_path}")
    plot_all_tracks(all_tracks, (width, height), str(plot_path))
    print(f"Summary plot saved to {plot_path}")
    return all_tracks


def main() -> None:
    parser = argparse.ArgumentParser(description="Track and forecast multiple bright objects in a video.")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--output", default="tracked_output.mp4", help="Annotated video path")
    parser.add_argument("--plot", default="all_tracks.png", help="Track summary image path")
    parser.add_argument("--threshold", type=int, default=220, help="Brightness threshold: 0-255")
    parser.add_argument("--min-area", type=int, default=8, help="Minimum bright-component area in pixels")
    parser.add_argument("--max-area", type=int, default=10000, help="Maximum bright-component area in pixels")
    parser.add_argument("--match-distance", type=float, default=70.0, help="Maximum detection-to-track match distance in pixels")
    parser.add_argument("--max-misses", type=int, default=12, help="Frames retained without a detection")
    parser.add_argument("--min-hits", type=int, default=3, help="Matches required before a track becomes confirmed")
    parser.add_argument("--predict-frames", type=int, default=10, help="Future frames to predict")
    parser.add_argument("--fit-window", type=int, default=12, help="Recent observations used for fitting")
    parser.add_argument(
        "--motion-model",
        choices=["auto", "constant_velocity", "constant_acceleration"],
        default="auto",
        help="Pixel-space prediction model",
    )
    args = parser.parse_args()

    process_video(
        args.video,
        output_path=args.output,
        plot_path=args.plot,
        brightness_threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        max_match_distance=args.match_distance,
        max_misses=args.max_misses,
        min_hits_to_confirm=args.min_hits,
        predict_ahead_frames=args.predict_frames,
        fit_window=args.fit_window,
        motion_model=args.motion_model,
    )


if __name__ == "__main__":
    main()