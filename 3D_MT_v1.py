from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

Point2D = Tuple[float, float]


@dataclass
class TrajectoryResult:
    object_id: str
    label: str
    score: float
    confidence: str
    samples: int
    path_length: float
    mean_speed: float
    speed_cv: float
    mean_acceleration: float
    heading_change_deg: float
    straightness: float
    reasons: List[str]


class TrajectoryMissileClassifier:
    def __init__(
        self,
        min_points: int = 6,
        min_path_length: float = 30.0,
        min_mean_speed: float = 8.0,
        max_speed_cv: float = 0.35,
        max_heading_change_deg: float = 25.0,
        min_straightness: float = 0.90,
        positive_score_threshold: float = 0.72,
    ) -> None:
        self.min_points = min_points
        self.min_path_length = min_path_length
        self.min_mean_speed = min_mean_speed
        self.max_speed_cv = max_speed_cv
        self.max_heading_change_deg = max_heading_change_deg
        self.min_straightness = min_straightness
        self.positive_score_threshold = positive_score_threshold

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _as_points(coordinates: Sequence[Sequence[float]]) -> np.ndarray:
        points = np.asarray(coordinates, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("coordinates must have shape (N, 2), e.g. [[x1, y1], [x2, y2], ...]")
        if not np.isfinite(points).all():
            raise ValueError("coordinates must contain only finite numeric values")
        return points

    def classify_one(
        self,
        object_id: str,
        coordinates: Sequence[Sequence[float]],
        dt: float = 1.0,
    ) -> TrajectoryResult:
        if dt <= 0:
            raise ValueError("dt must be greater than zero")

        p = self._as_points(coordinates)
        n = len(p)
        reasons: List[str] = []

        if n < self.min_points:
            return TrajectoryResult(
                object_id=str(object_id), label="insufficient_data", score=0.0,
                confidence="low", samples=n, path_length=0.0, mean_speed=0.0,
                speed_cv=0.0, mean_acceleration=0.0, heading_change_deg=0.0,
                straightness=0.0,
                reasons=[f"Need at least {self.min_points} coordinate samples; received {n}."],
            )

        displacements = np.diff(p, axis=0)
        segment_lengths = np.linalg.norm(displacements, axis=1)
        nonzero = segment_lengths > 1e-9

        if nonzero.sum() < 2:
            return TrajectoryResult(
                object_id=str(object_id), label="not_missile_like", score=0.0,
                confidence="high", samples=n, path_length=float(segment_lengths.sum()),
                mean_speed=0.0, speed_cv=0.0, mean_acceleration=0.0,
                heading_change_deg=0.0, straightness=0.0,
                reasons=["Trajectory has too little movement to evaluate sustained motion."],
            )

        speeds = segment_lengths / dt
        valid_vectors = displacements[nonzero]
        valid_lengths = segment_lengths[nonzero]
        unit_vectors = valid_vectors / valid_lengths[:, None]

        dot_products = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)
        turn_angles_deg = np.degrees(np.arccos(np.clip(dot_products, -1.0, 1.0)))

        path_length = float(segment_lengths.sum())
        net_displacement = float(np.linalg.norm(p[-1] - p[0]))
        straightness = net_displacement / path_length if path_length > 1e-9 else 0.0
        mean_speed = float(np.mean(speeds))
        speed_cv = float(np.std(speeds) / (mean_speed + 1e-9))
        mean_acceleration = float(np.mean(np.abs(np.diff(speeds) / dt))) if len(speeds) > 1 else 0.0
        heading_change_deg = float(np.mean(turn_angles_deg)) if len(turn_angles_deg) else 0.0

        length_score = self._clamp01(path_length / self.min_path_length)
        speed_score = self._clamp01(mean_speed / self.min_mean_speed)
        stability_score = self._clamp01(1.0 - speed_cv / self.max_speed_cv)
        heading_score = self._clamp01(1.0 - heading_change_deg / self.max_heading_change_deg)
        straightness_score = self._clamp01(
            (straightness - (self.min_straightness - 0.15)) / 0.15
        )

        score = float(
            0.15 * length_score
            + 0.25 * speed_score
            + 0.20 * stability_score
            + 0.20 * heading_score
            + 0.20 * straightness_score
        )

        if path_length < self.min_path_length:
            reasons.append(f"Path length {path_length:.2f} is below the configured minimum {self.min_path_length:.2f}.")
        else:
            reasons.append(f"Sustained path length: {path_length:.2f}.")

        if mean_speed < self.min_mean_speed:
            reasons.append(f"Mean speed {mean_speed:.2f} is below the configured minimum {self.min_mean_speed:.2f}.")
        else:
            reasons.append(f"Mean speed meets the configured threshold: {mean_speed:.2f}.")

        if speed_cv > self.max_speed_cv:
            reasons.append(f"Speed is variable (coefficient of variation {speed_cv:.2f}).")
        else:
            reasons.append(f"Speed is comparatively stable (coefficient of variation {speed_cv:.2f}).")

        if heading_change_deg > self.max_heading_change_deg:
            reasons.append(f"Average heading change is high: {heading_change_deg:.1f} degrees.")
        else:
            reasons.append(f"Average heading change is low: {heading_change_deg:.1f} degrees.")

        if straightness < self.min_straightness:
            reasons.append(f"Trajectory straightness {straightness:.2f} is below the configured minimum {self.min_straightness:.2f}.")
        else:
            reasons.append(f"Trajectory is straight (straightness {straightness:.2f}).")

        label = "missile_like" if score >= self.positive_score_threshold else "not_missile_like"
        decisive_features = sum([
            path_length >= self.min_path_length,
            mean_speed >= self.min_mean_speed,
            speed_cv <= self.max_speed_cv,
            heading_change_deg <= self.max_heading_change_deg,
            straightness >= self.min_straightness,
        ])
        confidence = "high" if decisive_features >= 4 and abs(score - self.positive_score_threshold) >= 0.15 else "medium"

        return TrajectoryResult(
            object_id=str(object_id), label=label, score=round(score, 4), confidence=confidence,
            samples=n, path_length=round(path_length, 4), mean_speed=round(mean_speed, 4),
            speed_cv=round(speed_cv, 4), mean_acceleration=round(mean_acceleration, 4),
            heading_change_deg=round(heading_change_deg, 4), straightness=round(straightness, 4),
            reasons=reasons,
        )

# receive by argument 
    def classify_many(
        self,
        tracks: Mapping[str, Sequence[Sequence[float]]],
        dt: float = 1.0,
    ) -> Dict[str, Dict[str, object]]:
        results: Dict[str, Dict[str, object]] = {}
        for object_id, coordinates in tracks.items():
            try:
                results[str(object_id)] = asdict(self.classify_one(str(object_id), coordinates, dt))
            except ValueError as exc:
                results[str(object_id)] = {
                    "object_id": str(object_id),
                    "label": "invalid_input",
                    "score": 0.0,
                    "confidence": "low",
                    "error": str(exc),
                }
        return results

# 10 coordinates min
if __name__ == "__main__":
    tracks = {
        "track_001": [
            [0.0, 0.0], [8.7, 1.0], [22.5, 1.8], [29.1, 3.0],
            [40.6, 4.0], [50.0, 5.2], [61.8, 6.0],
        ],
        "track_002": [
            [0.0, 0.0], [3.0, 2.0], [5.0, 7.0], [2.0, 11.0],
            [-3.0, 10.0], [-5.0, 5.0], [-2.0, 1.0],
        ],
        "track_003": [[10.0, 10.0], [10.1, 10.0], [10.0, 10.1]],
    }

    classifier = TrajectoryMissileClassifier()
    results = classifier.classify_many(tracks, dt=1.0)

    for object_id, result in results.items():
        print(f"\n{object_id}: {result['label']} | score={result['score']} | confidence={result['confidence']}")
        for reason in result.get("reasons", []):
            print(f"  - {reason}")
