"""Trajectory-only 2D motion-shape classifier.

Input: a dictionary mapping an object ID to an ordered sequence of 2D coordinates.
Output: one independent classification result per object.

This is a heuristic classifier, not a real-world identification system. A trajectory
alone cannot reliably establish whether an object is a missile: aircraft, drones,
vehicles, and tracking artifacts can share similar motion. Do not use this module
for safety-critical, targeting, or operational decisions without calibrated sensors,
validated data, and qualified human oversight.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


Point2D = Tuple[float, float]


@dataclass
class ParabolicFit:
    """Diagnostics from a rotation-invariant parabolic trajectory fit."""

    is_parabolic: bool
    score: float
    quadratic_coefficient: float
    concavity: str
    quadratic_rmse: float
    linear_rmse: float
    rmse_improvement: float
    quadratic_r2: float
    forward_span: float
    reasons: List[str]


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

    # New trajectory-shape fields
    is_parabolic: bool
    parabolicity_score: float
    quadratic_coefficient: float
    parabola_concavity: str
    quadratic_fit_rmse: float
    linear_fit_rmse: float
    quadratic_fit_r2: float

    reasons: List[str]


class TrajectoryMissileClassifier:
    """Score a sampled 2D trajectory using generic kinematic and shape features.

    Coordinates must be in a consistent metric coordinate frame (for example,
    meters in a local ENU plane), ordered from oldest to newest. `dt` is the
    interval between successive samples in seconds.

    The parabolic test is geometric rather than ballistic-model validation:
    it identifies whether the observed 2D points are well described by a
    quadratic curve after the track is rotated into its principal direction.

    It does not prove that an object is ballistic, powered, or missile-related.
    """

    def __init__(
        self,
        min_points: int = 6,
        min_path_length: float = 30.0,
        min_mean_speed: float = 8.0,
        max_speed_cv: float = 0.35,
        max_heading_change_deg: float = 25.0,
        min_straightness: float = 0.90,
        positive_score_threshold: float = 0.72,

        # Parabola-detection parameters
        min_parabola_r2: float = 0.90,
        min_parabola_improvement: float = 0.20,
        min_normalized_curvature: float = 0.01,
        max_quadratic_rmse_ratio: float = 0.08,
    ) -> None:
        self.min_points = min_points
        self.min_path_length = min_path_length
        self.min_mean_speed = min_mean_speed
        self.max_speed_cv = max_speed_cv
        self.max_heading_change_deg = max_heading_change_deg
        self.min_straightness = min_straightness
        self.positive_score_threshold = positive_score_threshold

        self.min_parabola_r2 = min_parabola_r2
        self.min_parabola_improvement = min_parabola_improvement
        self.min_normalized_curvature = min_normalized_curvature
        self.max_quadratic_rmse_ratio = max_quadratic_rmse_ratio

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _as_points(coordinates: Sequence[Sequence[float]]) -> np.ndarray:
        points = np.asarray(coordinates, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                "coordinates must have shape (N, 2), "
                "e.g. [[x1, y1], [x2, y2], ...]"
            )
        if not np.isfinite(points).all():
            raise ValueError("coordinates must contain only finite numeric values")
        return points

    @staticmethod
    def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
        return float(np.sqrt(np.mean((observed - predicted) ** 2)))

    def _fit_parabola(self, points: np.ndarray) -> ParabolicFit:
        """Fit cross-track position as a quadratic of along-track position.

        Steps:
        1. Center the trajectory.
        2. Find its principal direction with SVD/PCA.
        3. Rotate points into along-track (u) and cross-track (v) axes.
        4. Fit v = a*u^2 + b*u + c.
        5. Compare the quadratic residual to a straight-line fit.

        This avoids assuming that a parabola is aligned with the input x-axis.
        """
        n = len(points)
        reasons: List[str] = []

        if n < 4:
            return ParabolicFit(
                is_parabolic=False,
                score=0.0,
                quadratic_coefficient=0.0,
                concavity="unknown",
                quadratic_rmse=0.0,
                linear_rmse=0.0,
                rmse_improvement=0.0,
                quadratic_r2=0.0,
                forward_span=0.0,
                reasons=["At least 4 points are required for a stable quadratic fit."],
            )

        centered = points - np.mean(points, axis=0)

        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        forward_axis = vt[0]

        # Make orientation follow time order where possible.
        net_direction = points[-1] - points[0]
        if np.dot(forward_axis, net_direction) < 0:
            forward_axis = -forward_axis

        lateral_axis = np.array([-forward_axis[1], forward_axis[0]])

        u = centered @ forward_axis
        v = centered @ lateral_axis

        forward_span = float(np.ptp(u))
        lateral_span = float(np.ptp(v))

        if forward_span <= 1e-9:
            return ParabolicFit(
                is_parabolic=False,
                score=0.0,
                quadratic_coefficient=0.0,
                concavity="unknown",
                quadratic_rmse=0.0,
                linear_rmse=0.0,
                rmse_improvement=0.0,
                quadratic_r2=0.0,
                forward_span=forward_span,
                reasons=["Trajectory has insufficient forward extent for a parabolic fit."],
            )

        # Scale u before fitting to improve conditioning.
        u_center = float(np.mean(u))
        u_scale = max(float(np.std(u)), forward_span / 2.0, 1e-9)
        u_scaled = (u - u_center) / u_scale

        quadratic_coeffs = np.polyfit(u_scaled, v, deg=2)
        linear_coeffs = np.polyfit(u_scaled, v, deg=1)

        v_quadratic = np.polyval(quadratic_coeffs, u_scaled)
        v_linear = np.polyval(linear_coeffs, u_scaled)

        quadratic_rmse = self._rmse(v, v_quadratic)
        linear_rmse = self._rmse(v, v_linear)

        ss_res = float(np.sum((v - v_quadratic) ** 2))
        ss_tot = float(np.sum((v - np.mean(v)) ** 2))
        quadratic_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        rmse_improvement = (
            (linear_rmse - quadratic_rmse) / max(linear_rmse, 1e-12)
        )

        # Convert the scaled coefficient back to physical coordinate units:
        # v = a_scaled*((u-u_center)/u_scale)^2 + ...
        # Therefore a_physical = a_scaled / u_scale^2.
        quadratic_coefficient = float(quadratic_coeffs[0] / (u_scale ** 2))

        # Dimensionless curvature measure, robust across coordinate scales.
        normalized_curvature = abs(quadratic_coefficient) * forward_span

        # Quadratic residual relative to the observed cross-track span.
        # If a path is nearly straight, lateral_span is tiny, so use forward_span
        # as a stable denominator floor.
        scale_for_rmse = max(lateral_span, 0.02 * forward_span, 1e-9)
        quadratic_rmse_ratio = quadratic_rmse / scale_for_rmse

        if quadratic_coefficient > 0:
            concavity = "opens_left_of_forward_axis"
        elif quadratic_coefficient < 0:
            concavity = "opens_right_of_forward_axis"
        else:
            concavity = "flat"

        r2_score = self._clamp01(
            (quadratic_r2 - (self.min_parabola_r2 - 0.15)) / 0.15
        )
        improvement_score = self._clamp01(
            rmse_improvement / max(self.min_parabola_improvement, 1e-9)
        )
        curvature_score = self._clamp01(
            normalized_curvature / max(self.min_normalized_curvature, 1e-9)
        )
        residual_score = self._clamp01(
            1.0 - quadratic_rmse_ratio / self.max_quadratic_rmse_ratio
        )

        score = float(
            0.35 * r2_score
            + 0.30 * improvement_score
            + 0.20 * curvature_score
            + 0.15 * residual_score
        )

        is_parabolic = bool(
            quadratic_r2 >= self.min_parabola_r2
            and rmse_improvement >= self.min_parabola_improvement
            and normalized_curvature >= self.min_normalized_curvature
            and quadratic_rmse_ratio <= self.max_quadratic_rmse_ratio
        )

        if quadratic_r2 >= self.min_parabola_r2:
            reasons.append(
                f"Quadratic fit is strong (R²={quadratic_r2:.3f})."
            )
        else:
            reasons.append(
                f"Quadratic fit is weak (R²={quadratic_r2:.3f}; "
                f"minimum={self.min_parabola_r2:.3f})."
            )

        if rmse_improvement >= self.min_parabola_improvement:
            reasons.append(
                f"Quadratic fit improves RMSE by {rmse_improvement:.1%} "
                f"over a straight-line fit."
            )
        else:
            reasons.append(
                f"Quadratic fit improves RMSE by only {rmse_improvement:.1%} "
                f"over a straight-line fit."
            )

        if normalized_curvature >= self.min_normalized_curvature:
            reasons.append(
                f"Measured curvature is non-trivial "
                f"(normalized curvature={normalized_curvature:.4f})."
            )
        else:
            reasons.append(
                f"Measured curvature is too small to distinguish from a "
                f"straight path (normalized curvature={normalized_curvature:.4f})."
            )

        if is_parabolic:
            reasons.append(
                f"Trajectory is consistent with a 2D parabolic shape "
                f"({concavity})."
            )
        else:
            reasons.append(
                "Trajectory is not sufficiently consistent with a parabolic "
                "shape under the configured thresholds."
            )

        return ParabolicFit(
            is_parabolic=is_parabolic,
            score=score,
            quadratic_coefficient=quadratic_coefficient,
            concavity=concavity,
            quadratic_rmse=quadratic_rmse,
            linear_rmse=linear_rmse,
            rmse_improvement=rmse_improvement,
            quadratic_r2=quadratic_r2,
            forward_span=forward_span,
            reasons=reasons,
        )

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
                object_id=str(object_id),
                label="insufficient_data",
                score=0.0,
                confidence="low",
                samples=n,
                path_length=0.0,
                mean_speed=0.0,
                speed_cv=0.0,
                mean_acceleration=0.0,
                heading_change_deg=0.0,
                straightness=0.0,
                is_parabolic=False,
                parabolicity_score=0.0,
                quadratic_coefficient=0.0,
                parabola_concavity="unknown",
                quadratic_fit_rmse=0.0,
                linear_fit_rmse=0.0,
                quadratic_fit_r2=0.0,
                reasons=[
                    f"Need at least {self.min_points} coordinate samples; "
                    f"received {n}."
                ],
            )

        displacements = np.diff(p, axis=0)
        segment_lengths = np.linalg.norm(displacements, axis=1)
        nonzero = segment_lengths > 1e-9

        if nonzero.sum() < 2:
            return TrajectoryResult(
                object_id=str(object_id),
                label="not_missile_like",
                score=0.0,
                confidence="high",
                samples=n,
                path_length=float(segment_lengths.sum()),
                mean_speed=0.0,
                speed_cv=0.0,
                mean_acceleration=0.0,
                heading_change_deg=0.0,
                straightness=0.0,
                is_parabolic=False,
                parabolicity_score=0.0,
                quadratic_coefficient=0.0,
                parabola_concavity="unknown",
                quadratic_fit_rmse=0.0,
                linear_fit_rmse=0.0,
                quadratic_fit_r2=0.0,
                reasons=[
                    "Trajectory has too little movement to evaluate sustained motion."
                ],
            )

        speeds = segment_lengths / dt
        valid_vectors = displacements[nonzero]
        valid_lengths = segment_lengths[nonzero]
        unit_vectors = valid_vectors / valid_lengths[:, None]

        dot_products = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)
        turn_angles_deg = np.degrees(
            np.arccos(np.clip(dot_products, -1.0, 1.0))
        )

        path_length = float(segment_lengths.sum())
        net_displacement = float(np.linalg.norm(p[-1] - p[0]))
        straightness = net_displacement / path_length if path_length > 1e-9 else 0.0
        mean_speed = float(np.mean(speeds))
        speed_cv = float(np.std(speeds) / (mean_speed + 1e-9))
        mean_acceleration = (
            float(np.mean(np.abs(np.diff(speeds) / dt)))
            if len(speeds) > 1
            else 0.0
        )
        heading_change_deg = (
            float(np.mean(turn_angles_deg)) if len(turn_angles_deg) else 0.0
        )

        parabolic_fit = self._fit_parabola(p)

        length_score = self._clamp01(path_length / self.min_path_length)
        speed_score = self._clamp01(mean_speed / self.min_mean_speed)
        stability_score = self._clamp01(
            1.0 - speed_cv / self.max_speed_cv
        )
        heading_score = self._clamp01(
            1.0 - heading_change_deg / self.max_heading_change_deg
        )
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
            reasons.append(
                f"Path length {path_length:.2f} is below the configured "
                f"minimum {self.min_path_length:.2f}."
            )
        else:
            reasons.append(f"Sustained path length: {path_length:.2f}.")

        if mean_speed < self.min_mean_speed:
            reasons.append(
                f"Mean speed {mean_speed:.2f} is below the configured "
                f"minimum {self.min_mean_speed:.2f}."
            )
        else:
            reasons.append(
                f"Mean speed meets the configured threshold: {mean_speed:.2f}."
            )

        if speed_cv > self.max_speed_cv:
            reasons.append(
                f"Speed is variable (coefficient of variation {speed_cv:.2f})."
            )
        else:
            reasons.append(
                f"Speed is comparatively stable "
                f"(coefficient of variation {speed_cv:.2f})."
            )

        if heading_change_deg > self.max_heading_change_deg:
            reasons.append(
                f"Average heading change is high: {heading_change_deg:.1f} degrees."
            )
        else:
            reasons.append(
                f"Average heading change is low: {heading_change_deg:.1f} degrees."
            )

        if straightness < self.min_straightness:
            reasons.append(
                f"Trajectory straightness {straightness:.2f} is below the "
                f"configured minimum {self.min_straightness:.2f}."
            )
        else:
            reasons.append(
                f"Trajectory is straight (straightness {straightness:.2f})."
            )

        # Keep shape assessment separate from the "missile-like" heuristic.
        reasons.extend(parabolic_fit.reasons)

        if parabolic_fit.is_parabolic:
            label = "parabolic_trajectory"
        elif score >= self.positive_score_threshold:
            label = "missile_like"
        else:
            label = "not_missile_like"

        decisive_features = sum(
            [
                path_length >= self.min_path_length,
                mean_speed >= self.min_mean_speed,
                speed_cv <= self.max_speed_cv,
                heading_change_deg <= self.max_heading_change_deg,
                straightness >= self.min_straightness,
            ]
        )

        if parabolic_fit.is_parabolic:
            confidence = (
                "high"
                if parabolic_fit.score >= 0.80
                and parabolic_fit.quadratic_r2 >= 0.95
                else "medium"
            )
        else:
            confidence = (
                "high"
                if decisive_features >= 4
                and abs(score - self.positive_score_threshold) >= 0.15
                else "medium"
            )

        return TrajectoryResult(
            object_id=str(object_id),
            label=label,
            score=round(score, 4),
            confidence=confidence,
            samples=n,
            path_length=round(path_length, 4),
            mean_speed=round(mean_speed, 4),
            speed_cv=round(speed_cv, 4),
            mean_acceleration=round(mean_acceleration, 4),
            heading_change_deg=round(heading_change_deg, 4),
            straightness=round(straightness, 4),
            is_parabolic=parabolic_fit.is_parabolic,
            parabolicity_score=round(parabolic_fit.score, 4),
            quadratic_coefficient=round(
                parabolic_fit.quadratic_coefficient, 8
            ),
            parabola_concavity=parabolic_fit.concavity,
            quadratic_fit_rmse=round(parabolic_fit.quadratic_rmse, 4),
            linear_fit_rmse=round(parabolic_fit.linear_rmse, 4),
            quadratic_fit_r2=round(parabolic_fit.quadratic_r2, 4),
            reasons=reasons,
        )

    def classify_many(
        self,
        tracks: Mapping[str, Sequence[Sequence[float]]],
        dt: float = 1.0,
    ) -> Dict[str, Dict[str, object]]:
        """Independently classify each object ID and return JSON-ready dictionaries."""
        results: Dict[str, Dict[str, object]] = {}

        for object_id, coordinates in tracks.items():
            try:
                results[str(object_id)] = asdict(
                    self.classify_one(str(object_id), coordinates, dt)
                )
            except ValueError as exc:
                results[str(object_id)] = {
                    "object_id": str(object_id),
                    "label": "invalid_input",
                    "score": 0.0,
                    "confidence": "low",
                    "error": str(exc),
                }

        return results


if __name__ == "__main__":
    tracks = {
        # Nearly straight trajectory.
        "track_001": [
            [0.0, 0.0],
            [8.7, 1.0],
            [22.5, 1.8],
            [29.1, 3.0],
            [40.6, 4.0],
            [50.0, 5.2],
            [61.8, 6.0],
        ],

        # Curved, but not a clean single parabola.
        "track_002": [
            [0.0, 0.0],
            [3.0, 2.0],
            [5.0, 7.0],
            [2.0, 11.0],
            [-3.0, 10.0],
            [-5.0, 5.0],
            [-2.0, 1.0],
        ],

        # Synthetic parabola: y = -0.025*x^2 + 1.5*x + 5.
        "track_003": [
            [0.0, 5.0],
            [5.0, 11.875],
            [10.0, 17.5],
            [15.0, 21.875],
            [20.0, 25.0],
            [25.0, 26.875],
            [30.0, 27.5],
            [35.0, 26.875],
            [40.0, 25.0],
        ],

        "track_004": [
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ],
    }

    classifier = TrajectoryMissileClassifier(
        min_parabola_r2=0.90,
        min_parabola_improvement=0.20,
        min_normalized_curvature=0.01,
        max_quadratic_rmse_ratio=0.08,
    )

    results = classifier.classify_many(tracks, dt=1.0)

    for object_id, result in results.items():
        print(
            f"\n{object_id}: {result['label']} | "
            f"motion_score={result['score']} | "
            f"parabolic={result.get('is_parabolic', False)} | "
            f"parabola_score={result.get('parabolicity_score', 0.0)} | "
            f"confidence={result['confidence']}"
        )

        if "quadratic_fit_r2" in result:
            print(
                f"  quadratic R²={result['quadratic_fit_r2']} | "
                f"quadratic RMSE={result['quadratic_fit_rmse']} | "
                f"linear RMSE={result['linear_fit_rmse']} | "
                f"concavity={result['parabola_concavity']}"
            )

        for reason in result.get("reasons", []):
            print(f"  - {reason}")