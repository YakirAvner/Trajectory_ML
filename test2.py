"""
trajectory_tracker.py

A general-purpose pipeline for detecting a bright, fast-moving object
(rocket exhaust, flare, drone light, thrown ball, etc.) in camera frames,
tracking it across frames with a Kalman filter, and fitting/predicting a
ballistic (parabolic) trajectory.

This is the same three-stage pipeline (detect -> track -> predict) used in
things like ball-tracking for sports broadcasts and hobbyist rocket-launch
tracking. It contains no weapon-specific content -- no guidance, targeting,
or propulsion modeling. It just finds a bright blob in an image, follows it
frame to frame, and fits a curve.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. Detection
# ---------------------------------------------------------------------------

def detect_bright_object(frame, min_area=8, blur_ksize=9):
    """
    Find the brightest sufficiently-large blob in a single frame.

    Works well for a hot exhaust plume, a flare, or any small object that is
    much brighter than its background (e.g. against open sky).

    Parameters
    ----------
    frame : np.ndarray
        BGR image, as returned by cv2.imread / cv2.VideoCapture.
    min_area : int
        Minimum contour area (pixels) to count as a real object rather than
        sensor noise.
    blur_ksize : int
        Gaussian blur kernel size, used to suppress single-pixel noise
        before thresholding.

    Returns
    -------
    dict or None
        {'center': (x, y), 'bbox': (x, y, w, h), 'area': area, 'mask': mask}
        or None if nothing was found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Otsu's method picks a brightness threshold automatically from the
    # image's own histogram, so it adapts to different lighting conditions
    # instead of relying on one fixed cutoff.
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If Otsu comes back with an implausibly large bright region (e.g. an
    # overexposed sky), fall back to a stricter percentile-based cutoff so
    # we isolate only the very brightest point instead of half the frame.
    if np.count_nonzero(mask) > 0.05 * mask.size:
        cutoff = np.percentile(blurred, 99.5)
        _, mask = cv2.threshold(blurred, cutoff, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep the largest bright blob. In a cluttered real-world scene you'd
    # add more filters here: shape/aspect ratio, or consistency with the
    # previous frame's position so you don't jump to an unrelated bright spot.
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    M = cv2.moments(largest)
    cx = M["m10"] / M["m00"] if M["m00"] != 0 else x + w / 2
    cy = M["m01"] / M["m00"] if M["m00"] != 0 else y + h / 2

    return {"center": (cx, cy), "bbox": (x, y, w, h), "area": area, "mask": mask}


# ---------------------------------------------------------------------------
# 2. Tracking -- Kalman filter, constant-acceleration model
# ---------------------------------------------------------------------------

class TrajectoryKalmanTracker:
    """
    Tracks a single point (x, y) over time using a constant-*acceleration*
    Kalman filter. Ballistic motion speeds up/slows down under gravity, so
    this fits the physics better than a plain constant-velocity model.

    State vector: [x, y, vx, vy, ax, ay]
    """

    def __init__(self, dt=1 / 30):
        self.dt = dt
        self.kf = cv2.KalmanFilter(6, 2)

        dt2 = 0.5 * dt * dt
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)

        # How much we trust the motion model vs. each new measurement.
        # Raise measurementNoiseCov if detections are jittery; raise
        # processNoiseCov if the object maneuvers more than the model expects.
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

        self._initialized = False

    def update(self, measurement):
        """Feed in one (x, y) detection; returns the filter's smoothed state."""
        x, y = measurement
        if not self._initialized:
            self.kf.statePost = np.array([[x], [y], [0], [0], [0], [0]], dtype=np.float32)
            self._initialized = True

        self.kf.predict()
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        corrected = self.kf.correct(measured)
        return corrected.flatten()  # [x, y, vx, vy, ax, ay]

    def predict_only(self):
        """Advance one step with no new measurement (e.g. a missed detection)."""
        return self.kf.predict().flatten()


# ---------------------------------------------------------------------------
# 3. Trajectory fitting + forward prediction
# ---------------------------------------------------------------------------

def fit_ballistic_trajectory(times, xs, ys):
    """
    Fit x(t) and y(t) each as a quadratic (constant-acceleration curve) --
    the standard "projectile motion" model: roughly linear in x, a parabola
    in y, assuming gravity dominates (fine for an unpowered, low-drag phase
    of flight).

    Returns two np.poly1d objects: (x_of_t, y_of_t)
    """
    times = np.asarray(times, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    x_coeffs = np.polyfit(times, xs, deg=2)
    y_coeffs = np.polyfit(times, ys, deg=2)

    return np.poly1d(x_coeffs), np.poly1d(y_coeffs)


def predict_future_positions(x_of_t, y_of_t, last_t, n_steps=30, dt=1 / 30):
    """Extrapolate the fitted curves n_steps beyond the last observed time."""
    future_t = last_t + dt * np.arange(1, n_steps + 1)
    return future_t, x_of_t(future_t), y_of_t(future_t)
