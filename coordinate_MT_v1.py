"""
2D Trajectory Classifier — Starter Project
============================================
Classifies a flight/trajectory (given as a sequence of 2D coordinates,
x = downrange distance, y = altitude) into one of several kinematic
classes:

    ballistic_standard   - classic parabolic ballistic arc
    lofted                - high launch angle, high apogee ballistic arc
    depressed             - low launch angle, flat ballistic arc
    boost_glide           - boosts, then glides with reduced effective gravity
    maneuvering_cruise    - near-constant altitude with periodic maneuvers

WHY SIMULATION-BASED DATA?
Real missile telemetry is not publicly available (and is often classified
or export-controlled). The standard approach in the published literature
(see e.g. AIAA SciTech 2025 "FDA Preprocessing... Missile Classification",
and "Dynamic classification of ballistic missiles using neural networks and
HMMs", Singh & Padmanabhan 2014) is to:
  1. Simulate physically plausible trajectories from equations of motion.
  2. Extract kinematic features (altitude, velocity, acceleration, angles).
  3. Train a classifier (NN, RF, HMM, k-NN+DTW, etc.) on those features.
This script implements that full pipeline end-to-end so you can extend it
with your own physics, real radar/optical-tracking data, or deep sequence
models (LSTM/Transformer) later.

PIPELINE
  1. Physics-based simulators generate labeled (x, y) trajectories.
  2. Feature extraction converts each raw trajectory into a fixed-length
     feature vector (works for variable-length flights).
  3. A classifier (Random Forest by default) is trained and evaluated.
  4. A CLI lets you classify a new trajectory from a CSV of x,y points.

HOW TO EXTEND
  - Swap extract_features() with an LSTM/GRU consuming the raw (x,y) series
    directly (see the "next_steps" notes at the bottom of this file).
  - Feed in real radar/optical tracking data (e.g. from your OpenCV/CV
    pipeline) in place of the simulators.
  - Add measurement noise models matching your actual sensor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import argparse
import sys

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 1. PHYSICS-BASED TRAJECTORY SIMULATORS
#    Each returns two numpy arrays: xs (downrange, m), ys (altitude, m)
# ---------------------------------------------------------------------------

def simulate_ballistic(g=9.81, drag_k=0.0, v0_range=(300, 900),
                        angle_range=(20, 70), dt=0.05, noise=0.0):
    """Basic ballistic arc, optional quadratic drag term drag_k."""
    v0 = np.random.uniform(*v0_range)
    angle = np.radians(np.random.uniform(*angle_range))
    vx, vy = v0 * np.cos(angle), v0 * np.sin(angle)
    x, y = 0.0, 0.0
    xs, ys = [x], [y]
    while y >= 0:
        speed = np.hypot(vx, vy)
        ax = -drag_k * speed * vx
        ay = -g - drag_k * speed * vy
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        xs.append(x)
        ys.append(max(y, 0))
        if len(xs) > 2000:
            break
    xs, ys = np.array(xs), np.array(ys)
    if noise > 0:
        xs = xs + np.random.normal(0, noise, size=xs.shape)
        ys = ys + np.random.normal(0, noise, size=ys.shape)
    return xs, ys


def simulate_lofted(g=9.81, v0_range=(400, 1000), angle_range=(60, 85),
                     dt=0.05, noise=0.0):
    """High launch angle -> high apogee, short range."""
    return simulate_ballistic(g=g, drag_k=0.00003, v0_range=v0_range,
                               angle_range=angle_range, dt=dt, noise=noise)


def simulate_depressed(g=9.81, v0_range=(300, 700), angle_range=(5, 20),
                        dt=0.05, noise=0.0):
    """Low launch angle -> flat, fast, low-altitude trajectory."""
    return simulate_ballistic(g=g, drag_k=0.00005, v0_range=v0_range,
                               angle_range=angle_range, dt=dt, noise=noise)


def simulate_boost_glide(g=9.81, v0_range=(500, 1200), angle_range=(30, 55),
                          dt=0.05, noise=0.0):
    """Boosts ballistically, then glides with reduced effective gravity
    after apogee (crude stand-in for a lifting re-entry vehicle)."""
    v0 = np.random.uniform(*v0_range)
    angle = np.radians(np.random.uniform(*angle_range))
    vx, vy = v0 * np.cos(angle), v0 * np.sin(angle)
    x, y = 0.0, 0.0
    xs, ys = [x], [y]
    apogee_reached = False
    glide_lift = np.random.uniform(0.3, 0.6)
    while y >= 0:
        speed = np.hypot(vx, vy)
        if vy < 0 and not apogee_reached:
            apogee_reached = True
        if apogee_reached:
            ax = -0.00002 * speed * vx
            ay = -g * (1 - glide_lift) - 0.00002 * speed * vy
        else:
            ax = -0.00001 * speed * vx
            ay = -g - 0.00001 * speed * vy
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        xs.append(x)
        ys.append(max(y, 0))
        if len(xs) > 2500:
            break
    xs, ys = np.array(xs), np.array(ys)
    if noise > 0:
        xs = xs + np.random.normal(0, noise, size=xs.shape)
        ys = ys + np.random.normal(0, noise, size=ys.shape)
    return xs, ys


def simulate_maneuvering_cruise(v0_range=(200, 350), alt_range=(500, 2000),
                                 dt=0.05, noise=0.0):
    """Near-constant-altitude cruise with periodic sinusoidal maneuvers
    (stand-in for a cruise-missile-like flight profile)."""
    v0 = np.random.uniform(*v0_range)
    alt = np.random.uniform(*alt_range)
    total_time = np.random.uniform(60, 120)
    n = int(total_time / dt)
    xs = np.zeros(n)
    ys = np.zeros(n)
    ys[:] = alt
    x = 0.0
    freq = np.random.uniform(0.05, 0.15)
    amp = np.random.uniform(50, 300)
    for i in range(1, n):
        x += v0 * dt
        xs[i] = x
        ys[i] = alt + amp * np.sin(freq * x / 100.0)
    if noise > 0:
        xs = xs + np.random.normal(0, noise, size=xs.shape)
        ys = ys + np.random.normal(0, noise, size=ys.shape)
    return xs, ys


SIMULATORS = {
    'ballistic_standard': lambda noise: simulate_ballistic(noise=noise),
    'lofted': lambda noise: simulate_lofted(noise=noise),
    'depressed': lambda noise: simulate_depressed(noise=noise),
    'boost_glide': lambda noise: simulate_boost_glide(noise=noise),
    'maneuvering_cruise': lambda noise: simulate_maneuvering_cruise(noise=noise),
}


# ---------------------------------------------------------------------------
# 2. FEATURE EXTRACTION
#    Converts a variable-length (x, y) trajectory into a fixed-length
#    vector of physically meaningful features.
# ---------------------------------------------------------------------------

def extract_features(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)

    vx = np.diff(xs)
    vy = np.diff(ys)
    speed = np.hypot(vx, vy)
    ax = np.diff(vx)
    ay = np.diff(vy)
    accel = np.hypot(ax, ay)

    max_alt = ys.max()
    range_total = xs[-1] - xs[0]
    apogee_idx = int(np.argmax(ys))
    apogee_frac = apogee_idx / n

    mean_speed = speed.mean() if len(speed) else 0.0
    max_speed = speed.max() if len(speed) else 0.0
    min_speed = speed.min() if len(speed) else 0.0
    speed_std = speed.std() if len(speed) else 0.0
    mean_accel = accel.mean() if len(accel) else 0.0
    accel_std = accel.std() if len(accel) else 0.0

    launch_angle = np.degrees(np.arctan2(vy[0], vx[0])) if len(vy) else 0.0
    impact_angle = np.degrees(np.arctan2(vy[-1], vx[-1])) if len(vy) else 0.0
    aspect_ratio = max_alt / (range_total + 1e-6)

    ascent_len = apogee_idx + 1 if apogee_idx > 0 else n
    descent_len = n - apogee_idx
    ascent_descent_ratio = ascent_len / (descent_len + 1e-6)

    curvature_proxy = accel_std / (mean_speed + 1e-6)
    vy_sign_changes = int(np.sum(np.diff(np.sign(vy)) != 0)) if len(vy) > 1 else 0

    return {
        'max_altitude': max_alt,
        'range_total': range_total,
        'apogee_fraction': apogee_frac,
        'flight_time_steps': n,
        'mean_speed': mean_speed,
        'max_speed': max_speed,
        'min_speed': min_speed,
        'speed_std': speed_std,
        'mean_accel': mean_accel,
        'accel_std': accel_std,
        'launch_angle_deg': launch_angle,
        'impact_angle_deg': impact_angle,
        'aspect_ratio': aspect_ratio,
        'ascent_descent_ratio': ascent_descent_ratio,
        'curvature_proxy': curvature_proxy,
        'vy_sign_changes': vy_sign_changes,
    }


FEATURE_COLUMNS = [
    'max_altitude', 'range_total', 'apogee_fraction', 'flight_time_steps',
    'mean_speed', 'max_speed', 'min_speed', 'speed_std', 'mean_accel',
    'accel_std', 'launch_angle_deg', 'impact_angle_deg', 'aspect_ratio',
    'ascent_descent_ratio', 'curvature_proxy', 'vy_sign_changes',
]


# ---------------------------------------------------------------------------
# 3. DATASET GENERATION
# ---------------------------------------------------------------------------

def build_dataset(n_per_class=400, noise_max=5.0, seed=RANDOM_SEED):
    np.random.seed(seed)
    rows, labels = [], []
    for label, sim_fn in SIMULATORS.items():
        for _ in range(n_per_class):
            noise = np.random.uniform(0, noise_max)
            xs, ys = sim_fn(noise)
            rows.append(extract_features(xs, ys))
            labels.append(label)
    df = pd.DataFrame(rows)
    df['label'] = labels
    return df


# ---------------------------------------------------------------------------
# 4. TRAIN / EVALUATE
# ---------------------------------------------------------------------------

def train_model(df, model=None):
    X = df[FEATURE_COLUMNS].values
    y = df['label'].values

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, stratify=y_enc, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model is None:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_SEED)

    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    print(f"Test accuracy: {accuracy_score(y_test, preds):.4f}\n")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, preds),
        index=label_encoder.classes_, columns=label_encoder.classes_))

    return model, scaler, label_encoder


def save_artifacts(model, scaler, label_encoder, path_prefix='trajectory_model'):
    joblib.dump(model, f'{path_prefix}.joblib')
    joblib.dump(scaler, f'{path_prefix}_scaler.joblib')
    joblib.dump(label_encoder, f'{path_prefix}_labels.joblib')
    print(f"Saved: {path_prefix}.joblib, {path_prefix}_scaler.joblib, "
          f"{path_prefix}_labels.joblib")


def load_artifacts(path_prefix='trajectory_model'):
    model = joblib.load(f'{path_prefix}.joblib')
    scaler = joblib.load(f'{path_prefix}_scaler.joblib')
    label_encoder = joblib.load(f'{path_prefix}_labels.joblib')
    return model, scaler, label_encoder


def classify_csv(csv_path, model, scaler, label_encoder):
    """csv_path must contain columns 'x' and 'y' (2D coordinates in order)."""
    points = pd.read_csv(csv_path)
    xs, ys = points['x'].values, points['y'].values
    frame_object = 480
    new_y_axis = frame_object - np.array(ys)
    ys = new_y_axis
    feats = extract_features(xs, ys)
    X = np.array([[feats[c] for c in FEATURE_COLUMNS]])
    X_s = scaler.transform(X)
    pred_idx = model.predict(X_s)[0]
    proba = model.predict_proba(X_s)[0] if hasattr(model, 'predict_proba') else None
    label = label_encoder.inverse_transform([pred_idx])[0]
    print(f"Predicted class: {label}")
    if proba is not None:
        for cls, p in sorted(zip(label_encoder.classes_, proba), key=lambda t: -t[1]):
            print(f"  {cls:22s} {p:.3f}")
    return label


# ---------------------------------------------------------------------------
# Graphing the trajectory from CSV
# ---------------------------------------------------------------------------

def trajectory_graph(csv_path):
    array = np.array(pd.read_csv(csv_path)[['x', 'y']].values)
    frame_object = 480
    new_y_axis = frame_object - np.array([point[1] for point in array])
    # Convert the array to a DataFrame
    df = pd.DataFrame(array, columns=['x', 'y'])
    df['y'] = new_y_axis

    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)

    # Plot the trajectory
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()  # Invert y-axis to match the coordinate system
    ax.set_xlim(0, 1000)
    ax.set_ylim(1000, 0)
    ax.plot(df['x'], df['y'], marker='o', linestyle='-', color='blue')
    ax.set_xlabel('Downrange Distance (m)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Trajectory')
    ax.grid(True)
    plt.show()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="2D Trajectory Classifier")
    parser.add_argument('--train', action='store_true', help="Train a fresh model on simulated data")
    parser.add_argument('--classify', type=str, default=None,
                         help="Path to CSV with columns x,y to classify")
    parser.add_argument('--n_per_class', type=int, default=400)
    args = parser.parse_args()

    if args.train:
        df = build_dataset(n_per_class=args.n_per_class)
        model, scaler, label_encoder = train_model(df)
        save_artifacts(model, scaler, label_encoder)

    if args.classify:
        try:
            model, scaler, label_encoder = load_artifacts()
        except FileNotFoundError:
            print("No trained model found. Run with --train first.")
            sys.exit(1)
        classify_csv(args.classify, model, scaler, label_encoder)
        trajectory_graph(args.classify)

    if not args.train and not args.classify:
        print("No action specified. Running a full train + demo cycle...\n")
        df = build_dataset(n_per_class=args.n_per_class)
        model, scaler, label_encoder = train_model(df)
        save_artifacts(model, scaler, label_encoder)


if __name__ == '__main__':
    main()



# ---------------------------------------------------------------------------
# NEXT STEPS / HOW TO GROW THIS PROJECT
# ---------------------------------------------------------------------------
# 1. Real sensor data: replace SIMULATORS with trajectories from your own
#    optical tracking / radar pipeline (x, y pixel or geo-coordinates over
#    time). Keep timestamps if your sampling rate is irregular, and resample
#    to a fixed dt before feature extraction, or use time-aware features.
#
# 2. Sequence models (no feature engineering): feed the raw (x, y) series
#    directly into an LSTM/GRU/1D-CNN using padded sequences
#    (tf.keras.preprocessing.sequence.pad_sequences or PyTorch
#    pack_padded_sequence). This captures temporal patterns feature
#    engineering might miss, at the cost of needing more data.
#
# 3. Early/partial-trajectory classification: many real systems must
#    classify during the boost phase, before the full flight is observed.
#    Try training on truncated versions of each trajectory (e.g. first 20%,
#    40%, 60% of points) so the model works with partial observations too.
#
# 4. Dynamic time warping + k-NN: an alternative to feature engineering
#    that works well on raw variable-length trajectories (see "Similarity-
#    Based Launch Classification Tool", Dichter et al. 2021). Useful when
#    you don't trust hand-crafted features to generalize.
#
# 5. Add realistic sensor noise/occlusion models matching your actual
#    tracking hardware (dropped frames, quantization, jitter) so the
#    classifier is robust to real deployment conditions.
#
# 6. Track uncertainty: use model.predict_proba() and require a confidence
#    threshold before acting on a classification, especially important
#    for early-flight (low-information) predictions.


