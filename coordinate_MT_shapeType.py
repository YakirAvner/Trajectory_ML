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
# 1. SHAPE-CATEGORY SIMULATORS
#    Each returns (xs, ys) as plain numpy arrays, in whatever units you like
#    (pixels, meters — irrelevant, since features below are scale-normalized).
# ---------------------------------------------------------------------------

def sim_linear(noise=0.0):
    """Straight-line motion at a random slope and length."""
    n = np.random.randint(30, 120)
    slope = np.random.uniform(-3, 3)
    length = np.random.uniform(50, 1000)
    xs = np.linspace(0, length, n)
    ys = slope * xs + np.random.uniform(-10, 10)
    if noise > 0:
        xs = xs + np.random.normal(0, noise, n)
        ys = ys + np.random.normal(0, noise, n)
    return xs, ys


def sim_parabola_arc(noise=0.0):
    """Symmetric parabolic arc — classic ballistic shape, apogee centered."""
    n = np.random.randint(40, 200)
    length = np.random.uniform(100, 2000)
    height = np.random.uniform(20, 1500)
    xs = np.linspace(0, length, n)
    a = -4 * height / length ** 2
    ys = a * (xs - length / 2) ** 2 + height
    ys = ys - ys.min()
    if noise > 0:
        xs = xs + np.random.normal(0, noise, n)
        ys = ys + np.random.normal(0, noise, n)
    return xs, ys


def sim_skewed_arc(noise=0.0):
    """Asymmetric arc: apogee not at the midpoint (lofted/depressed-style,
    or wind/thrust-skewed ballistic shots)."""
    n = np.random.randint(40, 200)
    length = np.random.uniform(100, 2000)
    height = np.random.uniform(20, 1500)
    apogee_frac = np.random.uniform(0.2, 0.8)
    xs = np.linspace(0, length, n)
    apogee_x = length * apogee_frac
    ys = np.where(
        xs < apogee_x,
        height * (xs / apogee_x) ** 1.5,
        height * (1 - (xs - apogee_x) / (length - apogee_x + 1e-9)) ** 1.2,
    )
    ys = np.nan_to_num(ys)
    if noise > 0:
        xs = xs + np.random.normal(0, noise, n)
        ys = ys + np.random.normal(0, noise, n)
    return xs, ys


def sim_s_curve(noise=0.0):
    """Oscillating / wavy path with net forward drift — maneuvering or
    cruise-missile-style motion."""
    n = np.random.randint(50, 200)
    length = np.random.uniform(200, 2000)
    xs = np.linspace(0, length, n)
    amp = np.random.uniform(20, 400)
    freq = np.random.uniform(1.5, 3)
    ys = amp * np.sin(freq * np.pi * xs / length) + xs * np.random.uniform(0.05, 0.3)
    if noise > 0:
        xs = xs + np.random.normal(0, noise, n)
        ys = ys + np.random.normal(0, noise, n)
    return xs, ys


def sim_erratic_nonmissile(noise=0.0):
    """Random-walk path with no consistent curve fit — should NOT be
    classified as a coherent ballistic/aerodynamic trajectory."""
    n = np.random.randint(20, 150)
    xs = np.cumsum(np.random.uniform(-15, 15, n))
    ys = np.cumsum(np.random.uniform(-15, 15, n))
    if noise > 0:
        xs = xs + np.random.normal(0, noise, n)
        ys = ys + np.random.normal(0, noise, n)
    return xs, ys


def sim_stationary_jitter(noise=2.0):
    """Near-zero net displacement: sensor/detector noise on something
    that isn't really moving (false positive for a "trajectory")."""
    n = np.random.randint(20, 100)
    cx, cy = np.random.uniform(0, 500), np.random.uniform(0, 500)
    xs = cx + np.random.normal(0, noise, n)
    ys = cy + np.random.normal(0, noise, n)
    return xs, ys


SHAPE_SIMULATORS = {
    'linear': sim_linear,
    'parabolic_arc': sim_parabola_arc,
    'skewed_arc': sim_skewed_arc,
    's_curve_maneuver': sim_s_curve,
    'erratic_nonmissile': sim_erratic_nonmissile,
    'stationary_jitter': sim_stationary_jitter,
}


# ---------------------------------------------------------------------------
# 2. SHAPE FEATURE EXTRACTION
#    Scale- and origin-invariant: works regardless of camera resolution,
#    distance-to-target, or coordinate units (pixels, meters, etc.).
# ---------------------------------------------------------------------------

def _r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-9)


def shape_features(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)

    # Origin-shift + scale-normalize
    # resolution, distance-to-target, and absolute position in frame.
    x0, y0 = xs[0], ys[0]
    xs_n, ys_n = xs - x0, ys - y0
    scale = max(np.hypot(xs_n, ys_n).max(), 1e-9)
    xu, yu = xs_n / scale, ys_n / scale

    # Linear fit quality: R^2 close to 1 means "this is basically a line".
    if np.ptp(xu) > 1e-6:
        lin_coef = np.polyfit(xu, yu, 1)
        lin_r2 = _r_squared(yu, np.polyval(lin_coef, xu))
    else:
        lin_r2 = 0.0

    # Quadratic fit quality: R^2 close to 1 (and lin_r2 clearly lower)
    # means "this is a parabola, not a line".
    if np.ptp(xu) > 1e-6 and n >= 4:
        quad_coef = np.polyfit(xu, yu, 2)
        quad_pred = np.polyval(quad_coef, xu)
        quad_r2 = _r_squared(yu, quad_pred)
        quad_curvature = quad_coef[0]
    else:
        quad_r2 = 0.0
        quad_curvature = 0.0

    vx, vy = np.diff(xu), np.diff(yu)
    heading = np.arctan2(vy, vx)
    heading_unwrapped = np.unwrap(heading)
    total_turning = (np.sum(np.abs(np.diff(heading_unwrapped)))
                      if len(heading_unwrapped) > 1 else 0.0)
    net_heading_change = (abs(heading_unwrapped[-1] - heading_unwrapped[0])
                           if len(heading_unwrapped) else 0.0)

    path_len = np.sum(np.hypot(vx, vy))
    straight_dist = np.hypot(xu[-1] - xu[0], yu[-1] - yu[0])
    tortuosity = path_len / (straight_dist + 1e-9)

    speed = np.hypot(vx, vy)
    speed_cv = speed.std() / (speed.mean() + 1e-9) if len(speed) else 0.0

    vy_sign_changes = (int(np.sum(np.diff(np.sign(vy)) != 0))
                        if len(vy) > 1 else 0)
    vy_sign_change_rate = vy_sign_changes / n

    apogee_idx = int(np.argmax(yu))
    apogee_frac = apogee_idx / n
    aspect_ratio = np.ptp(yu) / (np.ptp(xu) + 1e-9)

    jitter = np.mean(np.abs(np.diff(speed))) if len(speed) > 1 else 0.0

    max_step = np.hypot(vx, vy).max() if len(vx) else 0
    mean_step = np.hypot(vx, vy).mean() if len(vx) else 1e-9
    jump_ratio = max_step / (mean_step + 1e-9)

    return {
        'linear_r2': lin_r2,
        'quad_r2': quad_r2,
        'quad_curvature': quad_curvature,
        'total_turning': total_turning,
        'net_heading_change': net_heading_change,
        'tortuosity': tortuosity,
        'speed_cv': speed_cv,
        'vy_sign_change_rate': vy_sign_change_rate,
        'apogee_frac': apogee_frac,
        'aspect_ratio': aspect_ratio,
        'jitter': jitter,
        'jump_ratio': jump_ratio,
        'n_points': n,
    }


FEATURE_COLUMNS = [
    'linear_r2', 'quad_r2', 'quad_curvature', 'total_turning',
    'net_heading_change', 'tortuosity', 'speed_cv', 'vy_sign_change_rate',
    'apogee_frac', 'aspect_ratio', 'jitter', 'jump_ratio', 'n_points',
]


# ---------------------------------------------------------------------------
# 3. DATASET GENERATION
# ---------------------------------------------------------------------------

def build_dataset(n_per_class=350, noise_max=3.0, seed=RANDOM_SEED):
    np.random.seed(seed)
    rows, labels = [], []
    for label, sim_fn in SHAPE_SIMULATORS.items():
        for _ in range(n_per_class):
            noise = np.random.uniform(0, noise_max)
            xs, ys = sim_fn(noise=noise)
            rows.append(shape_features(xs, ys))
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
            n_estimators=250, max_depth=12, random_state=RANDOM_SEED)

    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    print(f"Test accuracy: {accuracy_score(y_test, preds):.4f}\n")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, preds),
        index=label_encoder.classes_, columns=label_encoder.classes_))

    return model, scaler, label_encoder


def save_artifacts(model, scaler, label_encoder, path_prefix='shape_model'):
    joblib.dump(model, f'{path_prefix}.joblib')
    joblib.dump(scaler, f'{path_prefix}_scaler.joblib')
    joblib.dump(label_encoder, f'{path_prefix}_labels.joblib')
    print(f"Saved: {path_prefix}.joblib, {path_prefix}_scaler.joblib, "
          f"{path_prefix}_labels.joblib")


def load_artifacts(path_prefix='shape_model'):
    model = joblib.load(f'{path_prefix}.joblib')
    scaler = joblib.load(f'{path_prefix}_scaler.joblib')
    label_encoder = joblib.load(f'{path_prefix}_labels.joblib')
    return model, scaler, label_encoder


# ---------------------------------------------------------------------------
# 5. CLASSIFY A REAL CSV (pixel or any-unit coordinates)
# ---------------------------------------------------------------------------

def classify_csv(csv_path, model, scaler, label_encoder,
                  flip_y=True, frame_height=480, confidence_threshold=0.5):
    """csv_path must contain columns 'x' and 'y'.

    flip_y: set True if your coordinates come from image/pixel space
    (y increases downward). This converts to the "y increases upward"
    convention used consistently across feature extraction.
    """
    points = pd.read_csv(csv_path)
    xs, ys = points['x'].values.astype(float), points['y'].values.astype(float)
    if flip_y:
        ys = frame_height - ys

    feats = shape_features(xs, ys)
    X = np.array([[feats[c] for c in FEATURE_COLUMNS]])
    X_s = scaler.transform(X)

    pred_idx = model.predict(X_s)[0]
    proba = model.predict_proba(X_s)[0] if hasattr(model, 'predict_proba') else None
    label = label_encoder.inverse_transform([pred_idx])[0]

    print(f"Predicted shape class: {label}")
    top_conf = 0.0
    if proba is not None:
        ranked = sorted(zip(label_encoder.classes_, proba), key=lambda t: -t[1])
        top_conf = ranked[0][1]
        for cls, p in ranked:
            print(f"  {cls:22s} {p:.3f}")

    if top_conf < confidence_threshold:
        print(f"\nWARNING: top confidence ({top_conf:.2f}) is below "
              f"{confidence_threshold:.2f} - treat this classification as "
              f"unreliable (ambiguous or out-of-distribution shape).")

    if label in ('erratic_nonmissile', 'stationary_jitter'):
        print("\nNOTE: this shape does NOT resemble a coherent ballistic / "
              "aerodynamic trajectory. Likely causes: sensor noise, a "
              "non-projectile object, tracking error, or too short/partial "
              "a segment of a real flight to show curvature.")

    print(f"\n[diagnostic] linear_r2={feats['linear_r2']:.3f}  "
          f"quad_r2={feats['quad_r2']:.3f}  tortuosity={feats['tortuosity']:.3f}  "
          f"n_points={feats['n_points']}")

    return label


# ---------------------------------------------------------------------------
# 6. PLOT A TRAJECTORY FROM CSV
# ---------------------------------------------------------------------------

def trajectory_graph(csv_path, frame_height=480, save_path=None):
    points = pd.read_csv(csv_path)
    xs, ys = points['x'].values.astype(float), points['y'].values.astype(float)
    ys = frame_height - ys

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.invert_yaxis()  # optional: flip y-axis to match "up is positive" convention
    ax.plot(xs, ys, marker='o', linestyle='-', color='blue', markersize=3)
    ax.set_xlim(0, 1000)
    ax.set_ylim(1000, 0)
    ax.set_xlabel('X (normalized to physics-up convention)')
    ax.set_ylabel('Y (altitude-like, down is positive)')
    ax.set_title('Trajectory Shape')
    ax.grid(True)
    out = save_path or 'trajectory_plot.png'
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Saved plot: {out}")


# ---------------------------------------------------------------------------
# Command Line Interface (CLI)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="2D Trajectory Shape Classifier")
    parser.add_argument('--train', action='store_true',
                         help="Train a fresh shape-classifier on simulated data")
    parser.add_argument('--classify', type=str, default=None,
                         help="Path to CSV with columns x,y to classify")
    parser.add_argument('--n_per_class', type=int, default=350)
    parser.add_argument('--graph', action='store_true',
                         help="Also save a plot of the classified trajectory")
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
        if args.graph:
            trajectory_graph(args.classify)

    if not args.train and not args.classify:
        print("No action specified. Running a full train + demo cycle...\n")
        df = build_dataset(n_per_class=args.n_per_class)
        model, scaler, label_encoder = train_model(df)
        save_artifacts(model, scaler, label_encoder)


if __name__ == '__main__':
    main()
