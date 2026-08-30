import cv2
import numpy as np
import sys
from scipy import stats
from scipy.optimize import linear_sum_assignment

# --- Configuration Constants ---
MIN_OBJECT_AREA = 10
TRACKING_RADIUS = 20
MIN_FRAMES_FOR_LOGIC = 10
DESCENT_RATIO_THRESHOLD = 0.7
BINARY_THRESHOLD = 10
MAX_MISSED = 3
_DIS_FLOW = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    video_path = r"c:\Users\User\workspace\missile_detection\20.mp4"

class MissileTracker:

    def __init__(self):
        """
        Initializes the tracking system with an empty object list
        and an ID generator.
        """
        self.tracked_objects = []
        self.next_object_id = 0

    def align_frames(self, prev_gray, current_gray, max_shift=3.0, min_response=0.3):

        h, w = prev_gray.shape

        scale = 0.1
        small_prev = cv2.resize(prev_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        small_curr = cv2.resize(current_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        sh, sw = small_prev.shape
        cy_, cx_ = sh // 2, sw // 2
        half = min(sh, sw) // 4
        y1, y2 = max(0, cy_ - half), min(sh, cy_ + half)
        x1, x2 = max(0, cx_ - half), min(sw, cx_ + half)

        roi_prev = small_prev[y1:y2, x1:x2]
        roi_curr = small_curr[y1:y2, x1:x2]

        shift, response = cv2.phaseCorrelate(np.float32(roi_prev), np.float32(roi_curr))
        dx, dy = shift
        dx /= scale
        dy /= scale

        magnitude = np.hypot(dx, dy)
        if magnitude > max_shift or response < min_response:
            return current_gray

        idx, idy = int(round(dx)), int(round(dy))
        if idx == 0 and idy == 0:
            return current_gray

        aligned = np.roll(current_gray, (-idy, -idx), axis=(0, 1))

        m_y, m_x = abs(idy), abs(idx)
        if m_y > 0:
            aligned[:m_y, :] = 0 if idy > 0 else aligned[:m_y, :]
            if idy > 0:
                aligned[:m_y, :] = 0
            else:
                aligned[-m_y:, :] = 0
        if m_x > 0:
            if idx > 0:
                aligned[:, :m_x] = 0
            else:
                aligned[:, -m_x:] = 0

        # --- Empirical gate: did the correction actually help? ---
        margin = max(m_y, m_x, 1)
        prev_c = prev_gray[margin:-margin, margin:-margin]
        curr_c = current_gray[margin:-margin, margin:-margin]
        aligned_c = aligned[margin:-margin, margin:-margin]

        energy_before = np.sum(cv2.absdiff(curr_c, prev_c).astype(np.int32))
        energy_after = np.sum(cv2.absdiff(aligned_c, prev_c).astype(np.int32))

        if energy_after >= energy_before:
            return current_gray  

        return aligned
    
    def update_tracking(self, binary_mask, curr_gray):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        updated_list = []
        matched_ids = set()
    
        # predict() once per object per frame (was: recomputed per contour comparison)
        pred_cache = {}
        for obj in self.tracked_objects:
            prediction = obj["kalman"].predict()
            pred_cache[obj["id"]] = (int(prediction[0][0]), int(prediction[1][0]))

        # --- Pass 1: collect all valid detections from this frame ---
        detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_OBJECT_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            x, y, w, h, cx, cy = tighten_bbox(curr_gray, (x, y, w, h), padding=2, min_area=3)
            detections.append((cx, cy, x, y, w, h))

        # --- Pass 2: global-optimal assignment (Hungarian) instead of greedy ---
        matched_det_idx = set()
        if self.tracked_objects and detections:
            cost = np.zeros((len(self.tracked_objects), len(detections)))
            for i, obj in enumerate(self.tracked_objects):
                pred_x, pred_y = pred_cache[obj["id"]]
                for j, (dx, dy, *_r) in enumerate(detections):
                    cost[i, j] = np.linalg.norm([dx - pred_x, dy - pred_y])

            BIG_M = TRACKING_RADIUS * 1000
            gated_cost = np.where(cost < TRACKING_RADIUS, cost, BIG_M)

            row_ind, col_ind = linear_sum_assignment(gated_cost)
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] >= TRACKING_RADIUS:
                    continue 

                obj = self.tracked_objects[i]
                cx, cy, x, y, w, h = detections[j]
                measurement = np.array([[np.float32(cx)], [np.float32(cy)]], np.float32)
                obj["kalman"].correct(measurement)

                obj.update({
                    "pos": (cx, cy),
                    "bbox": (x, y, w, h),
                    "age": obj["age"] + 1,
                    "missed_frames": 0
                })
                roi = curr_gray[y:y + h, x:x + w]
                intensity = float(np.mean(roi))
                obj["intensity_hist"].append(intensity)
                obj["history"].append((cx, cy))
                updated_list.append(obj)
                matched_ids.add(obj["id"])
                matched_det_idx.add(j)

        # --- Pass 3: unmatched detections become new objects ---
        for j, (cx, cy, x, y, w, h) in enumerate(detections):
            if j not in matched_det_idx:   
        # for cnt in contours:
        #     if cv2.contourArea(cnt) < MIN_OBJECT_AREA:
        #         continue
    
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     M = cv2.moments(cnt)
        #     cx = M["m10"] / M["m00"]
        #     cy = M["m01"] / M["m00"]
        #     matched = False
        #     for obj in self.tracked_objects:

        #         if obj["id"] in matched_ids:
        #             continue  # already matched by another contour this frame
    
        #         pred_x, pred_y = pred_cache[obj["id"]]
        #         dist = np.linalg.norm([cx - pred_x, cy - pred_y])
        #         if dist < TRACKING_RADIUS:
        #             measurement = np.array([[np.float32(cx)], [np.float32(cy)]], np.float32)
        #             obj["kalman"].correct(measurement)
    
        #             obj.update({
        #                 "pos": (cx, cy),
        #                 "bbox": (x, y, w, h),
        #                 "age": obj["age"] + 1,
        #                 "missed_frames": 0
        #             })
        #             roi = curr_gray[y:y + h, x:x + w]
        #             intensity = float(np.mean(roi))
        #             obj["intensity_hist"].append(intensity)
        #             obj["history"].append((cx, cy))
        #             updated_list.append(obj)
        #             matched_ids.add(obj["id"])
        #             matched = True
        #             break
    
        #     if not matched:
                # --- Register new object ---
                kf = create_kalman_filter(cx, cy)
                kf.statePost = np.array([[np.float32(cx)], [np.float32(cy)], [0], [0]], np.float32)
    
                updated_list.append({
                    "id": self.next_object_id,
                    "kalman": kf,
                    "pos": (cx, cy),
                    "bbox": (x, y, w, h),
                    "history": [(cx, cy)],
                    "age": 1,
                    "confirmed": False,
                    "intensity_hist": [],
                    "missed_frames": 0,
                    "has_trail": False,
                    "fail_stats": {
                        "prediction": 0,
                        "acceleration": 0,
                        "model": 0,
                        "intensity_std": 0,
                        "max_diff": 0,
                        "light_fit": 0,
                        "trail": 0,
                        "count": 0
                    }
                })
                self.next_object_id += 1
    
        for obj in self.tracked_objects:
            if obj["id"] not in matched_ids:
                obj["missed_frames"] += 1
                if obj["missed_frames"] <= MAX_MISSED:
                    pred_x, pred_y = pred_cache[obj["id"]]
                    obj["pos"] = (pred_x, pred_y)
                    updated_list.append(obj)
                else:
                    if obj["confirmed"]:
                        pass
                        # print(f"\n=== Object {obj['id']} lost ===")
                        # for k, v in obj["fail_stats"].items():
                        #     print(f"{k}: {v}")
    
        self.tracked_objects = updated_list
 
    def check_missile(self):
        """
        Analyzes the historical movement of each object to confirm missile-like behavior.
        """
        for obj in self.tracked_objects:
            obj["confirmed"] = confidence_calculation(obj)

    def draw_results(self, frame):
        """
        Renders bounding boxes and status labels onto the frame.
        """
        for obj in self.tracked_objects:
            x, y, w, h = obj["bbox"]
            color = (0, 0, 255) if obj["confirmed"] else (0, 255, 255)
            label = f"ID {obj['id']} MISSILE"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if obj["confirmed"]:
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def tighten_bbox(frame, bbox, padding=2, min_area=3, threshold=None, min_roi_side=6):
    x, y, w, h = bbox
    H, W = frame.shape[:2]
 
    # Clip ROI to frame bounds
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    roi = frame[y0:y1, x0:x1]

    fallback_cx, fallback_cy = x + w / 2.0, y + h / 2.0

    if roi.size == 0 or roi.shape[0] < min_roi_side or roi.shape[1] < min_roi_side:
        return (x, y, w, h, fallback_cx, fallback_cy)
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
 
    # Light blur to smooth single-pixel sensor noise before thresholding
    roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
 
    if threshold is None:
        otsu_val, _ = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        strict_val = min(otsu_val + 40, 250)  # 40 = כמה "יותר קפדני" מ-Otsu; לכייל
        _, mask = cv2.threshold(roi_blur, strict_val, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(roi_blur, threshold, 255, cv2.THRESH_BINARY)
 
    # Keep only the largest connected bright blob -> drops stray noise pixels
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return (x, y, w, h, fallback_cx, fallback_cy)  # nothing bright found

    areas = stats[1:, cv2.CC_STAT_AREA]  # skip label 0 (background)
    best_label = 1 + int(np.argmax(areas))
    if stats[best_label, cv2.CC_STAT_AREA] < min_area:
        return (x, y, w, h, fallback_cx, fallback_cy)
 
    bx = stats[best_label, cv2.CC_STAT_LEFT]
    by = stats[best_label, cv2.CC_STAT_TOP]
    bw = stats[best_label, cv2.CC_STAT_WIDTH]
    bh = stats[best_label, cv2.CC_STAT_HEIGHT]

    #pad = max(padding, int(0.15 * max(bw, bh)))
    pad = padding

    # translate back to full-frame coords + padding, clipped to frame bounds
    new_x = max(x0 + bx - pad, 0)
    new_y = max(y0 + by - pad, 0)
    new_x2 = min(x0 + bx + bw + pad, W)
    new_y2 = min(y0 + by + bh + pad, H)

    local_cx, local_cy = centroids[best_label]
    new_cx = x0 + local_cx
    new_cy = y0 + local_cy

    return (new_x, new_y, new_x2 - new_x, new_y2 - new_y, new_cx, new_cy)
def refine_bbox_dynamically(image, bbox, jump_sensitivity=0.15, padding=4):
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]
    
    roi = image[y:y+h, x:x+w]
    if roi.size == 0:
        return bbox

    gray_roi = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)

    x_profile = np.sum(blurred, axis=0)
    y_profile = np.sum(blurred, axis=1)

    x_diff = np.abs(np.diff(x_profile))
    y_diff = np.abs(np.diff(y_profile))

    if len(x_diff) == 0 or len(y_diff) == 0:
        return bbox

    x_thresh = np.max(x_diff) * jump_sensitivity
    y_thresh = np.max(y_diff) * jump_sensitivity

    x_edges = np.where(x_diff > x_thresh)[0]
    y_edges = np.where(y_diff > y_thresh)[0]

    if len(x_edges) == 0 or len(y_edges) == 0:
        return bbox

    raw_x1, raw_x2 = x_edges[0], x_edges[-1] + 1
    raw_y1, raw_y2 = y_edges[0], y_edges[-1] + 1

    M = cv2.moments(blurred[raw_y1:raw_y2, raw_x1:raw_x2])
    if M["m00"] > 0:
        center_x = raw_x1 + (M["m10"] / M["m00"])
        center_y = raw_y1 + (M["m01"] / M["m00"])
    else:
        center_x = (raw_x1 + raw_x2) / 2.0
        center_y = (raw_y1 + raw_y2) / 2.0

    radius_x = int(max(abs(center_x - raw_x1), abs(raw_x2 - center_x)) + padding)
    radius_y = int(max(abs(center_y - raw_y1), abs(raw_y2 - center_y)) + padding)

    local_x1 = int(round(center_x - radius_x))
    local_x2 = int(round(center_x + radius_x))
    local_y1 = int(round(center_y - radius_y))
    local_y2 = int(round(center_y + radius_y))

    new_x = x + local_x1
    new_y = y + local_y1
    new_w = local_x2 - local_x1
    new_h = local_y2 - local_y1

    new_x = max(0, min(new_x, img_w - 1))
    new_y = max(0, min(new_y, img_h - 1))
    new_w = max(1, min(new_w, img_w - new_x))
    new_h = max(1, min(new_h, img_h - new_y))

    return (new_x, new_y, new_w, new_h)

def confidence_calculation(obj):
    hist = obj["history"]
    confidence = 0
    if len(hist) >= 30:
        prediction = obj["kalman"].predict()
        pred_x = int(prediction[0][0])
        pred_y = int(prediction[1][0])

        prediction_error = np.linalg.norm(
            np.array(obj["pos"]) - np.array([pred_x, pred_y])
        )

        confidence = checking_coniditions(obj, prediction_error)
        if len(hist) == 30:
#            windows = [
#                hist[0:10],
#                hist[10:20],
#                hist[20:30]
#            ]
            windows = [
                hist[i:i + 10]
                for i in range(len(hist) - 9)
            ]
        else:
            windows = [
                hist[-10:]
            ]

        threshold = int(MIN_FRAMES_FOR_LOGIC * DESCENT_RATIO_THRESHOLD)

        for recent in windows:
            drops = sum(
                1
                for i in range(len(recent) - 1)
                if recent[i][1] > recent[i + 1][1]
            )
            if drops < threshold:
                return False
        return confidence


def create_kalman_filter(cx, cy):
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.statePost = np.array([[np.float32(cx)], [np.float32(cy)], [0], [0]], np.float32)
    return kf

def direction_filtered_mask(prev_gray, curr_gray, fg_mask, direction=(0, -1),
                             max_angle_deg=50, min_flow_mag=0.5, downscale=0.35):
    if not np.any(fg_mask):
        return fg_mask 
    flow = _DIS_FLOW.calc(prev_gray, curr_gray, None)
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx ** 2 + fy ** 2)

    dx, dy = direction
    norm = np.hypot(dx, dy) + 1e-6
    dx, dy = dx / norm, dy / norm

    cos_angle = (fx * dx + fy * dy) / (mag + 1e-6)
    cos_th = np.cos(np.deg2rad(max_angle_deg))
    direction_ok = (cos_angle > cos_th) & (mag > min_flow_mag)

    filtered = np.zeros_like(fg_mask)
    filtered[direction_ok] = fg_mask[direction_ok]
    return filtered


def thermal_behavior(obj):
    intensities = obj["intensity_hist"]
    try:
        if len(intensities) < 10:
            return False
        score = 0
        intensity_std = np.std(intensities)
        if intensity_std < 35:
            score += 1
        diffs = np.abs(np.diff(intensities))
        if np.max(diffs) < 45:
            score += 1

        y = np.array(intensities)
        x = np.arange(len(y))
        a, b = np.polyfit(x, y, 1)
        y_hat = a * x + b
        fit_error = np.mean((y - y_hat) ** 2)
        norm_error = fit_error / (np.mean(y) ** 2 + 1e-6)
        if norm_error < 0.02:
            score += 1
        return score >= 2
    except:
        return False


def classify_object(obj, prediction_error, pred_th=8, acceleration_variability_th=3, model_th=6):
    hist = obj["history"]
    if len(hist) < 10:
        return False

    score = 0
    if prediction_error < pred_th:
        score += 1

    xs = np.array([p[0] for p in hist])
    ys = np.array([p[1] for p in hist])
    dx = np.diff(xs)
    dy = np.diff(ys)
    speed = np.sqrt(dx ** 2 + dy ** 2)
    acc = np.diff(speed)

    acceleration_variability = np.std(acc)
    if acceleration_variability < acceleration_variability_th:
        score += 1

    try:
        linear = np.poly1d(np.polyfit(xs, ys, 1))
        quad = np.poly1d(np.polyfit(xs, ys, 2))
        err1 = np.mean((ys - linear(xs)) ** 2)
        err2 = np.mean((ys - quad(xs)) ** 2)
        best_error = min(err1, err2)
        if best_error < model_th:
            score += 1
    except:
        pass
    return score >= 2


def checking_coniditions(obj, prediction_error, pred_th=8, acceleration_variability_th=3, model_th=6):
    hist = obj["history"]
    if len(hist) < 10:
        return False
    #print(f"len(hist) = {len(obj['history'])}")
    obj["fail_stats"]["count"] += 1
    score = 0
    # 1. Prediction
    if prediction_error < pred_th:
        #score += 1
        #score += 2
        score += 3
    else:
        #print("prediction fail 1")
        obj["fail_stats"]["prediction"] += 1
        pass
    # 2. Acceleration Variability
    xs = np.array([p[0] for p in hist])
    ys = np.array([p[1] for p in hist])
    dx = np.diff(xs)
    dy = np.diff(ys)
    speed = np.sqrt(dx ** 2 + dy ** 2)
    acc = np.diff(speed)

    acceleration_variability = np.std(acc)
    if acceleration_variability < acceleration_variability_th:
        #score += 1
        score += 2
    else:
        #print("acceleration variability fail 2")
        obj["fail_stats"]["acceleration"] += 1
        pass
    # 3. Model fit
    try:
        linear = np.poly1d(np.polyfit(xs, ys, 1))
        quad = np.poly1d(np.polyfit(xs, ys, 2))
        err1 = np.mean((ys - linear(xs)) ** 2)
        err2 = np.mean((ys - quad(xs)) ** 2)
        best_error = min(err1, err2)
        if best_error < model_th:
            #score += 1
            #score += 2
            score += 3
        else:
            #print("function fail 3")
            obj["fail_stats"]["model"] += 1
            pass
    except:
        pass

    # 4. Thermal & Trail behavior
    intensities = obj["intensity_hist"]
    try:
        intensity_std = np.std(intensities)
        if intensity_std < 35:
            #score += 1
            #score += 2
            score += 3
        else:
            #print("intensity std fail 4")
            obj["fail_stats"]["intensity_std"] += 1
            pass

        diffs = np.abs(np.diff(intensities))
        if np.max(diffs) < 45:
            #score += 1
            score += 3
        else:
            obj["fail_stats"]["max_diff"] += 1
            #print("max diff fail 5")
            pass

        y = np.array(intensities)
        x = np.arange(len(y))
        a, b = np.polyfit(x, y, 1)
        y_hat = a * x + b
        fit_error = np.mean((y - y_hat) ** 2)
        norm_error = fit_error / (np.mean(y) ** 2 + 1e-6)
        if norm_error < 0.02:
            #score += 1
            score += 2
        else:
            obj["fail_stats"]["light_fit"] += 1
            #print("light function fail 6")
            pass
        if obj.get("has_trail", False):
            score += 1
            #print("trail confirmed")
            #print("-----------")
        else:
            obj["fail_stats"]["trail"] += 1
            #print("trail fail 7")
    except:
        pass
    #return score >= 4
    return score >= 10

def check_trail_in_color_frame(obj, frame, contrast_threshold=12):

    hist = obj["history"]
    if len(hist) < 6: return False

    curr_x, curr_y = hist[-1]
    past_x, past_y = hist[-6]
    dx, dy = curr_x - past_x, curr_y - past_y
    dist = np.linalg.norm([dx, dy])
    if dist < 2: return False

    ux, uy = -dx / dist, -dy / dist

    nx, ny = -uy, ux

    shape_len = len(frame.shape)
    if shape_len == 3:
        h, w, _ = frame.shape
        is_color = True
    else:
        h, w = frame.shape
        is_color = False

    hits = 0

    for d in [15, 30, 45, 60]:
        cx = int(curr_x + ux * d)
        cy = int(curr_y + uy * d)

        if not (0 <= cx < w and 0 <= cy < h): continue

        lx = int(cx + nx * 12)
        ly = int(cy + ny * 12)
        rx = int(cx - nx * 12)
        ry = int(cy - ny * 12)

        if 0 <= lx < w and 0 <= ly < h and 0 <= rx < w and 0 <= ry < h:
            if is_color:
                color_center = frame[cy, cx].astype(np.float32)
                color_left = frame[ly, lx].astype(np.float32)
                color_right = frame[ry, rx].astype(np.float32)

                diff_left = np.linalg.norm(color_center - color_left)
                diff_right = np.linalg.norm(color_center - color_right)
            else:
                val_center = int(frame[cy, cx])
                val_left = int(frame[ly, lx])
                val_right = int(frame[ry, rx])

                diff_left = abs(val_center - val_left)
                diff_right = abs(val_center - val_right)

            if diff_left > contrast_threshold and diff_right > contrast_threshold:
                hits += 1

    return hits >= 1


def main():
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    # print(fps)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('output7.mp4', fourcc, fps, (width, height))
    frame_count = 0
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    tracker = MissileTracker()
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    if not ret:
        return

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Stabilization
        aligned_gray = tracker.align_frames(prev_gray, curr_gray)

        # 2. Motion Detection
        diff = cv2.subtract(aligned_gray, prev_gray)
        #diff = cv2.subtract(curr_gray, prev_gray)
        diff = cv2.GaussianBlur(diff, (3, 3), 0)
        _, thresh = cv2.threshold(diff, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

        ###
        #thresh = direction_filtered_mask(
        #    prev_gray, aligned_gray, thresh, direction=(0, -1), max_angle_deg=50
        #)
        ###

        # 3. Noise Reduction
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        '''
        flow = cv2.calcOpticalFlowFarneback(prev_gray, aligned_gray, None, 
                                            pyr_scale=0.5, levels=3, winsize=15, 
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        vy = flow[..., 1]

        upward_mask = np.where(vy < -0.5, 255, 0).astype(np.uint8)

        diff = cv2.subtract(aligned_gray, prev_gray)
        diff = cv2.GaussianBlur(diff, (3, 3), 0)

        diff = cv2.bitwise_and(diff, diff, mask=upward_mask)

        _, thresh = cv2.threshold(diff, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Noise Reduction
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        '''        
        # 4. Logic Updates
        tracker.update_tracking(thresh, curr_gray)

        for obj in tracker.tracked_objects:
            obj["has_trail"] = check_trail_in_color_frame(obj, frame, contrast_threshold=12)

        tracker.check_missile()

        # 5. Visualization
        tracker.draw_results(frame)
        cv2.namedWindow("Detection Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detection Window", width, height)
        #cv2.imshow("gray", aligned_gray)
        cv2.imshow("Detection Window", frame)
        #cv2.imshow("thresh", thresh)
        #out.write(frame)
        if cv2.waitKey(delay) & 0xFF == 27:
            break

        prev_gray = curr_gray
    #out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()