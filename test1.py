"""
run_on_video.py

Video-loop wrapper for trajectory_tracker.py.

This is the piece that was missing: detect_bright_object(),
TrajectoryKalmanTracker, and fit_ballistic_trajectory() are all
single-frame / point-in-time functions. To use them on a VIDEO instead
of a single photo, you need a loop that:

  1. Opens the video file (a path on disk).
  2. Reads it frame by frame.
  3. Calls detect_bright_object() on each frame.
  4. Feeds each detection into TrajectoryKalmanTracker.update(), or calls
     .predict_only() when detection fails on a given frame.
  5. Accumulates (t, x, y) history and periodically refits the ballistic
     curve with fit_ballistic_trajectory().
  6. Draws everything and writes it to an output video file.

Usage:
    python run_on_video.py --video path/to/your_clip.mp4 --output tracked.mp4

If you don't have a local file yet, download or export the clip first
(e.g. save an attachment, or `yt-dlp <url>` for a YouTube link) -- this
script needs a real file path on disk, not a web URL.
"""

import cv2
import numpy as np
from collections import deque
from pathlib import Path

from test2 import (
    detect_bright_object,
    TrajectoryKalmanTracker,
    fit_ballistic_trajectory,
    predict_future_positions,
)


def run_on_video(video_path, output_path="C:\\ofir\\after_2_tracked.mp4",
                  min_area=8, refit_every=10, forecast_steps=30,
                  max_missed_frames=15, trail_length=90):

    cap = cv2.VideoCapture(video_path)          # <-- THIS is where the video path goes
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = TrajectoryKalmanTracker(dt=dt)
    trail = deque(maxlen=trail_length)   # for drawing the recent path
    t_hist, x_hist, y_hist = [], [], []  # full history, for curve fitting
    missed_frames = 0
    frame_idx = 0
    x_of_t = y_of_t = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break  # end of video

        t = frame_idx * dt
        detection = detect_bright_object(frame, min_area=min_area)
        state = None

        if detection is not None:
            cx, cy = detection["center"]
            state = tracker.update((cx, cy))
            missed_frames = 0

            bx, by, bw, bh = detection["bbox"]
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)

            t_hist.append(t)
            x_hist.append(state[0])
            y_hist.append(state[1])
        else:
            missed_frames += 1
            if tracker._initialized and missed_frames <= max_missed_frames:
                state = tracker.predict_only()

        if state is not None:
            sx, sy = float(state[0]), float(state[1])
            trail.append((sx, sy))
            cv2.circle(frame, (int(sx), int(sy)), 6, (0, 0, 255), -1)
            speed = (state[2] ** 2 + state[3] ** 2) ** 0.5
            cv2.putText(frame, f"v={speed:.1f}px/s", (int(sx) + 10, int(sy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw the recent trail
        pts = list(trail)
        for i in range(1, len(pts)):
            p1 = tuple(map(int, pts[i - 1]))
            p2 = tuple(map(int, pts[i]))
            cv2.line(frame, p1, p2, (0, 200, 0), 2)

        # Refit the ballistic curve periodically once enough points exist,
        # then draw the fitted curve + forecast ahead of the current point.
        if len(t_hist) >= 6 and frame_idx % refit_every == 0:
            x_of_t, y_of_t = fit_ballistic_trajectory(t_hist, x_hist, y_hist)

        if x_of_t is not None:
            future_t, future_x, future_y = predict_future_positions(
                x_of_t, y_of_t, t_hist[-1], n_steps=forecast_steps, dt=dt
            )
            for i in range(1, len(future_t)):
                p1 = (int(future_x[i - 1]), int(future_y[i - 1]))
                p2 = (int(future_x[i]), int(future_y[i]))
                cv2.line(frame, p1, p2, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, f"frame {frame_idx}  t={t:.2f}s", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Processed {frame_idx} frames from '{video_path}'.")
    print(f"Output written to '{output_path}'.")


if __name__ == "__main__":
    video_link = r"C:\ofir\2.mp4"
    # Example usage:
    # video_folder_link = r"C:\ofir\ofir's_videos"
    # video_folder = Path(video_folder_link)
    # if video_folder.exists() and video_folder.is_dir():
    #     for video_file in video_folder.iterdir():
    #         if video_file.is_file() and video_file.suffix.lower() in [".mp4", ".avi", ".mov"]:
    #             cap = cv2.VideoCapture(str(video_file))
    #             while True:
    #                 success, frame = cap.read()
    #                 if not success:
    #                     break
    #                 frame = cv2.resize(frame, (640, 480))
    #                 cv2.imshow(f"video: {video_file.name}", frame)
    #                 if cv2.waitKey(1) & 0xFF == ord("q"):
    #                     break
    #             cap.release()
    #             cv2.destroyAllWindows()

    run_on_video(video_path=video_link)
    