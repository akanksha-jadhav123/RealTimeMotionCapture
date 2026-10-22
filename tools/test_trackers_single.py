# tools/test_trackers_single.py
"""
Phase 1 test: run face and pose trackers on ONE camera and display results.
Press 'q' to quit.
"""

import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from tracking.face_tracker import FaceTracker
from tracking.pose_tracker import PoseTracker
from visualization.overlay import draw_face, draw_pose, draw_fps


def main():
    print("─" * 60)
    print(" Phase 1 test: single-camera tracking")
    print("─" * 60)

    cap = cv2.VideoCapture(config.CAM_LEFT_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("!! Could not open camera", config.CAM_LEFT_INDEX)
        return

    print(f" Camera {config.CAM_LEFT_INDEX} opened.")
    print(" Loading MediaPipe models...")
    face_tracker = FaceTracker()
    pose_tracker = PoseTracker()
    print(" Models loaded. Starting preview...")
    print(" Press 'q' in the window to quit.")
    print()

    t_start = time.time()
    fps = 0.0
    prev_t = time.time()
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("!! Frame grab failed")
            break

        # MediaPipe requires monotonically increasing timestamp in ms
        ts_ms = int((time.time() - t_start) * 1000)

        face_pts, _   = face_tracker.detect(frame, ts_ms)
        pose_pts, _   = pose_tracker.detect(frame, ts_ms)

        draw_face(frame, face_pts)
        draw_pose(frame, pose_pts)

        # Rolling FPS
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        draw_fps(frame, fps)

        # Detection status overlay
        face_status = "FACE OK"  if face_pts else "FACE --"
        pose_status = "POSE OK"  if pose_pts else "POSE --"
        cv2.putText(frame, face_status, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if face_pts else (0, 0, 255), 2)
        cv2.putText(frame, pose_status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if pose_pts else (0, 0, 255), 2)

        cv2.imshow("Phase 1 — 2D Tracking", frame)
        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if cv2.getWindowProperty("Phase 1 — 2D Tracking",
                                 cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    face_tracker.close()
    pose_tracker.close()
    print(f" Done. Processed {frame_count} frames at avg {fps:.1f} FPS")


if __name__ == "__main__":
    main()