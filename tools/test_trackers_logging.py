# tools/test_trackers_logging.py
"""
Phase 1.5: Track face and pose, draw overlays, and save data to CSV.
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
from io_utils.csv_logger import CSVLogger

def main():
    print("Starting logger session...")
    logger = CSVLogger()
    
    cap = cv2.VideoCapture(config.CAM_LEFT_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    face_tracker = FaceTracker()
    pose_tracker = PoseTracker()
    
    t_start = time.time()
    fps = 0.0
    prev_t = time.time()
    frame_count = 0

    print("Tracking... Press 'q' to quit and save.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts_ms = int((time.time() - t_start) * 1000)
        face_pts, _ = face_tracker.detect(frame, ts_ms)
        pose_pts, _ = pose_tracker.detect(frame, ts_ms)

        # Log to CSV (frame_count as ID, ts_ms as timestamp)
        logger.log_frame(frame_count, ts_ms, face_pts, pose_pts)

        # Draw visual overlays
        draw_face(frame, face_pts)
        draw_pose(frame, pose_pts)

        # FPS calculation
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        draw_fps(frame, fps)

        cv2.imshow("Data Logging - Press 'q' to Quit", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_tracker.close()
    pose_tracker.close()
    logger.close()
    print(f"Session complete. Saved {frame_count} frames.")

if __name__ == "__main__":
    main()