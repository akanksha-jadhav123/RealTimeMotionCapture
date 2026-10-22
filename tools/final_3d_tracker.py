# tools/final_3d_tracker.py
import cv2
import time
import sys
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from visualization.overlay import draw_face, draw_pose, draw_fps
from io_utils.csv_logger import CSVLogger

def main():
    print("Starting single-camera 3D tracker...")
    logger = CSVLogger() # Starts saving to CSV
    
    cap = cv2.VideoCapture(config.CAM_LEFT_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    base_face = mp_python.BaseOptions(model_asset_path=str(config.MP_FACE_MODEL))
    opts_face = vision.FaceLandmarkerOptions(base_options=base_face, running_mode=vision.RunningMode.VIDEO)
    face_mesh = vision.FaceLandmarker.create_from_options(opts_face)

    base_pose = mp_python.BaseOptions(model_asset_path=str(config.MP_POSE_MODEL))
    opts_pose = vision.PoseLandmarkerOptions(base_options=base_pose, running_mode=vision.RunningMode.VIDEO)
    pose_mesh = vision.PoseLandmarker.create_from_options(opts_pose)

    t_start = time.time()
    prev_t = time.time()
    fps = 0.0
    frame_count = 0

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok: break

        ts_ms = int((time.time() - t_start) * 1000)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        res_face = face_mesh.detect_for_video(mp_img, ts_ms)
        res_pose = pose_mesh.detect_for_video(mp_img, ts_ms)

        face_pts_2d, face_pts_3d = {}, {}
        pose_pts_2d, pose_pts_3d = {}, {}

        # Extract Face 3D
        if res_face.face_landmarks:
            lm = res_face.face_landmarks[0]
            world_lm = res_face.face_world_landmarks[0] if hasattr(res_face, 'face_world_landmarks') else []
            h, w = frame.shape[:2]
            for name, idx in config.FACE_INDICES.items():
                face_pts_2d[name] = (lm[idx].x * w, lm[idx].y * h)
                if world_lm:
                    face_pts_3d[name] = (world_lm[idx].x, world_lm[idx].y, world_lm[idx].z)
            draw_face(frame, face_pts_2d)

        # Extract Pose 3D
        if res_pose.pose_landmarks:
            lm = res_pose.pose_landmarks[0]
            world_lm = res_pose.pose_world_landmarks[0]
            h, w = frame.shape[:2]
            for name, idx in config.POSE_INDICES.items():
                pose_pts_2d[name] = (lm[idx].x * w, lm[idx].y * h, lm[idx].visibility)
                pose_pts_3d[name] = (world_lm[idx].x, world_lm[idx].y, world_lm[idx].z)
            draw_pose(frame, pose_pts_2d)

        # Log data to CSV
        logger.log_frame(frame_count, ts_ms, face_pts_2d, face_pts_3d, pose_pts_2d, pose_pts_3d)
        frame_count += 1

        # Display 3D coordinates on screen
        y_offset = 120
        for name in ["nose_tip", "left_wrist", "right_wrist"]:
            if name in face_pts_3d:
                x, y, z = face_pts_3d[name]
                text = f"{name}: X:{x:.2f}m Y:{y:.2f}m Z:{z:.2f}m"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 25
            elif name in pose_pts_3d:
                x, y, z = pose_pts_3d[name]
                text = f"{name}: X:{x:.2f}m Y:{y:.2f}m Z:{z:.2f}m"
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 25

        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0: fps = 0.9 * fps + 0.1 * (1.0 / dt)
        draw_fps(frame, fps)

        cv2.imshow("3D Motion Capture (Single Camera)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    pose_mesh.close()
    logger.close()
    print(f"Done. Processed {frame_count} frames.")

if __name__ == "__main__":
    main()