# tracking/pose_tracker.py
"""MediaPipe PoseLandmarker wrapper. Detects arm joints."""

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import config


class PoseTracker:
    def __init__(self):
        base = mp_python.BaseOptions(model_asset_path=str(config.MP_POSE_MODEL))
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.POSE_MIN_CONF,
            min_pose_presence_confidence=config.POSE_MIN_CONF,
            min_tracking_confidence=config.POSE_MIN_CONF,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)

    def detect(self, bgr_frame, timestamp_ms):
        """
        Returns (points_dict, raw_landmarks) or (None, None) if no pose.
        points_dict: {name: (x_px, y_px, visibility)}
        """
        rgb = bgr_frame[:, :, ::-1].copy()
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_img, int(timestamp_ms))

        if not result.pose_landmarks:
            return None, None

        lm = result.pose_landmarks[0]
        h, w = bgr_frame.shape[:2]
        pts = {}
        for name, idx in config.POSE_INDICES.items():
            pts[name] = (lm[idx].x * w, lm[idx].y * h, lm[idx].visibility)
        return pts, lm

    def close(self):
        self.landmarker.close()