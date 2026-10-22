# tracking/face_tracker.py
"""MediaPipe FaceLandmarker wrapper. Detects selected face points."""

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import config


class FaceTracker:
    def __init__(self):
        base = mp_python.BaseOptions(model_asset_path=str(config.MP_FACE_MODEL))
        opts = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=config.FACE_MIN_CONF,
            min_face_presence_confidence=config.FACE_MIN_CONF,
            min_tracking_confidence=config.FACE_MIN_CONF,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)

    def detect(self, bgr_frame, timestamp_ms):
        """
        Returns (points_dict, raw_landmarks) or (None, None) if no face.
        points_dict: {name: (x_px, y_px)}
        raw_landmarks: full MediaPipe landmark list (normalized 0..1)
        """
        rgb = bgr_frame[:, :, ::-1].copy()          # BGR → RGB
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_img, int(timestamp_ms))

        if not result.face_landmarks:
            return None, None

        lm = result.face_landmarks[0]
        h, w = bgr_frame.shape[:2]
        pts = {}
        for name, idx in config.FACE_INDICES.items():
            pts[name] = (lm[idx].x * w, lm[idx].y * h)
        return pts, lm

    def close(self):
        self.landmarker.close()