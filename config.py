# config.py
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CALIB_IMG_DIR = DATA_DIR / "calibration_images"
CALIB_LEFT_DIR = CALIB_IMG_DIR / "left"
CALIB_RIGHT_DIR = CALIB_IMG_DIR / "right"
LOG_DIR = DATA_DIR / "logs"
MODELS_DIR = ROOT / "models"
CALIB_DATA_PATH = ROOT / "calibration" / "calibration_data.npz"

for p in (DATA_DIR, CALIB_LEFT_DIR, CALIB_RIGHT_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Cameras
CAM_LEFT_INDEX  = 0
CAM_RIGHT_INDEX = 1
CAM_BACKEND_NAME = "dshow"

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
FPS_TARGET   = 30
BASELINE_M = 0.15

# Checkerboard
# config.py
CHECKERBOARD_COLS = 10
CHECKERBOARD_ROWS = 7
SQUARE_SIZE_M = 0.025

# MediaPipe
MP_FACE_MODEL = MODELS_DIR / "face_landmarker.task"
MP_POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"

FACE_MIN_CONF = 0.5
POSE_MIN_CONF = 0.5

# Landmarks
FACE_INDICES = {
    "nose_tip":        1,
    "left_eye_outer":  33,
    "left_eye_inner":  133,
    "right_eye_outer": 263,
    "right_eye_inner": 362,
    "mouth_left":      61,
    "mouth_right":     291,
}

POSE_INDICES = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
}

POSE_CONNECTIONS = [
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "right_shoulder"),
]

# Stereo / triangulation
MIN_VISIBILITY    = 0.5
EPIPOLAR_Y_TOL_PX = 3.0
MAX_SYNC_DT_MS    = 34
Z_VALID_MIN_M     = 0.2
Z_VALID_MAX_M     = 5.0

# Colours (BGR)
COLOR_FACE = (0, 255, 0)
COLOR_POSE = (255, 128, 0)
COLOR_LINE = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_BG   = (0, 0, 0)
