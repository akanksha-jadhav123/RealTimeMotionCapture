# tools/test_install.py
import cv2
import mediapipe as mp
import numpy as np
import matplotlib
from pathlib import Path

print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)

# Check model files exist
root = Path(__file__).parent.parent
face_model = root / "models" / "face_landmarker.task"
pose_model = root / "models" / "pose_landmarker_lite.task"

print("\nModel files:")
print("  face_landmarker.task:", "OK" if face_model.exists() else "MISSING")
print("  pose_landmarker_lite.task:", "OK" if pose_model.exists() else "MISSING")

print("\nAll imports OK")