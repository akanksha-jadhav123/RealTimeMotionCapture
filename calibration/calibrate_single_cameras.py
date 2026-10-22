# calibration/calibrate_single_cameras.py
"""
Calculates individual camera intrinsics using the captured images.
Saves results to calibration_data.npz
"""
import cv2
import numpy as np
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def calibrate_folder(image_folder, label):
    print(f"Calibrating {label}...")
    pattern_size = (config.CHECKERBOARD_COLS, config.CHECKERBOARD_ROWS)
    
    # 3D points of the checkerboard in real world space
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= config.SQUARE_SIZE_M

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(str(image_folder / "*.png"))
    if not images:
        raise ValueError(f"No images found in {image_folder}. Did you capture them?")

    h, w = 0, 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
        else:
            print(f"  Warning: Corners not found in {fname}")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    print(f"  {label} RMS error: {ret:.4f} pixels (Target: < 0.5)")
    
    return K, dist, rvecs, tvecs, objpoints, imgpoints, (w, h)

def main():
    # Calculate left

    K_l, dist_l, rvecs_l, tvecs_l, objp, imgp_l, img_size = calibrate_folder(config.CALIB_LEFT_DIR, "Left Camera")
    # Calculate right
    K_r, dist_r, rvecs_r, tvecs_r, _, imgp_r, _ = calibrate_folder(config.CALIB_RIGHT_DIR, "Right Camera")
    
    # Save raw calibration for stereo step
    np.savez(config.CALIB_DATA_PATH,
             K_l=K_l, dist_l=dist_l,
             K_r=K_r, dist_r=dist_r,
             objp=objp, imgp_l=imgp_l, imgp_r=imgp_r,
             img_size=img_size)
    print(f"\nCalibration saved to {config.CALIB_DATA_PATH}")

if __name__ == "__main__":
    main()