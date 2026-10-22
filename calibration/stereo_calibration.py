# calibration/stereo_calibration.py
"""
Computes stereo extrinsics and rectification matrices.
Overwrites calibration_data.npz with the complete stereo parameters.
"""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def main():
    print("Loading single-camera calibration data...")
    data = np.load(config.CALIB_DATA_PATH)
    
    K_l, dist_l = data['K_l'], data['dist_l']
    K_r, dist_r = data['K_r'], data['dist_r']
    objp, imgp_l, imgp_r = data['objp'], data['imgp_l'], data['imgp_r']
    img_size = tuple(data['img_size'])

    print("Calculating stereo extrinsics (R, T, E, F)...")
    # CALIB_FIX_INTRINSIC prevents changing the K matrices we already calculated
    ret, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objp, imgp_l, imgp_r, K_l, dist_l, K_r, dist_r, img_size,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    print(f"Stereo RMS error: {ret:.4f} pixels")
    print(f"Baseline (T): {np.linalg.norm(T):.4f} meters")

    print("Calculating rectification transforms...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_l, dist_l, K_r, dist_r, img_size, R, T, alpha=0
    )

    print(f"Saving complete stereo calibration to {config.CALIB_DATA_PATH}")
    np.savez(config.CALIB_DATA_PATH,
             K_l=K_l, dist_l=dist_l, K_r=K_r, dist_r=dist_r,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             img_size=img_size)
    
    print("Done! You are ready for Phase 3 (Triangulation).")

if __name__ == "__main__":
    main()