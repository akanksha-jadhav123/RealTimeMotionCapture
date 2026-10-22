# calibration/capture_calibration_images.py
"""
Opens both cameras. Press SPACE to capture a pair when both see the checkerboard.
Press 'q' to quit.
"""
import sys
import time
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def main():
    print("Opening cameras...")
    # NOTE: Change backends if DSHOW fails for your USB camera
    cap_l = cv2.VideoCapture(config.CAM_LEFT_INDEX, cv2.CAP_DSHOW)
    cap_r = cv2.VideoCapture(config.CAM_RIGHT_INDEX, cv2.CAP_DSHOW)

    for cap in (cap_l, cap_r):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not (cap_l.isOpened() and cap_r.isOpened()):
        print("!! Could not open both cameras. Check USB connections and indices in config.py.")
        return

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    pattern_size = (config.CHECKERBOARD_COLS, config.CHECKERBOARD_ROWS)
    
    print(f"Target: {pattern_size[0]}x{pattern_size[1]} corners. Press SPACE to save, 'q' to quit.")
    print("Move the board around (near, far, tilted left/right/up/down). Capture 25+ pairs.")

    pair_count = 0
    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not (ret_l and ret_r):
            print("Frame drop.")
            continue

        # Detect corners for preview
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        
        found_l, corners_l = cv2.findChessboardCorners(gray_l, pattern_size, None)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, pattern_size, None)

        vis_l, vis_r = frame_l.copy(), frame_r.copy()
        
        if found_l:
            cv2.drawChessboardCorners(vis_l, pattern_size, corners_l, found_l)
        if found_r:
            cv2.drawChessboardCorners(vis_r, pattern_size, corners_r, found_r)

        # Status text
        status_l = "OK" if found_l else "--"
        status_r = "OK" if found_r else "--"
        cv2.putText(vis_l, f"Left: {status_l}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if found_l else (0, 0, 255), 2)
        cv2.putText(vis_r, f"Right: {status_r}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if found_r else (0, 0, 255), 2)

        combo = cv2.hconcat([vis_l, vis_r])
        cv2.imshow("Calibration Capture", combo)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key == ord(' '):  # SPACE
            if found_l and found_r:
                # Refine corners
                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
                
                # Save
                left_path = config.CALIB_LEFT_DIR / f"pair_{pair_count:03d}.png"
                right_path = config.CALIB_RIGHT_DIR / f"pair_{pair_count:03d}.png"
                
                cv2.imwrite(str(left_path), frame_l)
                cv2.imwrite(str(right_path), frame_r)
                
                pair_count += 1
                print(f"Saved pair {pair_count} (total)")
            else:
                print("Cannot save: Checkerboard not found in BOTH cameras.")

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
    print(f"\nCapture complete. Saved {pair_count} pairs to {config.CALIB_IMG_DIR}")

if __name__ == "__main__":
    main()