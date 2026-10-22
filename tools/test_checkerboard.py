import cv2
import sys
from pathlib import Path

# Fix the path so it can find config.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

cap = cv2.VideoCapture(config.CAM_LEFT_INDEX, cv2.CAP_DSHOW)
pattern_size = (config.CHECKERBOARD_COLS, config.CHECKERBOARD_ROWS)

print(f"Looking for a {pattern_size[0]}x{pattern_size[1]} inner-corner grid.")
print("Hold the checkerboard in front of the camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if found:
        cv2.drawChessboardCorners(frame, pattern_size, corners, found)
        cv2.putText(frame, "FOUND!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "NOT FOUND", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
    cv2.imshow("Checkerboard Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
