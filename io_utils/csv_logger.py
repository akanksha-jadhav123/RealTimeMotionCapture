# io_utils/csv_logger.py
import csv
import time
from pathlib import Path
import config

class CSVLogger:
    def __init__(self, filename=None):
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = config.LOG_DIR / f"3d_session_{timestamp}.csv"
        else:
            filename = Path(filename)
            
        self.filename = filename
        self.file = open(self.filename, mode='w', newline='')
        self.writer = csv.writer(self.file)
        
        # Header now includes 3D data
        self.writer.writerow([
            "timestamp", "frame_id", "landmark_name", 
            "x_2d", "y_2d", "x_3d", "y_3d", "z_3d"
        ])
        print(f"Logging 3D data to: {self.filename}")

    def log_frame(self, frame_id, timestamp, face_pts_2d, face_pts_3d, pose_pts_2d, pose_pts_3d):
        # Log face
        for name, (x2, y2) in (face_pts_2d or {}).items():
            x3, y3, z3 = face_pts_3d.get(name, (0, 0, 0))
            self.writer.writerow([timestamp, frame_id, f"face_{name}", x2, y2, x3, y3, z3])
            
        # Log pose
        for name, (x2, y2, vis) in (pose_pts_2d or {}).items():
            x3, y3, z3 = pose_pts_3d.get(name, (0, 0, 0))
            self.writer.writerow([timestamp, frame_id, f"pose_{name}", x2, y2, x3, y3, z3])

    def close(self):
        self.file.close()