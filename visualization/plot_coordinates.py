# visualization/plot_coordinates.py
"""
Reads the latest CSV log and plots X/Y coordinates over time.
"""
import csv
import glob
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def get_latest_log():
    """Find the most recently created CSV in the logs directory."""
    list_of_files = glob.glob(str(config.LOG_DIR / "*.csv"))
    if not list_of_files:
        raise FileNotFoundError(f"No CSV logs found in {config.LOG_DIR}")
    latest_file = max(list_of_files, key=lambda x: Path(x).stat().st_mtime)
    return latest_file

def main():
    log_file = get_latest_log()
    print(f"Plotting data from: {log_file}")

    # We will track these specific landmarks for the plot
    targets = ["face_nose_tip", "pose_left_wrist", "pose_right_wrist"]
    
    # Dictionary to hold data: { "landmark_name": {"x": [], "y": [], "t": []} }
    data = {name: {"x": [], "y": [], "t": []} for name in targets}

    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["landmark_name"]
            if name in targets:
                data[name]["x"].append(float(row["x_px"]))
                data[name]["y"].append(float(row["y_px"]))
                data[name]["t"].append(float(row["timestamp"]) / 1000.0) # convert to seconds

    # Create a figure with two subplots (X and Y)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Landmark Movement Over Time')

    colors = {"face_nose_tip": "green", "pose_left_wrist": "blue", "pose_right_wrist": "red"}

    for name in targets:
        if data[name]["t"]:
            # Plot X coordinates
            ax1.plot(data[name]["t"], data[name]["x"], label=name, color=colors[name])
            # Plot Y coordinates (inverted because Y=0 is top of image)
            ax2.plot(data[name]["t"], data[name]["y"], label=name, color=colors[name])

    ax1.set_ylabel('X Coordinate (pixels)')
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Y Coordinate (pixels)')
    ax2.invert_yaxis() # Match OpenCV image coordinates
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    # Save the plot image for the report
    out_file = config.LOG_DIR / "movement_plot.png"
    plt.savefig(out_file)
    print(f"Plot saved to: {out_file}")
    
    plt.show()

if __name__ == "__main__":
    main()