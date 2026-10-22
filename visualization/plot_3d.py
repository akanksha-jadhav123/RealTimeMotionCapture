import csv
import glob
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def get_latest_log():
    files = glob.glob(str(config.LOG_DIR / "3d_session_*.csv"))
    if not files:
        raise FileNotFoundError("No 3D logs found. Run the tracker first.")
    return max(files, key=lambda x: Path(x).stat().st_mtime)

def main():
    log_file = get_latest_log()
    print(f"Plotting 3D data from: {log_file}")

    targets = ["face_nose_tip", "pose_left_wrist", "pose_right_wrist"]
    data = {name: {"t": [], "x": [], "y": [], "z": []} for name in targets}

    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["landmark_name"]
            if name in targets:
                data[name]["t"].append(float(row["timestamp"]) / 1000.0)
                data[name]["x"].append(float(row["x_3d"]))
                data[name]["y"].append(float(row["y_3d"]))
                data[name]["z"].append(float(row["z_3d"]))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle('3D Landmark Movement (World Coordinates)')

    colors = {"face_nose_tip": "green", "pose_left_wrist": "blue", "pose_right_wrist": "red"}

    for name in targets:
        if data[name]["t"]:
            ax1.plot(data[name]["t"], data[name]["x"], label=name, color=colors[name])
            ax2.plot(data[name]["t"], data[name]["y"], label=name, color=colors[name])
            ax3.plot(data[name]["t"], data[name]["z"], label=name, color=colors[name])

    ax1.set_ylabel('X (meters)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_ylabel('Y (meters)')
    ax2.legend()
    ax2.grid(True)

    ax3.set_ylabel('Z (meters)')
    ax3.set_xlabel('Time (seconds)')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    out_file = config.LOG_DIR / "3d_movement_plot.png"
    plt.savefig(out_file)
    print(f"Plot saved to: {out_file}")
    plt.show()

if __name__ == "__main__":
    main()