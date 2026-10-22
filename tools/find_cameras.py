# tools/find_cameras.py
"""
Camera discovery and verification tool.
Scans indices 0..N-1 across DSHOW, MSMF, ANY. Verifies actual frame
delivery, then opens the two best cameras and previews them live.

Usage (from project root, with venv active):
    python tools/find_cameras.py
Press q in the preview window to quit.
"""

import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def named_backend(name):
    return {
        "dshow":        cv2.CAP_DSHOW,
        "msmf":         cv2.CAP_MSMF,
        "v4l2":         cv2.CAP_V4L2,
        "avfoundation": cv2.CAP_AVFOUNDATION,
        "any":          cv2.CAP_ANY,
    }.get(name.lower(), cv2.CAP_ANY)


def backend_name(flag):
    for nm, val in [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF",  cv2.CAP_MSMF),
        ("V4L2",  cv2.CAP_V4L2),
        ("AVFOUNDATION", cv2.CAP_AVFOUNDATION),
        ("ANY",   cv2.CAP_ANY),
    ]:
        if flag == val:
            return nm
    return "BACKEND(" + str(flag) + ")"


def probe_index(idx, backends):
    for be in backends:
        cap = cv2.VideoCapture(idx, be)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          config.FPS_TARGET)
        ret, frame = cap.read()
        if ret and frame is not None:
            info = {
                "idx":     idx,
                "backend": be,
                "width":   int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height":  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps":     float(cap.get(cv2.CAP_PROP_FPS)),
            }
            cap.release()
            return info
        cap.release()
    return None


def scan_all(max_idx=6):
    backends = []
    for name in (config.CAM_BACKEND_NAME, "msmf", "any"):
        flag = named_backend(name)
        if flag not in backends:
            backends.append(flag)

    print(" Backend priority: " + str([backend_name(b) for b in backends]))
    print()

    found = []
    for idx in range(max_idx):
        print(" Probing index " + str(idx) + " ...", end=" ", flush=True)
        info = probe_index(idx, backends)
        if info:
            print("OK  ->  " + backend_name(info["backend"]) + "  "
                  + str(info["width"]) + "x" + str(info["height"])
                  + " @ " + format(info["fps"], ".1f") + " fps")
            found.append(info)
        else:
            print("no camera")
    return found


def open_from_info(info):
    cap = cv2.VideoCapture(info["idx"], info["backend"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.FPS_TARGET)
    return cap


def preview_pair(cam_l, cam_r):
    cap_l = open_from_info(cam_l)
    cap_r = open_from_info(cam_r)

    if not (cap_l.isOpened() and cap_r.isOpened()):
        print("!! Failed to open pair simultaneously.")
        cap_l.release()
        cap_r.release()
        return

    win = "Stereo Preview  -  press q to quit"
    prev_t = time.time()
    fps = 0.0
    frame_count = 0

    while True:
        ok_l, f_l = cap_l.read()
        ok_r, f_r = cap_r.read()
        if not (ok_l and ok_r):
            print("!! Frame drop - one camera stopped delivering frames")
            break

        f_l = cv2.resize(f_l, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        f_r = cv2.resize(f_r, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        frame_count += 1

        cv2.putText(f_l, "LEFT", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(f_r, "RIGHT", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(f_l, "FPS " + format(fps, "5.1f"), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(f_r, "frames " + str(frame_count), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        combo = cv2.hconcat([f_l, f_r])
        cv2.imshow(win, combo)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()
    print(" Preview closed after " + str(frame_count)
          + " frames. Final FPS ~ " + format(fps, ".1f"))


def main():
    print("=" * 64)
    print(" Camera discovery & stereo preview")
    print("=" * 64)
    print(" Config backend:      " + config.CAM_BACKEND_NAME)
    print(" Config indices:      L=" + str(config.CAM_LEFT_INDEX)
          + "  R=" + str(config.CAM_RIGHT_INDEX))
    print(" Target resolution:   " + str(config.FRAME_WIDTH) + "x"
          + str(config.FRAME_HEIGHT) + " @ " + str(config.FPS_TARGET) + " fps")
    print()

    print("-- STEP 1: Scanning camera indices 0..5 --")
    cameras = scan_all(max_idx=6)
    print()

    if len(cameras) == 0:
        print("!! No cameras found at all.")
        print("   Check Windows Camera app. Close Teams/Zoom/browsers.")
        return

    if len(cameras) == 1:
        print("!! Only ONE camera found. Stereo requires two.")
        print("   1. Plug USB webcam into a DIFFERENT USB port.")
        print("   2. Close every other app using the camera.")
        print("   3. Check Device Manager -> Cameras.")
        print("   4. Run: Get-PnpDevice -Class Camera,Image -Status OK")
        return

    print("-- STEP 2: Selecting left/right pair --")
    cfg_left  = next((c for c in cameras if c["idx"] == config.CAM_LEFT_INDEX),  None)
    cfg_right = next((c for c in cameras if c["idx"] == config.CAM_RIGHT_INDEX), None)

    if cfg_left and cfg_right and cfg_left["idx"] != cfg_right["idx"]:
        cam_l, cam_r = cfg_left, cfg_right
        print(" Using config.py pair: L=idx" + str(cfg_left["idx"])
              + "  R=idx" + str(cfg_right["idx"]))
    else:
        cam_l, cam_r = cameras[0], cameras[1]
        print(" config.py indices unavailable - auto-paired:")
        print("   L=idx" + str(cam_l["idx"]) + " (" + backend_name(cam_l["backend"]) + ")")
        print("   R=idx" + str(cam_r["idx"]) + " (" + backend_name(cam_r["backend"]) + ")")

    if (cam_l["width"], cam_l["height"]) != (cam_r["width"], cam_r["height"]):
        print("!! Resolution mismatch:")
        print("   L: " + str(cam_l["width"]) + "x" + str(cam_l["height"]))
        print("   R: " + str(cam_r["width"]) + "x" + str(cam_r["height"]))
    print()

    print("-- STEP 3: Live preview --")
    print(" Hold up a hand and verify it appears in BOTH sides.")
    print(" Press q in the window to quit.")
    print()
    preview_pair(cam_l, cam_r)

    print()
    print("=" * 64)
    print(" Summary - copy these into config.py")
    print("=" * 64)
    print(" CAM_LEFT_INDEX   = " + str(cam_l["idx"]))
    print(" CAM_RIGHT_INDEX  = " + str(cam_r["idx"]))
    print(" CAM_BACKEND_NAME = \"" + config.CAM_BACKEND_NAME + "\"")
    print()
    print(" Detected cameras:")
    for c in cameras:
        marker = " <- LEFT " if c is cam_l else (" <- RIGHT" if c is cam_r else "")
        print("   idx " + str(c["idx"]) + ":  "
              + str(c["width"]) + "x" + str(c["height"]) + " @ "
              + format(c["fps"], ".1f") + " fps  via "
              + backend_name(c["backend"]) + marker)
    print()
    print(" Next: measure physical baseline in cm, set BASELINE_M.")


if __name__ == "__main__":
    main()
