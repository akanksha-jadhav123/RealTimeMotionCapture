# visualization/overlay.py
"""Draw landmarks and skeletons on frames."""

import cv2
import config


def draw_face(frame, pts):
    if pts is None:
        return
    for name, (x, y) in pts.items():
        cv2.circle(frame, (int(x), int(y)), 3, config.COLOR_FACE, -1)
        cv2.putText(frame, name, (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, config.COLOR_FACE, 1)


def draw_pose(frame, pts):
    if pts is None:
        return

    # Draw connections first (so joints land on top)
    for a, b in config.POSE_CONNECTIONS:
        if a in pts and b in pts:
            xa, ya, va = pts[a]
            xb, yb, vb = pts[b]
            if va > config.MIN_VISIBILITY and vb > config.MIN_VISIBILITY:
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                         config.COLOR_LINE, 2)

    # Draw joint points
    for name, (x, y, vis) in pts.items():
        if vis < config.MIN_VISIBILITY:
            continue
        cv2.circle(frame, (int(x), int(y)), 5, config.COLOR_POSE, -1)
        cv2.putText(frame, name, (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, config.COLOR_POSE, 1)


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS {fps:5.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_TEXT, 2)

def _draw_label(frame, text, x, y, color):
    """Draw text with a filled background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.35
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (int(x) - 1, int(y) - th - 2),
                  (int(x) + tw + 1, int(y) + baseline - 1),
                  (0, 0, 0), -1)
    cv2.putText(frame, text, (int(x), int(y)), font, scale, color, thickness)


def draw_face(frame, pts):
    if pts is None:
        return
    for name, (x, y) in pts.items():
        cv2.circle(frame, (int(x), int(y)), 3, config.COLOR_FACE, -1)
        _draw_label(frame, name, int(x) + 5, int(y) - 5, config.COLOR_FACE)


def draw_pose(frame, pts):
    if pts is None:
        return
    for a, b in config.POSE_CONNECTIONS:
        if a in pts and b in pts:
            xa, ya, va = pts[a]
            xb, yb, vb = pts[b]
            if va > config.MIN_VISIBILITY and vb > config.MIN_VISIBILITY:
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                         config.COLOR_LINE, 2)
    for name, (x, y, vis) in pts.items():
        if vis < config.MIN_VISIBILITY:
            continue
        cv2.circle(frame, (int(x), int(y)), 5, config.COLOR_POSE, -1)
        _draw_label(frame, name, int(x) + 6, int(y) - 6, config.COLOR_POSE)