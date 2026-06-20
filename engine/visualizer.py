"""
Visualizer — Drawing utilities for overlaying analysis results on video frames.

Provides:
- Pose skeleton overlay (with phase colors)
- Ball trajectory and predicted path
- Bat swing path
- Metric HUD (real-time angle display)
- Phase labels
- Wagon wheel generation
- Session report charts
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64


# Phase color coding
PHASE_COLORS = {
    "stance": (200, 200, 200),       # grey
    "backlift": (255, 165, 0),        # orange
    "stride": (255, 255, 0),          # yellow
    "downswing": (0, 165, 255),       # orange-ish
    "impact": (0, 0, 255),            # RED (important)
    "follow_through": (0, 255, 0),    # green
    "recovery": (200, 200, 200),      # grey
    "unknown": (128, 128, 128),       # dark grey
}


class Visualizer:
    """Draws analysis overlays on video frames."""

    def __init__(self, batting_hand="right"):
        self.batting_hand = batting_hand

    def draw_phase_overlay(self, frame, phase, alpha=0.3):
        """Tint frame edges with phase color."""
        color = PHASE_COLORS.get(phase, (128, 128, 128))
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Draw colored border
        border_w = 8
        cv2.rectangle(overlay, (0, 0), (border_w, h), color, -1)  # left
        cv2.rectangle(overlay, (w - border_w, 0), (w, h), color, -1)  # right
        cv2.rectangle(overlay, (0, 0), (w, border_w), color, -1)  # top
        cv2.rectangle(overlay, (0, h - border_w), (w, h), color, -1)  # bottom

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def draw_metric_hud(self, frame, metrics, x=10, y=10):
        """
        Draw real-time metrics on frame.

        Args:
            metrics: dict of metric_name -> value
        """
        hud_lines = []
        label_map = {
            "front_knee_angle": "Front Knee",
            "back_knee_angle": "Back Knee",
            "front_elbow_angle": "Front Elbow",
            "back_elbow_angle": "Back Elbow",
            "shoulder_angle": "Shoulders",
            "spine_angle": "Spine Lean",
            "bat_speed_px": "Bat Speed",
            "phase": "Phase",
        }

        for key, label in label_map.items():
            if key in metrics and metrics[key] is not None:
                val = metrics[key]
                if isinstance(val, float):
                    if "angle" in key:
                        hud_lines.append(f"{label}: {val:.1f}°")
                    else:
                        hud_lines.append(f"{label}: {val:.1f}")
                else:
                    hud_lines.append(f"{label}: {val}")

        if not hud_lines:
            return frame

        # Draw semi-transparent background
        h = len(hud_lines) * 22 + 10
        cv2.rectangle(frame, (x - 5, y - 5), (x + 200, y + h),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 5, y - 5), (x + 200, y + h),
                      (255, 255, 255), 1)

        for i, line in enumerate(hud_lines):
            cv2.putText(frame, line, (x, y + 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

    def draw_ball_trajectory(self, frame, trajectory, color=(0, 255, 255),
                              trail_length=30):
        """Draw ball trajectory trail on frame."""
        if not trajectory:
            return frame

        # Draw only recent trail
        trail = trajectory[-trail_length:]
        for i in range(len(trail) - 1):
            pt1 = trail[i]
            pt2 = trail[i + 1]
            if pt1 and pt2:
                alpha = i / len(trail)
                trail_color = (int(color[0] * alpha), int(color[1] * alpha), 255)
                cv2.line(frame, pt1, pt2, trail_color, 2)

        # Draw current ball position
        if trail:
            last = trail[-1]
            cv2.circle(frame, last, 6, color, -1)
            cv2.circle(frame, last, 6, (255, 255, 255), 2)

        return frame

    def draw_swing_path(self, frame, swing_path, color=(255, 0, 255)):
        """Draw bat swing path on frame."""
        if not swing_path:
            return frame

        for i in range(len(swing_path) - 1):
            if swing_path[i] and swing_path[i + 1]:
                alpha = i / len(swing_path)
                trail_color = (int(color[0]), 255 - int(alpha * 200),
                               int(color[2]))
                cv2.line(frame, swing_path[i], swing_path[i + 1],
                         trail_color, 2)

        return frame

    def draw_bat_line(self, frame, bat_analyzer_result, color=(255, 0, 255)):
        """Draw inferred bat position as a line from hands to tip."""
        if not bat_analyzer_result:
            return frame

        hands_pos = bat_analyzer_result.get("hands_position")
        bat_tip = bat_analyzer_result.get("bat_tip")

        if hands_pos and bat_tip:
            hx, hy = int(hands_pos[0]), int(hands_pos[1])
            cv2.line(frame, (hx, hy), bat_tip, color, 3)
            # Draw circles at hands and tip
            cv2.circle(frame, (hx, hy), 5, (0, 255, 255), -1)
            cv2.circle(frame, bat_tip, 4, color, -1)

        return frame

    def draw_angle_arc(self, frame, p1, p2, p3, angle_deg, color=(0, 255, 0)):
        """Draw an arc showing the angle at p2 between p1-p2-p3."""
        if not all([p1, p2, p3]) or angle_deg is None:
            return frame

        # Calculate angle in radians
        a = np.arctan2(p1["pixel_y"] - p2["pixel_y"],
                       p1["pixel_x"] - p2["pixel_x"])
        b = np.arctan2(p3["pixel_y"] - p2["pixel_y"],
                       p3["pixel_x"] - p2["pixel_x"])

        # Draw arc
        radius = 30
        start_angle = float(np.degrees(a))
        end_angle = float(np.degrees(b))

        cv2.ellipse(frame, (p2["pixel_x"], p2["pixel_y"]),
                    (radius, radius), 0,
                    start_angle, end_angle, color, 2)

        # Draw angle text
        mid_angle = (start_angle + end_angle) / 2
        mid_rad = np.radians(mid_angle)
        text_x = int(p2["pixel_x"] + (radius + 15) * np.cos(mid_rad))
        text_y = int(p2["pixel_y"] + (radius + 15) * np.sin(mid_rad))
        cv2.putText(frame, f"{angle_deg:.0f}°", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame

    def draw_phase_text(self, frame, phase, frame_w):
        """Draw large phase label at top of frame."""
        color = PHASE_COLORS.get(phase, (128, 128, 128))
        text = phase.upper().replace("_", " ")

        cv2.putText(frame, text, (frame_w // 2 - 80, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame

    def create_angle_chart(self, angle_history, title="", labels=None):
        """
        Create a matplotlib line chart of angle over time.

        Returns base64 PNG string.
        """
        plt.figure(figsize=(10, 4))
        if labels:
            for i, (data, label) in enumerate(zip(angle_history, labels)):
                plt.plot(data, label=label, linewidth=2)
        else:
            for data in angle_history:
                plt.plot(data, linewidth=2)

        plt.xlabel("Frame")
        plt.ylabel("Angle (degrees)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_base64

    def create_session_charts(self, session_data):
        """
        Create comprehensive session analysis charts.

        Returns dict of chart_name -> base64 PNG.
        """
        charts = {}

        # 1. Joint angles over time
        if session_data.get("front_knee_history"):
            charts["knee_angles"] = self.create_angle_chart(
                [session_data.get("front_knee_history", []),
                 session_data.get("back_knee_history", [])],
                title="Knee Angles Through Session",
                labels=["Front Knee", "Back Knee"],
            )

        # 2. Bat speed over time
        if session_data.get("bat_speed_history"):
            plt.figure(figsize=(10, 3))
            plt.plot(session_data["bat_speed_history"], color='purple',
                     linewidth=2)
            plt.xlabel("Frame")
            plt.ylabel("Speed (px/frame)")
            plt.title("Bat Swing Speed")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            charts["bat_speed"] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

        # 3. Head stability
        if session_data.get("head_movement_history"):
            plt.figure(figsize=(10, 2))
            plt.plot(session_data["head_movement_history"],
                     color='orange', linewidth=1.5)
            plt.axhline(y=np.mean(session_data["head_movement_history"]),
                        color='r', linestyle='--', alpha=0.5)
            plt.xlabel("Frame")
            plt.ylabel("Movement")
            plt.title("Head Stability (lower = better)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            charts["head_stability"] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

        return charts

    @staticmethod
    def put_text(frame, text, position, color=(255, 255, 255),
                 size=0.5, thickness=1):
        """Helper to draw text with background."""
        x, y = position
        cv2.putText(frame, text, (x + 1, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness + 1)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
        return frame
