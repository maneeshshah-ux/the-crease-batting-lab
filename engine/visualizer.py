"""
Visualizer — CLEAN cricket analysis overlays (Fox Sports inspired).

Design:
- Real-world units (cm, km/h, degrees) not pixels
- Traffic-light colours (green=good, amber=warning, red=problem)
- ONE clear visual per concept
- Weight transfer bar (coaching gold — kept front and centre)
- Balance beam near feet with percentage
- Speedometer with calibration note
- No skeleton lines — only essential coaching overlays
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

PHASE_COLORS = {
    "stance": (180, 180, 180),
    "backlift": (0, 165, 255),
    "stride": (0, 255, 255),
    "downswing": (0, 80, 255),
    "impact": (0, 0, 255),
    "follow_through": (0, 255, 0),
    "recovery": (180, 180, 180),
    "unknown": (100, 100, 100),
}

# Phase name shorthand for legend
PHASE_NAMES_SHORT = {
    "stance": "STANCE",
    "backlift": "LIFT",
    "stride": "STRIDE",
    "downswing": "SWING",
    "impact": "HIT",
    "follow_through": "FOLW",
    "recovery": "RECV",
    "unknown": "---",
}


class Visualizer:
    """Clean cricket analysis overlays."""

    def __init__(self, batting_hand="right", px_per_cm=None):
        self.batting_hand = batting_hand
        self.px_per_cm = px_per_cm  # calibration: pixels per cm (e.g. 1.2)
        self.session_peak_kmh = 0   # updated as we see swings
        self._calibration_note_shown = False

    # ════════════════════════════════════════════════════
    # 1. PHASE BAR  (top edge, thicker)
    # ════════════════════════════════════════════════════

    def draw_phase_bar(self, frame, phase):
        color = PHASE_COLORS.get(phase, (100, 100, 100))
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 8), color, -1)
        # Phase label on the bar
        label = PHASE_NAMES_SHORT.get(phase, "")
        if label:
            cv2.putText(frame, label, (6, 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return frame

    def draw_phase_legend(self, frame):
        """Small colour key — persistent, top-left (bigger)."""
        h, w = frame.shape[:2]
        lx, ly = 8, 16
        bw, bh = 130, 95
        cell_h = 14
        pad = 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (lx, ly), (lx + bw, ly + bh), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, "PHASES", (lx + 6, ly + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)
        for i, (name, ck) in enumerate([
            ("Stance", "stance"),
            ("Backlift", "backlift"),
            ("Stride", "stride"),
            ("Swing", "downswing"),
            ("Impact", "impact"),
            ("Follow", "follow_through"),
        ]):
            color = PHASE_COLORS.get(ck, (128, 128, 128))
            y = ly + 18 + i * (cell_h + pad)
            cv2.rectangle(frame, (lx + 6, y), (lx + 16, y + cell_h - 2), color, -1)
            cv2.putText(frame, name, (lx + 20, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (220, 220, 220), 1)
        return frame

    # ════════════════════════════════════════════════════
    # 2. HEAD STABILITY — traffic light (REAL cm)
    # ════════════════════════════════════════════════════

    HEAD_GREEN_PX = 3.0   # ~2.5 cm
    HEAD_AMBER_PX = 8.0   # ~6.6 cm
    HEAD_RED_PX = 15.0    # ~12.4 cm

    def draw_head_indicator(self, frame, head_movement_px, head_x, head_y):
        """Traffic-light dot near the batter's head. Shows cm when calibrated."""
        h, w = frame.shape[:2]

        if head_x is None or head_y is None:
            cx, cy = w - 35, 35
        else:
            cx = min(w - 30, max(30, int(head_x)))
            cy = max(30, int(head_y) - 45)

        # Convert to cm if we have calibration
        if self.px_per_cm and self.px_per_cm > 0:
            head_cm = head_movement_px / self.px_per_cm
            unit = "cm"
        else:
            head_cm = head_movement_px  # fallback: show px
            unit = "px"

        # Color
        if head_movement_px <= self.HEAD_GREEN_PX:
            color = (0, 200, 0)
            status = "STILL"
        elif head_movement_px <= self.HEAD_AMBER_PX:
            color = (0, 200, 255)
            status = "OK"
        else:
            color = (0, 0, 200)
            status = "MOVE"

        # Larger glow
        cv2.circle(frame, (cx, cy), 22, color, 2)
        inner = (color[0] // 3, color[1] // 3, color[2] // 3)
        cv2.circle(frame, (cx, cy), 14, inner, -1)
        cv2.circle(frame, (cx, cy), 14, color, 1)

        # Label — bigger text
        cv2.putText(frame, f"{head_cm:.1f}{unit}", (cx - 22, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Status text beside it
        cv2.putText(frame, status, (cx + 18, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        return frame

    # ════════════════════════════════════════════════════
    # 3. BALANCE — percentage display near feet
    # ════════════════════════════════════════════════════

    def draw_balance_level(self, frame, spine_angle, knee_angle):
        """
        Balance display as a percentage-based indicator near the batter's feet.
        Shows:
          - Spine lean as a percentage (ideal ~50%)
          - Knee bend as a percentage (ideal ~50%)
        Uses a compact vertical layout positioned near the bottom-left.
        """
        h, w = frame.shape[:2]

        # Position: near feet on the left side
        base_x = 10
        base_y = h - 90

        # ── Spine lean gauge ──
        spine_pct = 50  # default center
        if spine_angle is not None:
            # 150° = 0% (leaning forward), 180° = 100% (leaning back)
            # Ideal: 165-170° = ~50-67%
            spine_pct = max(0, min(100, (spine_angle - 150) / 0.30))

        # ── Knee bend gauge ──
        knee_pct = 50
        if knee_angle is not None:
            # 120° = 100% (deep bend), 180° = 0% (straight leg)
            # Ideal: 135-145° = ~58-75%
            knee_pct = max(0, min(100, (180 - knee_angle) / 0.60))

        # Background panel
        pw, ph = 115, 68
        px, py = base_x, base_y
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (40, 40, 40), 1)

        # Title
        cv2.putText(frame, "BALANCE", (px + 6, py + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # ── Spine bar ──
        bar_y1 = py + 18
        bar_w = 95
        bar_h = 10
        bar_x = px + 10
        # Green zone (40-70%)
        cv2.rectangle(frame, (bar_x + int(bar_w * 0.4), bar_y1),
                      (bar_x + int(bar_w * 0.7), bar_y1 + bar_h), (0, 60, 0), -1)
        # Fill
        fill_w = int(bar_w * spine_pct / 100)
        fill_color = (0, 200, 0) if 40 <= spine_pct <= 70 else (0, 200, 255) if 25 <= spine_pct <= 85 else (0, 0, 200)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + fill_w, bar_y1 + bar_h), fill_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y1), (bar_x + bar_w, bar_y1 + bar_h), (100, 100, 100), 1)
        cv2.putText(frame, f"SPINE {spine_pct:.0f}%", (bar_x + bar_w + 4, bar_y1 + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1)

        # ── Knee bar ──
        bar_y2 = py + 34
        # Green zone (50-80%)
        cv2.rectangle(frame, (bar_x + int(bar_w * 0.5), bar_y2),
                      (bar_x + int(bar_w * 0.8), bar_y2 + bar_h), (0, 60, 0), -1)
        fill_w2 = int(bar_w * knee_pct / 100)
        fill_color2 = (0, 200, 0) if 50 <= knee_pct <= 80 else (0, 200, 255) if 30 <= knee_pct <= 90 else (0, 0, 200)
        if fill_w2 > 0:
            cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + fill_w2, bar_y2 + bar_h), fill_color2, -1)
        cv2.rectangle(frame, (bar_x, bar_y2), (bar_x + bar_w, bar_y2 + bar_h), (100, 100, 100), 1)
        cv2.putText(frame, f"KNEE {knee_pct:.0f}%", (bar_x + bar_w + 4, bar_y2 + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180, 180, 180), 1)

        return frame

    # ════════════════════════════════════════════════════
    # 4. BAT SPEED — speedometer with player benchmarks
    # ════════════════════════════════════════════════════

    # Elite player benchmarks (bat speed km/h, approximate from known data)
    PLAYER_BENCHMARKS = [
        ("Perry", 115, (255, 100, 100)),    # Ellyse Perry elite women's
        ("Kohli", 130, (200, 130, 50)),     # Virat Kohli
        ("de Villiers", 150, (0, 180, 255)),# AB de Villiers peak
        ("Russell", 155, (0, 100, 200)),    # Andre Russell big hitting
    ]

    def draw_speedometer(self, frame, speed_kmh=None, calibration_available=False,
                         peak_session_kmh=None):
        """
        Analog speedometer with player benchmarks, zone colours,
        and session peak marker. Shows women's and men's elite references.
        """
        h, w = frame.shape[:2]
        cx, cy = w - 90, h - 100
        radius = 55
        max_speed = 160  # scale to 160 km/h (covers all benchmarks)

        # ── Arc ──
        for angle in range(180, 361):
            rad = np.radians(angle)
            x = int(cx + radius * np.cos(rad))
            y = int(cy + radius * np.sin(rad))
            pct = (angle - 180) / 180
            spd = pct * max_speed
            if spd < 35:
                col = (0, 120, 0)       # green: club
            elif spd < 65:
                col = (0, 180, 255)     # amber: county/pro
            elif spd < 100:
                col = (0, 100, 200)     # orange: domestic first-class
            else:
                col = (0, 40, 180)      # red: international elite
            cv2.circle(frame, (x, y), 2, col, -1)

        # ── Tick marks ──
        for sv in range(0, max_speed + 1, 10):
            pct = sv / max_speed
            ang = 180 + pct * 180
            rad = np.radians(ang)
            ir, or_ = radius - 10, radius - 2
            x1 = int(cx + ir * np.cos(rad))
            y1 = int(cy + ir * np.sin(rad))
            x2 = int(cx + or_ * np.cos(rad))
            y2 = int(cy + or_ * np.sin(rad))
            cv2.line(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)

        # ── Zone labels ──
        for label, spd, y_off, clr in [
            ("CLUB", 20, -14, (100, 180, 100)),
            ("PRO", 50, -14, (100, 200, 200)),
            ("INT'L", 110, -14, (100, 130, 200)),
        ]:
            pct = min(1, spd / max_speed)
            ang = 180 + pct * 180
            rad = np.radians(ang)
            lx = int(cx + (radius + 14) * np.cos(rad))
            ly = int(cy + (radius + 14) * np.sin(rad))
            cv2.putText(frame, label, (lx - 14, ly + y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, clr, 1)

        # ── Player benchmarks ──
        for pname, pspeed, pcolor in self.PLAYER_BENCHMARKS:
            pct = min(1, pspeed / max_speed)
            ang = 180 + pct * 180
            rad = np.radians(ang)
            mr = radius - 16
            mx = int(cx + mr * np.cos(rad))
            my = int(cy + mr * np.sin(rad))
            cv2.circle(frame, (mx, my), 2, pcolor, -1)
            # Label
            lx2 = int(cx + (radius + 14) * np.cos(rad))
            ly2 = int(cy + (radius + 14) * np.sin(rad))
            # Offset label to avoid overlap
            label_offset = -8 if pspeed < 140 else 0
            cv2.putText(frame, pname, (lx2 - 10, ly2 + label_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, pcolor, 1)

        # ── Needle ──
        if speed_kmh and speed_kmh > 0:
            pct = min(1, speed_kmh / max_speed)
            ang = 180 + pct * 180
            rad = np.radians(ang)
            nl = radius - 8
            nx = int(cx + nl * np.cos(rad))
            ny = int(cy + nl * np.sin(rad))
            cv2.line(frame, (cx, cy), (nx, ny), (255, 100, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 200, 0), -1)

        # ── Speed readout ──
        if speed_kmh:
            cv2.putText(frame, f"{speed_kmh:.0f}", (cx - 20, cy + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "--", (cx - 10, cy + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(frame, "km/h", (cx - 24, cy + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # ── Calibration badge ──
        if calibration_available:
            cv2.putText(frame, "CAL", (cx + radius - 14, cy + radius + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (80, 180, 80), 1)

        # ── Session peak marker on gauge ──
        if peak_session_kmh and peak_session_kmh > 0:
            pk = min(1, peak_session_kmh / max_speed)
            ang = 180 + pk * 180
            rad = np.radians(ang)
            mr = radius - 14
            mx = int(cx + mr * np.cos(rad))
            my = int(cy + mr * np.sin(rad))
            cv2.circle(frame, (mx, my), 3, (255, 255, 0), -1)

        return frame

    def draw_bat_speed_overlay(self, frame, speed_kmh, impact_frame=False):
        """
        Large, prominent bat speed text overlay.
        Shows speed when swinging; bigger at impact.
        """
        h, w = frame.shape[:2]
        if not speed_kmh or speed_kmh <= 0:
            return frame

        if impact_frame:
            # Big impact popup
            text = f"{speed_kmh:.0f} km/h"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(text, font, 1.2, 3)
            tx = (w - tw) // 2
            ty = h // 2 - 40
            # Background bar
            cv2.rectangle(frame, (tx - 10, ty - th - 6),
                          (tx + tw + 10, ty + 6), (0, 0, 0, 150), -1)
            cv2.putText(frame, text, (tx, ty), font, 1.2,
                        (255, 200, 0), 3)
            cv2.putText(frame, "BAT SPEED", (tx + 4, ty - th - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            # Top-right corner during swings
            text = f"BAT {speed_kmh:.0f} km/h"
            cv2.putText(frame, text, (w - 180, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        return frame

    # ════════════════════════════════════════════════════
    # 5. WEIGHT TRANSFER — clear BAR (Fox Focus style)
    # ════════════════════════════════════════════════════

    def draw_weight_transfer(self, frame, landmarks):
        """
        Weight transfer BAR — the most important coaching feedback.
        Shows front foot / back foot weight distribution as a percentage bar.
        Positioned at bottom-centre, prominent.
        """
        h, w = frame.shape[:2]

        if not landmarks:
            return frame

        l_hip = landmarks.get("LEFT_HIP", {})
        r_hip = landmarks.get("RIGHT_HIP", {})
        if not (l_hip.get("pixel_x") and r_hip.get("pixel_x")):
            return frame

        com_x = (l_hip["pixel_x"] + r_hip["pixel_x"]) / 2

        # Determine front/back foot
        l_foot = landmarks.get("LEFT_FOOT_INDEX", {})
        r_foot = landmarks.get("RIGHT_FOOT_INDEX", {})
        l_ankle = landmarks.get("LEFT_ANKLE", {})
        r_ankle = landmarks.get("RIGHT_ANKLE", {})

        if l_foot.get("pixel_y") and r_foot.get("pixel_y"):
            if l_foot["pixel_y"] > r_foot["pixel_y"]:
                fx, bx = l_foot["pixel_x"], r_foot["pixel_x"]
            else:
                fx, bx = r_foot["pixel_x"], l_foot["pixel_x"]
        elif l_ankle.get("pixel_y") and r_ankle.get("pixel_y"):
            if l_ankle["pixel_y"] > r_ankle["pixel_y"]:
                fx, bx = l_ankle["pixel_x"], r_ankle["pixel_x"]
            else:
                fx, bx = r_ankle["pixel_x"], l_ankle["pixel_x"]
        else:
            return frame

        fr = abs(fx - bx)
        if fr < 10:
            return frame

        pct = (com_x - min(fx, bx)) / fr
        pct = max(0, min(1, pct))
        front_pct = pct * 100
        back_pct = 100 - front_pct

        # ── Draw bar (bottom-center, prominent) ──
        bar_w = min(260, w - 40)
        bar_h = 24
        bar_x = (w - bar_w) // 2
        bar_y = h - 58

        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (25, 25, 25), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), 1)

        # Fill
        fill_w = int(bar_w * pct)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          (15, 120, 230), -1)  # orange for front
        if bar_w - fill_w > 0:
            cv2.rectangle(frame, (bar_x + fill_w, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          (200, 130, 50), -1)  # blue for back

        # Centre marker
        mid_x = bar_x + bar_w // 2
        cv2.line(frame, (mid_x, bar_y - 4), (mid_x, bar_y + bar_h + 4),
                 (200, 200, 200), 1)

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"BACK {back_pct:.0f}%", (bar_x + 8, bar_y + 17),
                    font, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"FRONT {front_pct:.0f}%",
                    (bar_x + bar_w - 85, bar_y + 17), font, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "WEIGHT", (bar_x + bar_w // 2 - 28, bar_y - 6),
                    font, 0.35, (150, 150, 150), 1)

        return frame

    # ════════════════════════════════════════════════════
    # 6. BALL TRAJECTORY — removed (user feedback: distracting)
    # ════════════════════════════════════════════════════

    # Ball trajectory drawing has been removed per user request.
    # All yellow line overlays have been removed.

    # ════════════════════════════════════════════════════
    # 7. BAT LINE — subtle, only at impact
    # ════════════════════════════════════════════════════

    def draw_bat_line(self, frame, bat_data):
        if not bat_data or not bat_data.get("has_swing_data"):
            return frame
        tip = bat_data.get("bat_tip")
        hands = bat_data.get("hands_position")
        if tip and hands:
            cv2.line(frame, (int(hands[0]), int(hands[1])),
                     (int(tip[0]), int(tip[1])), (180, 180, 120), 1, cv2.LINE_AA)
            # Smaller label, more transparent
            cv2.putText(frame, "BAT", (int(tip[0]) + 4, int(tip[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (180, 180, 120), 1)
        return frame

    # ════════════════════════════════════════════════════
    # 8. SWING PATH — thin, grey, goes away quickly
    # ════════════════════════════════════════════════════

    def draw_swing_path(self, frame, path_history, phase=None):
        if not path_history:
            return frame
        valid = [(int(p[0]), int(p[1])) for p in path_history[-30:] if p]
        if len(valid) < 3:
            return frame
        pts = np.array(valid, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], False, (100, 100, 100), 1, cv2.LINE_AA)
        # Only draw very small dots, few of them
        for pt in valid[::5]:
            cv2.circle(frame, pt, 1, (120, 120, 120), -1)
        return frame

    # ════════════════════════════════════════════════════
    # 9. WATERMARK
    # ════════════════════════════════════════════════════

    def draw_watermark(self, frame, text="CREASE Batting Lab"):
        h, w = frame.shape[:2]
        cv2.putText(frame, text, (w // 2 - 60, h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (60, 60, 60), 1)
        return frame

    # ════════════════════════════════════════════════════
    # 10. CHARTS
    # ════════════════════════════════════════════════════

    def create_angle_chart(self, angle_history, title="", labels=None):
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
        charts = {}
        if session_data.get("front_knee_history"):
            charts["knee_angles"] = self.create_angle_chart(
                [session_data.get("front_knee_history", []),
                 session_data.get("back_knee_history", [])],
                title="Knee Angles Through Session",
                labels=["Front Knee", "Back Knee"],
            )
        if session_data.get("bat_speed_history"):
            speeds = session_data["bat_speed_history"]
            cal = session_data.get("bat_calibration")
            if cal and cal.get("px_per_m"):
                ppm = cal["px_per_m"]
                fps = session_data.get("fps", 30)
                data = [s * fps / ppm * 3.6 for s in speeds if s > 0]
                yl = "Speed (km/h)"
            else:
                yl = "Speed (px/frame)"
                data = speeds
            plt.figure(figsize=(10, 3))
            plt.plot(data, color='purple', linewidth=2)
            plt.xlabel("Frame")
            plt.ylabel(yl)
            plt.title("Bat Swing Speed")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            charts["bat_speed"] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        if session_data.get("head_movement_history"):
            hd = session_data["head_movement_history"]
            plt.figure(figsize=(10, 2))
            plt.plot(hd, color='orange', linewidth=1.5)
            plt.axhline(y=np.mean(hd), color='r', linestyle='--', alpha=0.5)
            plt.axhline(y=self.HEAD_GREEN_PX, color='g', linestyle=':', alpha=0.5, label='Good')
            plt.axhline(y=self.HEAD_AMBER_PX, color='y', linestyle=':', alpha=0.5, label='Warning')
            plt.axhline(y=self.HEAD_RED_PX, color='r', linestyle=':', alpha=0.5, label='Problem')
            plt.xlabel("Frame")
            plt.ylabel("Pixels")
            plt.title("Head Stability (lower = better)")
            plt.legend(fontsize=8)
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
        x, y = position
        cv2.putText(frame, text, (x + 1, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness + 1)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)
        return frame
