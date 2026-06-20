"""
Bat Analyzer — Infers bat position, swing path and bat metrics from pose data.

Since we don't have a bat sensor, we infer bat position from hand landmarks:
- The bat handle is between the two hands (grip position)
- The blade extends from the bottom hand toward the ground/direction of swing
- Swing path = trajectory of bottom wrist/hand + inferred bat tip
"""

import numpy as np
from collections import deque


class BatAnalyzer:
    """
    Analyzes bat swing from pose landmarks.

    Bat model:
    - Top hand (non-dominant): LEFT_WRIST for right-handed batter
    - Bottom hand (dominant): RIGHT_WRIST for right-handed batter
    - Bat tip: Extrapolated from bottom hand position + swing direction
    """

    # Typical cricket bat length: ~96.5 cm
    # In normalized coordinates, we use a scaling factor
    BAT_LENGTH_FACTOR = 0.12  # ~12% of frame height

    def __init__(self, batting_hand="right", history_frames=60):
        """
        Args:
            batting_hand: "right" or "left" handed batter
            history_frames: swing history length
        """
        self.batting_hand = batting_hand
        self.history = deque(maxlen=history_frames)
        self.swing_phases = []  # detected swing events

        # Hand assignment based on batting hand
        if batting_hand == "right":
            self.top_hand = "LEFT_WRIST"
            self.bottom_hand = "RIGHT_WRIST"
            self.front_shoulder = "LEFT_SHOULDER"
            self.back_shoulder = "RIGHT_SHOULDER"
        else:
            self.top_hand = "RIGHT_WRIST"
            self.bottom_hand = "LEFT_WRIST"
            self.front_shoulder = "RIGHT_SHOULDER"
            self.back_shoulder = "LEFT_SHOULDER"

    def infer_bat_tip(self, top_wrist, bottom_wrist, frame_h, frame_w):
        """
        Infer bat tip position from hand positions.

        The bat is held with top hand at the handle end and
        bottom hand lower down. The blade extends forward/downward
        from the bottom hand.

        Returns (tip_x, tip_y) in pixel coordinates.
        """
        if not top_wrist or not bottom_wrist:
            return None

        # Vector from top hand to bottom hand (along the handle)
        handle_dx = bottom_wrist["pixel_x"] - top_wrist["pixel_x"]
        handle_dy = bottom_wrist["pixel_y"] - top_wrist["pixel_y"]
        handle_len = np.sqrt(handle_dx**2 + handle_dy**2)

        if handle_len < 1:
            return None

        # Normalize handle direction
        handle_dx_n = handle_dx / handle_len
        handle_dy_n = handle_dy / handle_len

        # The blade continues from bottom hand in same direction as handle,
        # but with a slight downward angle (natural bat position)
        # Bat length in pixels (proportional to frame)
        bat_len = int(frame_h * self.BAT_LENGTH_FACTOR)

        # Add a slight angle for the blade relative to handle
        # (typically 10-20 degrees open)
        angle_offset = np.radians(15)  # blade opens from handle line
        cos_a = np.cos(angle_offset)
        sin_a = np.sin(angle_offset)

        # Rotate direction vector
        blade_dx = handle_dx_n * cos_a - handle_dy_n * sin_a
        blade_dy = handle_dx_n * sin_a + handle_dy_n * cos_a

        tip_x = int(bottom_wrist["pixel_x"] + blade_dx * bat_len)
        tip_y = int(bottom_wrist["pixel_y"] + blade_dy * bat_len)

        return (tip_x, tip_y)

    def analyze_swing(self, landmarks, frame_h, frame_w, frame_idx=0):
        """
        Analyze bat swing for current frame.

        Returns dict with swing metrics.
        """
        result = {
            "frame": frame_idx,
            "has_swing_data": False,
            "bat_angle_deg": None,
            "bat_speed_px": 0,
            "bat_lift_height": None,
            "hands_position": None,
            "bat_tip": None,
            "swing_direction": None,
        }

        top_wrist = landmarks.get(self.top_hand)
        bottom_wrist = landmarks.get(self.bottom_hand)

        if not top_wrist or not bottom_wrist:
            return result

        # Position of hands (midpoint)
        hand_mid_x = (top_wrist["pixel_x"] + bottom_wrist["pixel_x"]) / 2
        hand_mid_y = (top_wrist["pixel_y"] + bottom_wrist["pixel_y"]) / 2
        result["hands_position"] = (hand_mid_x, hand_mid_y)

        # Infer bat tip
        bat_tip = self.infer_bat_tip(top_wrist, bottom_wrist, frame_h, frame_w)
        result["bat_tip"] = bat_tip

        if bat_tip:
            # Bat angle relative to horizontal (0° = horizontal, 90° = vertical up)
            bat_dx = bat_tip[0] - hand_mid_x
            bat_dy = bat_tip[1] - hand_mid_y
            bat_angle = np.degrees(np.arctan2(-bat_dy, bat_dx))  # negate y for screen coords
            # Normalize to [0, 360)
            bat_angle = bat_angle % 360
            result["bat_angle_deg"] = float(bat_angle)

            # Bat lift height: how high the bat tip is relative to shoulder
            shoulder = landmarks.get(self.front_shoulder)
            if shoulder:
                bat_lift = shoulder["pixel_y"] - bat_tip[1]
                result["bat_lift_height"] = max(0, int(bat_lift))

        # Determine swing direction from velocity of bottom wrist
        if len(self.history) > 1:
            prev = self.history[-1]
            if prev.get("hands_position"):
                dx = hand_mid_x - prev["hands_position"][0]
                dy = hand_mid_y - prev["hands_position"][1]
                speed = np.sqrt(dx**2 + dy**2)
                result["bat_speed_px"] = float(speed)

                if speed > 3:
                    angle = np.degrees(np.arctan2(-dy, dx)) % 360
                    if 315 <= angle or angle < 45:
                        result["swing_direction"] = "right"  # moving right (forward)
                    elif 135 <= angle < 225:
                        result["swing_direction"] = "left"  # moving left (back)
                    elif 45 <= angle < 135:
                        result["swing_direction"] = "up"  # lifting bat
                    elif 225 <= angle < 315:
                        result["swing_direction"] = "down"  # coming down
                    else:
                        result["swing_direction"] = "unknown"

        result["has_swing_data"] = True

        # Store in history
        self.history.append(result)

        return result

    def get_swing_path(self):
        """Get the full swing path as a list of bat tip positions."""
        return [h.get("bat_tip") for h in self.history if h.get("bat_tip")]

    def get_swing_velocity_curve(self):
        """Get bat speed across the swing history."""
        return [h.get("bat_speed_px", 0) for h in self.history]

    def detect_backlift_peak(self):
        """
        Find the frame where backlift was highest (peak of bat lift).

        Returns frame index and height.
        """
        peak_frame = 0
        peak_height = 0
        for h in self.history:
            height = h.get("bat_lift_height", 0)
            if height and height > peak_height:
                peak_height = height
                peak_frame = h["frame"]
        return {"frame": peak_frame, "height_px": peak_height}

    def detect_downswing_start(self):
        """
        Detect the frame where downswing begins (transition from up to down).
        Uses change in bat velocity direction.
        """
        if len(self.history) < 3:
            return None

        for i in range(1, len(self.history)):
            prev_dir = self.history[i - 1].get("swing_direction")
            curr_dir = self.history[i].get("swing_direction")
            if prev_dir == "up" and curr_dir == "down":
                return self.history[i]["frame"]

        return None

    def estimate_bat_speed_kmh(self, fps, calibration_mm_per_pixel=None):
        """
        Estimate bat speed in km/h.

        Args:
            fps: video frame rate
            calibration_mm_per_pixel: physical calibration

        Returns estimated speed.
        """
        speeds = self.get_swing_velocity_curve()
        if len(speeds) < 5:
            return None

        avg_speed_px = np.mean(speeds[-10:])  # recent speeds
        if calibration_mm_per_pixel:
            speed_mm_sec = avg_speed_px * fps * calibration_mm_per_pixel
            speed_kmh = speed_mm_sec * 3.6 / 1e6
            return float(speed_kmh)
        return float(avg_speed_px)

    def reset(self):
        self.history.clear()
        self.swing_phases = []
