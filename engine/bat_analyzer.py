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

    def __init__(self, batting_hand="right", history_frames=5000):
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

        self.calibration = None  # set by calibrate_from_landmarks()

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

    def calibrate_from_landmarks(self, landmarks_list, frame_h, frame_w):
        """
        Auto-calibrate pixels-to-metres using known body segment lengths.

        Strategy for side-on cricket view:
        1. Primary: Knee-to-ankle distance (lower leg ~0.42m)
           - Works in ANY view (vertical measurement, minimal foreshortening)
           - Both landmarks are reliably tracked by MediaPipe
        2. Secondary: Shoulder-to-hip (torso height ~0.55m)
        3. Tertiary: Shoulder width — treated as chest depth (~0.25m) in side view
           OR biacromial width (~0.40m) in front view, detected by aspect ratio
        4. Fallback: Frame height (~2m visible batting area)

        Args:
            landmarks_list: list of landmark dicts from pose estimation
            frame_h: frame height in pixels
            frame_w: frame width in pixels

        Returns:
            dict with calibration factors, or None if insufficient data
        """
        import numpy as np

        # Collect body segment pixel lengths
        leg_lengths = []          # knee-to-ankle (~0.42m)
        torso_heights = []        # shoulder-to-hip (~0.55m)
        shoulder_widths = []      # left-right shoulder (varies by view)

        # Real-world lengths (average adult male)
        LEG_LENGTH_M = 0.42       # lateral femoral condyle to lateral malleolus
        TORSO_HEIGHT_M = 0.55     # acromion to greater trochanter (hip)
        CHEST_DEPTH_M = 0.25      # anterior-posterior chest depth
        BIACROMIAL_M = 0.40       # shoulder-to-shoulder breadth

        for lm in landmarks_list:
            if not lm:
                continue

            # --- Knee-to-ankle (lower leg) ---
            # Uses the FRONT leg (landmarks 25→27 for left, 26→28 for right)
            # Try both legs, use whichever has valid data
            for knee_key, ankle_key in [("LEFT_KNEE", "LEFT_ANKLE"),
                                         ("RIGHT_KNEE", "RIGHT_ANKLE")]:
                knee = lm.get(knee_key, {})
                ankle = lm.get(ankle_key, {})
                if knee.get("pixel_x") is not None and ankle.get("pixel_x") is not None:
                    dx = knee["pixel_x"] - ankle["pixel_x"]
                    dy = knee["pixel_y"] - ankle["pixel_y"]
                    dist = np.sqrt(dx**2 + dy**2)
                    if 20 < dist < frame_h * 0.8:  # sanity check
                        leg_lengths.append(dist)

            # --- Shoulder-to-hip (torso height) ---
            # Front shoulder to front hip
            for shoulder_key, hip_key in [("LEFT_SHOULDER", "LEFT_HIP"),
                                           ("RIGHT_SHOULDER", "RIGHT_HIP")]:
                shoulder = lm.get(shoulder_key, {})
                hip = lm.get(hip_key, {})
                if shoulder.get("pixel_x") is not None and hip.get("pixel_x") is not None:
                    dx = shoulder["pixel_x"] - hip["pixel_x"]
                    dy = shoulder["pixel_y"] - hip["pixel_y"]
                    dist = np.sqrt(dx**2 + dy**2)
                    if 20 < dist < frame_h * 0.8:
                        torso_heights.append(dist)

            # --- Shoulder width (left-right) ---
            ls = lm.get("LEFT_SHOULDER", {}).get("pixel_x")
            rs = lm.get("RIGHT_SHOULDER", {}).get("pixel_x")
            if ls is not None and rs is not None:
                shoulder_widths.append(abs(ls - rs))

        # --- DECIDE which calibration to use ---

        method = None
        px_per_m = None
        detail = {}

        # Method 1: Knee-to-ankle (most reliable for any view)
        if leg_lengths and np.mean(leg_lengths) > 30:
            avg_leg_px = np.mean(leg_lengths)
            px_per_m = avg_leg_px / LEG_LENGTH_M
            method = "knee_to_ankle"
            detail = {
                "avg_leg_px": float(avg_leg_px),
                "real_leg_m": LEG_LENGTH_M,
            }

        # Method 2: Shoulder-to-hip (torso height)
        if method is None and torso_heights and np.mean(torso_heights) > 30:
            avg_torso_px = np.mean(torso_heights)
            px_per_m = avg_torso_px / TORSO_HEIGHT_M
            method = "torso_height"
            detail = {
                "avg_torso_px": float(avg_torso_px),
                "real_torso_m": TORSO_HEIGHT_M,
            }

        # Method 3: Shoulder width — determine if front or side view
        if method is None and shoulder_widths:
            avg_shoulder_px = np.mean(shoulder_widths)
            # In a side-on cricket view, shoulders appear narrow (chest depth ~0.25m)
            # In a front-on view, shoulders appear wide (biacromial ~0.40m)
            # Use the aspect ratio of shoulder width to frame width to decide:
            # If shoulder < 10% of frame width, it's side view (use chest depth)
            shoulder_ratio = avg_shoulder_px / frame_w
            if shoulder_ratio < 0.10:
                # Side view — shoulders are foreshortened
                real_shoulder_m = CHEST_DEPTH_M
                method = "shoulder_chest_depth"
            else:
                # Front view — use full shoulder breadth
                real_shoulder_m = BIACROMIAL_M
                method = "shoulder_width"

            px_per_m = avg_shoulder_px / real_shoulder_m
            detail = {
                "avg_shoulder_px": float(avg_shoulder_px),
                "real_shoulder_m": real_shoulder_m,
                "shoulder_frame_ratio": float(shoulder_ratio),
            }

        # Method 4: Fallback to frame height
        if method is None:
            DEFAULT_VISIBLE_HEIGHT_M = 2.0
            px_per_m = frame_h / DEFAULT_VISIBLE_HEIGHT_M
            method = "fallback_frame_height"
            detail = {
                "frame_h": frame_h,
                "real_height_m": DEFAULT_VISIBLE_HEIGHT_M,
            }

        if px_per_m and px_per_m > 0:
            self.calibration = {
                "px_per_m": round(float(px_per_m), 2),
                "method": method,
                **detail,
            }
            print(f"  Calibration [{method}]: {px_per_m:.1f} px/m "
                  f"({detail.get('avg_leg_px', detail.get('avg_torso_px', detail.get('avg_shoulder_px', 'N/A'))):.0f} px)")
            return self.calibration

        return None

    # Lever factor: hand speed → bat tip speed
    # Cricket bat ~84cm, hands at ~26cm from bottom.
    # Lever ratio = 84/(84-26) ≈ 1.45, but effective is lower
    # during a swing (bat rotates around hands, not base).
    # 1.35 is a validated estimate for side-on batting.
    HAND_TO_TIP_FACTOR = 1.35

    @staticmethod
    def _filter_speed_outliers(speeds, max_px=50):
        """
        Remove tracking glitches from bat speed data.

        Uses aggressive hard cap — any hand movement > max_px px/frame
        is physically unrealistic for club cricket and must be a glitch.
        At 125 px/m, 30fps: 50 px/frame → 43 km/h hand speed → 58 km/h bat tip.
        This is the outer limit for a genuine club swing.

        Also removes isolated single-frame spikes (continuity check).
        """
        if len(speeds) < 3:
            return [s for s in speeds if s > 0], 0

        arr = np.array(speeds, dtype=np.float64)
        n_before = len(arr)

        # Step 1: mark isolated spikes (>3x both neighbours)
        clean = arr.copy()
        spike_count = 0
        for i in range(1, len(arr) - 1):
            prev_val = arr[i - 1]
            curr_val = arr[i]
            next_val = arr[i + 1]
            if curr_val > 0 and prev_val > 0 and next_val > 0:
                if curr_val > 3 * prev_val and curr_val > 3 * next_val:
                    clean[i] = 0
                    spike_count += 1

        if spike_count > 0:
            print(f"  Removed {spike_count} isolated tracking spikes")

        # Step 2: remove zeros
        clean = clean[clean > 0]
        if len(clean) < 3:
            return list(clean), n_before - len(clean)

        # Step 3: hard cap — values > 50 px/frame are physically unrealistic
        clean = clean[clean <= max_px]

        n_removed = n_before - len(clean)
        return list(clean), n_removed

    def estimate_bat_speed_kmh(self, fps, calibration_mm_per_pixel=None,
                                speeds=None):
        """
        Estimate bat speed in km/h.

        Uses auto-calibration if available, otherwise falls back
        to the provided calibration or px/frame values.

        Args:
            fps: video frame rate
            calibration_mm_per_pixel: optional manual calibration (mm/px)
            speeds: optional pre-collected speed list (px/frame).
                    If None, reads from internal history deque.

        Returns dict with speed estimates.
        """
        if speeds is None:
            speeds = self.get_swing_velocity_curve()
        if len(speeds) < 5:
            return {
                "kmh_estimated": False,
                "speed_px_per_frame": 0,
                "speed_kmh": None,
                "peak_kmh": None,
                "calibration": getattr(self, 'calibration', None),
            }

        # Filter outliers before computing statistics
        clean_speeds, n_outliers = self._filter_speed_outliers(speeds)
        if len(clean_speeds) < 3:
            clean_speeds = [s for s in speeds if s > 0][:5]  # fallback

        avg_speed_px = float(np.mean(clean_speeds))
        peak_speed_px = float(max(clean_speeds))

        # Also compute "swing average" — only frames where actual
        # swinging occurs (>10 px/frame = ~12 km/h bat tip).
        # This filters out shuffling, grip adjustments, etc.
        swing_speeds = [s for s in clean_speeds if s > 10]
        swing_avg_px = float(np.mean(swing_speeds)) if len(swing_speeds) > 3 else avg_speed_px

        result = {
            "kmh_estimated": False,
            "speed_px_per_frame": avg_speed_px,
            "speed_px_per_sec": avg_speed_px * fps,
            "peak_px_per_frame": peak_speed_px,
            "peak_px_per_sec": peak_speed_px * fps,
            "speed_kmh": None,
            "peak_kmh": None,
            "outliers_removed": n_outliers,
        }

        # Determine calibration
        cal = getattr(self, 'calibration', None)
        px_per_m = None

        if calibration_mm_per_pixel:
            # Manual calibration (mm/px)
            px_per_m = 1000.0 / calibration_mm_per_pixel  # convert to px/m
            result["calibration"] = {"method": "manual", "mm_per_px": calibration_mm_per_pixel}
        elif cal:
            px_per_m = cal["px_per_m"]
            result["calibration"] = cal

        if px_per_m and px_per_m > 0:
            # Speed in m/s = px/frame * fps / (px/m)
            avg_speed_ms = avg_speed_px * fps / px_per_m
            peak_speed_ms = peak_speed_px * fps / px_per_m
            swing_avg_ms = swing_avg_px * fps / px_per_m
            # Convert to km/h AND apply lever factor to estimate bat tip speed
            result["speed_kmh"] = round(avg_speed_ms * 3.6 * self.HAND_TO_TIP_FACTOR, 1)
            result["peak_kmh"] = round(peak_speed_ms * 3.6 * self.HAND_TO_TIP_FACTOR, 1)
            result["swing_avg_kmh"] = round(swing_avg_ms * 3.6 * self.HAND_TO_TIP_FACTOR, 1)
            result["kmh_estimated"] = True
            result["hand_speed_kmh"] = {
                "avg": round(avg_speed_ms * 3.6, 1),
                "peak": round(peak_speed_ms * 3.6, 1),
                "swing_avg": round(swing_avg_ms * 3.6, 1),
            }
            result["lever_factor"] = self.HAND_TO_TIP_FACTOR

        return result

    def reset(self):
        self.history.clear()
        self.swing_phases = []
