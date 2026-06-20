"""
Phase Detector — Identifies key phases of a batting shot.

Phases detected:
1. STANCE    — Initial position before ball is bowled
2. BACKLIFT  — Bat is lifted up and back
3. STRIDE    — Front foot moves forward
4. DOWNSWING — Bat comes down towards the ball
5. IMPACT    — Bat contacts the ball
6. FOLLOW_THROUGH — Bat completes its arc
7. RECOVERY  — Batter returns to balanced position

Uses velocity profiles, joint angle changes, and hand/foot positions.
"""

import numpy as np
from enum import Enum


class BattingPhase(Enum):
    UNKNOWN = "unknown"
    STANCE = "stance"
    BACKLIFT = "backlift"
    STRIDE = "stride"
    DOWNSWING = "downswing"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"
    RECOVERY = "recovery"


class PhaseDetector:
    """
    Detects batting shot phases from temporal pose data.

    Works by analyzing:
    - Vertical velocity of hands (bat lift / downswing)
    - Horizontal velocity of front foot (stride detection)
    - Distance between hands and body (impact proximity)
    - Vertical velocity of bat tip
    """

    def __init__(self, batting_hand="right", fps=30):
        self.batting_hand = batting_hand
        self.fps = fps

        if batting_hand == "right":
            self.front_foot = "LEFT_HEEL"
            self.back_foot = "RIGHT_HEEL"
            self.front_knee = "LEFT_KNEE"
            self.back_knee = "RIGHT_KNEE"
        else:
            self.front_foot = "RIGHT_HEEL"
            self.back_foot = "LEFT_HEEL"
            self.front_knee = "RIGHT_KNEE"
            self.back_knee = "LEFT_KNEE"

        self.frame_data = []  # stores landmark data per frame
        self.phase_labels = []  # (frame_idx, phase) tuples
        self.shot_events = []  # detected individual shots

    def add_frame(self, landmarks, frame_idx):
        """Add a frame's landmarks for analysis."""
        self.frame_data.append({
            "frame": frame_idx,
            "landmarks": landmarks,
        })

    def analyze_phases(self, bat_analyzer=None):
        """
        Run phase detection on all accumulated frames.

        Args:
            bat_analyzer: optional BatAnalyzer instance with swing data

        Returns list of (frame_idx, phase) assignments.
        """
        if len(self.frame_data) < 5:
            return []

        n_frames = len(self.frame_data)

        # Extract signals for analysis
        hand_y_velocity = self._compute_hand_velocity()
        foot_x_velocity = self._compute_foot_velocity()
        bat_tip_velocity = self._compute_bat_tip_velocity(bat_analyzer)
        hip_height = self._compute_hip_height()

        phases = [BattingPhase.UNKNOWN] * n_frames

        # Phase 1: STANCE — first few frames, minimal movement
        phases = self._detect_stance(phases, hand_y_velocity, foot_x_velocity)

        # Phase 2: BACKLIFT — hands move upward (negative y velocity in image coords)
        phases = self._detect_backlift(phases, hand_y_velocity, bat_tip_velocity)

        # Phase 3: STRIDE — front foot moves forward
        phases = self._detect_stride(phases, foot_x_velocity)

        # Phase 4: DOWNSWING — hands move downward rapidly
        phases = self._detect_downswing(phases, hand_y_velocity, bat_tip_velocity)

        # Phase 5: IMPACT — hands at lowest point after downswing
        phases = self._detect_impact(phases, hand_y_velocity)

        # Phase 6: FOLLOW_THROUGH — hands continue moving after impact
        phases = self._detect_follow_through(phases, hand_y_velocity)

        # Phase 7: RECOVERY — everything settles
        phases = self._detect_recovery(phases, hand_y_velocity, foot_x_velocity)

        # Assign labels
        self.phase_labels = [(i, p.value) for i, p in enumerate(phases)
                             if p != BattingPhase.UNKNOWN]

        # Detect individual shot boundaries
        self._detect_shot_boundaries(phases)

        return self.phase_labels

    def _compute_hand_velocity(self):
        """Compute vertical velocity of hands (y-axis) across frames."""
        velocities = []
        for i in range(len(self.frame_data)):
            lm = self.frame_data[i]["landmarks"]
            if not lm:
                velocities.append(0)
                continue

            # Average of both wrists
            left_wrist = lm.get("LEFT_WRIST", {}).get("pixel_y",
                        lm.get("RIGHT_WRIST", {}).get("pixel_y", 0))
            right_wrist = lm.get("RIGHT_WRIST", {}).get("pixel_y", 0)
            avg_y = (left_wrist + right_wrist) / 2 if left_wrist and right_wrist else (left_wrist or right_wrist)

            if i == 0:
                velocities.append(0)
            else:
                prev_lm = self.frame_data[i - 1]["landmarks"]
                if prev_lm:
                    prev_left = prev_lm.get("LEFT_WRIST", {}).get("pixel_y", 0)
                    prev_right = prev_lm.get("RIGHT_WRIST", {}).get("pixel_y", 0)
                    prev_avg = (prev_left + prev_right) / 2 if prev_left and prev_right else (prev_left or prev_right)
                    # Positive = moving down (in image coords)
                    velocities.append(avg_y - prev_avg)
                else:
                    velocities.append(0)
        return velocities

    def _compute_foot_velocity(self):
        """Compute forward velocity of front foot."""
        velocities = []
        for i in range(len(self.frame_data)):
            lm = self.frame_data[i]["landmarks"]
            if not lm:
                velocities.append(0)
                continue

            front_foot = lm.get(self.front_foot, {})
            fx = front_foot.get("pixel_x", 0)

            if i == 0 or not self.frame_data[i - 1]["landmarks"]:
                velocities.append(0)
            else:
                prev_lm = self.frame_data[i - 1]["landmarks"]
                prev_foot = prev_lm.get(self.front_foot, {})
                prev_fx = prev_foot.get("pixel_x", 0)
                velocities.append(fx - prev_fx)

        return velocities

    def _compute_bat_tip_velocity(self, bat_analyzer):
        """Compute bat tip velocity if bat analyzer available."""
        velocities = []
        if not bat_analyzer:
            return [0] * len(self.frame_data)

        for i in range(len(self.frame_data)):
            frame_data_list = list(bat_analyzer.history)
            matching = [h for h in frame_data_list if h.get("frame") == i]
            if matching:
                tip = matching[0].get("bat_tip")
                if i > 0:
                    prev_matching = [h for h in frame_data_list if h.get("frame") == i - 1]
                    if prev_matching and tip and prev_matching[0].get("bat_tip"):
                        prev_tip = prev_matching[0]["bat_tip"]
                        vel = np.sqrt((tip[0] - prev_tip[0])**2 + (tip[1] - prev_tip[1])**2)
                        velocities.append(vel)
                    else:
                        velocities.append(0)
                else:
                    velocities.append(0)
            else:
                velocities.append(0)
        return velocities

    def _compute_hip_height(self):
        """Compute hip height (proxy for knee bend / stance depth)."""
        heights = []
        for fd in self.frame_data:
            lm = fd["landmarks"]
            if not lm:
                heights.append(0)
                continue
            left_hip = lm.get("LEFT_HIP", {}).get("pixel_y", 0)
            right_hip = lm.get("RIGHT_HIP", {}).get("pixel_y", 0)
            avg = (left_hip + right_hip) / 2 if left_hip and right_hip else (left_hip or right_hip)
            heights.append(avg)
        return heights

    def _detect_stance(self, phases, hand_vel, foot_vel, threshold=3):
        """Detect stance phase — minimal movement in hands and feet."""
        for i in range(min(20, len(phases))):
            if abs(hand_vel[i]) < threshold and abs(foot_vel[i]) < threshold:
                phases[i] = BattingPhase.STANCE
        return phases

    def _detect_backlift(self, phases, hand_vel, bat_tip_vel, threshold=-3):
        """
        Detect backlift — hands moving upward.
        In image coords, upward = negative y velocity.
        """
        in_backlift = False
        for i in range(len(phases)):
            if phases[i] == BattingPhase.STANCE:
                in_backlift = False
                continue

            if hand_vel[i] < threshold:  # hands moving up
                phases[i] = BattingPhase.BACKLIFT
                in_backlift = True
            elif in_backlift and abs(hand_vel[i]) < abs(threshold) * 0.5:
                # Still in backlift during pause at top
                phases[i] = BattingPhase.BACKLIFT
            else:
                in_backlift = False
        return phases

    def _detect_stride(self, phases, foot_vel, threshold=5):
        """Detect stride — front foot moving forward (positive x velocity)."""
        in_stride = False
        for i in range(len(phases)):
            if foot_vel[i] > threshold and phases[i] in (BattingPhase.UNKNOWN, BattingPhase.BACKLIFT):
                phases[i] = BattingPhase.STRIDE
                in_stride = True
            elif in_stride and abs(foot_vel[i]) > 1:
                phases[i] = BattingPhase.STRIDE
            else:
                in_stride = False
        return phases

    def _detect_downswing(self, phases, hand_vel, bat_tip_vel, threshold=5):
        """Detect downswing — hands moving downward rapidly (positive y velocity)."""
        in_downswing = False
        for i in range(len(phases)):
            if phases[i] in (BattingPhase.BACKLIFT, BattingPhase.STRIDE):
                if hand_vel[i] > threshold:
                    phases[i] = BattingPhase.DOWNSWING
                    in_downswing = True
            elif in_downswing and hand_vel[i] > 1:
                phases[i] = BattingPhase.DOWNSWING
            else:
                in_downswing = False
        return phases

    def _detect_impact(self, phases, hand_vel, vel_threshold=3):
        """
        Detect impact — transition from downswing (hands moving down)
        to minimal hand velocity (contact point).
        """
        for i in range(2, len(phases) - 1):
            if phases[i - 1] == BattingPhase.DOWNSWING:
                # Impact: velocity crosses zero (from positive to near-zero/negative)
                if abs(hand_vel[i]) < vel_threshold or (hand_vel[i] < 0 and hand_vel[i - 1] > 0):
                    phases[i] = BattingPhase.IMPACT
                    # Also mark next frame if still near impact
                    if i + 1 < len(phases) and abs(hand_vel[i + 1]) < vel_threshold * 2:
                        phases[i + 1] = BattingPhase.IMPACT
        return phases

    def _detect_follow_through(self, phases, hand_vel):
        """Detect follow-through — hands continue after impact."""
        after_impact = False
        for i in range(len(phases)):
            if phases[i] == BattingPhase.IMPACT:
                after_impact = True
            elif after_impact and phases[i] == BattingPhase.UNKNOWN:
                phases[i] = BattingPhase.FOLLOW_THROUGH
                if abs(hand_vel[i]) < 2:
                    after_impact = False  # movement settled
        return phases

    def _detect_recovery(self, phases, hand_vel, foot_vel, threshold=2):
        """Detect recovery — all movement settles."""
        settled_frames = 0
        for i in range(len(phases) - 1, -1, -1):
            if phases[i] in (BattingPhase.UNKNOWN, BattingPhase.FOLLOW_THROUGH):
                if abs(hand_vel[i]) < threshold and abs(foot_vel[i]) < threshold:
                    settled_frames += 1
                    if settled_frames >= 5:
                        phases[i] = BattingPhase.RECOVERY
                else:
                    settled_frames = 0
        return phases

    def _detect_shot_boundaries(self, phases):
        """Group contiguous phase regions into individual shots."""
        if not phases:
            return

        current_shot = []
        in_shot = False

        shot_phases = {BattingPhase.BACKLIFT, BattingPhase.STRIDE,
                       BattingPhase.DOWNSWING, BattingPhase.IMPACT,
                       BattingPhase.FOLLOW_THROUGH}

        for i, p in enumerate(phases):
            if p in shot_phases:
                if not in_shot:
                    if current_shot:
                        self.shot_events.append(current_shot)
                    current_shot = []
                    in_shot = True
                current_shot.append((i, p.value))
            else:
                if in_shot and p == BattingPhase.RECOVERY:
                    current_shot.append((i, p.value))
                    self.shot_events.append(current_shot)
                    current_shot = []
                    in_shot = False
                elif in_shot:
                    in_shot = False
                    if len(current_shot) > 3:
                        self.shot_events.append(current_shot)
                    current_shot = []

        if current_shot:
            self.shot_events.append(current_shot)

    def get_shot_count(self):
        """Get number of detected shots."""
        return len(self.shot_events)

    def get_shot_summary(self):
        """Get summary of each detected shot."""
        summaries = []
        for i, shot in enumerate(self.shot_events):
            phases_in_shot = set(p for _, p in shot)
            start_frame = shot[0][0]
            end_frame = shot[-1][0]
            duration_frames = end_frame - start_frame + 1
            duration_sec = duration_frames / self.fps if self.fps else 0

            summaries.append({
                "shot_number": i + 1,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": duration_frames,
                "duration_sec": round(duration_sec, 2),
                "phases": list(phases_in_shot),
                "has_impact": BattingPhase.IMPACT.value in phases_in_shot,
            })
        return summaries

    def get_phase_at_frame(self, frame_idx):
        """Get the phase label for a given frame index."""
        for f, p in self.phase_labels:
            if f == frame_idx:
                return p
        return BattingPhase.UNKNOWN.value

    def reset(self):
        self.frame_data = []
        self.phase_labels = []
        self.shot_events = []
