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
The phase detectors are intentionally liberal (1-frame threshold crossing).
Shot boundary detection is conservative (minimum duration + gap filtering).
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

    Phase detection is liberal (any threshold crossing is labeled).
    Shot boundary detection is conservative (min duration + gap filtering).

    Takes frame_step into account: thresholds and minimum shot/gap
    durations are scaled so behaviour is consistent regardless of how
    many frames are skipped during analysis.
    """

    # Minimum shot duration in ORIGINAL frames (~0.5s at 30fps)
    MIN_SHOT_FRAMES = 15
    # Minimum gap between separate shots (original frames)
    MIN_GAP_FRAMES = 15

    def __init__(self, batting_hand="right", fps=30, frame_step=1):
        self.batting_hand = batting_hand
        self.fps = fps

        # Scale minimums by frame_step so behaviour is consistent
        # whether every frame or every Nth frame is analysed.
        self.eff_min_shot_frames = max(5, self.MIN_SHOT_FRAMES // frame_step)
        self.eff_min_gap_frames = max(3, self.MIN_GAP_FRAMES // frame_step)

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

    def add_frame(self, landmarks, frame_idx, is_batter=True):
        """Add a frame's landmarks for analysis.

        Args:
            landmarks: MediaPipe pose landmarks for the detected person
            frame_idx: Frame number in the original video
            is_batter: Whether the detected person is believed to be the batter
                        (True = batter, False = wicketkeeper / other person)
        """
        self.frame_data.append({
            "frame": frame_idx,
            "landmarks": landmarks,
            "is_batter": is_batter,
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

    def _frame_gap(self, i):
        """Return the actual frame gap between frame_data[i] and frame_data[i-1]."""
        if i <= 0 or i >= len(self.frame_data):
            return 1
        return max(1, self.frame_data[i]["frame"] - self.frame_data[i - 1]["frame"])

    def _compute_hand_velocity(self):
        """Compute vertical velocity of hands (y-axis) in pixels per frame.

        Velocity is normalised by the actual frame gap so that thresholds
        remain consistent regardless of frame_step size.
        """
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
                    gap = self._frame_gap(i)
                    velocities.append((avg_y - prev_avg) / gap)
                else:
                    velocities.append(0)
        return velocities

    def _compute_foot_velocity(self):
        """Compute forward velocity of front foot in pixels per frame.

        Normalised by actual frame gap for frame_step independence.
        """
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
                gap = self._frame_gap(i)
                velocities.append((fx - prev_fx) / gap)

        return velocities

    def _compute_bat_tip_velocity(self, bat_analyzer):
        """Compute bat tip velocity if bat analyzer available.

        Velocity is normalised by the actual frame gap for frame_step
        independence.  Frame matching uses the real frame index stored
        in frame_data, not the local loop counter.
        """
        velocities = []
        if not bat_analyzer:
            return [0] * len(self.frame_data)

        frame_data_list = list(bat_analyzer.history)
        # Build a fast lookup: frame -> entry
        frame_map = {h.get("frame"): h for h in frame_data_list if h.get("frame") is not None}

        for i in range(len(self.frame_data)):
            current_frame = self.frame_data[i]["frame"]
            cur = frame_map.get(current_frame)
            if cur:
                tip = cur.get("bat_tip")
                if i > 0 and tip:
                    prev_frame = self.frame_data[i - 1]["frame"]
                    prev = frame_map.get(prev_frame)
                    if prev and prev.get("bat_tip"):
                        prev_tip = prev["bat_tip"]
                        gap = self._frame_gap(i)
                        vel = np.sqrt((tip[0] - prev_tip[0])**2 + (tip[1] - prev_tip[1])**2) / gap
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

        Liberal: any frame with hand_vel < threshold is labeled BACKLIFT.
        Stays in backlift during pause at top (|vel| < |threshold| * 0.5).
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
                # Still in backlift during pause at top (|vel| < 1.5)
                phases[i] = BattingPhase.BACKLIFT
            else:
                in_backlift = False
        return phases

    def _detect_stride(self, phases, foot_vel, threshold=5):
        """
        Detect stride — front foot moving forward (positive x velocity).
        Liberal: any frame with foot_vel > threshold is labeled STRIDE.
        """
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
        """
        Detect downswing — hands moving downward rapidly (positive y velocity).
        Liberal: any frame with hand_vel > threshold (after backlift/stride) is labeled DOWNSWING.
        """
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

    def _detect_follow_through(self, phases, hand_vel, max_follow_frames=30):
        """
        Detect follow-through — hands continue after impact.
        Limited to max_follow_frames after impact to prevent runaway labeling.
        """
        after_impact = False
        follow_count = 0
        for i in range(len(phases)):
            if phases[i] == BattingPhase.IMPACT:
                after_impact = True
                follow_count = 0
            elif after_impact and phases[i] == BattingPhase.UNKNOWN:
                follow_count += 1
                if follow_count > max_follow_frames:
                    after_impact = False
                    follow_count = 0
                else:
                    phases[i] = BattingPhase.FOLLOW_THROUGH
                    if abs(hand_vel[i]) < 2:
                        after_impact = False  # movement settled
        return phases

    def _detect_recovery(self, phases, hand_vel, foot_vel, threshold=2):
        """Detect recovery — all movement settles (scans backward from end)."""
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
        """
        Group contiguous phase regions into individual shots.

        Strategy:
        - Find all contiguous regions of shot-related phases
        - Merge regions whose END FRAME NUMBER gap is smaller than
          MIN_GAP_FRAMES (uses actual frame numbers, not frame_data indices)
        - Discard regions shorter than MIN_SHOT_FRAMES in terms of
          actual frame-count duration

        This conservative approach filters out noise (brief threshold crossings)
        while preserving real shots (sustained phase regions).
        """
        if not phases:
            return

        # Step 1: Identify all contiguous shot-phase regions
        raw_shots = []
        current_shot = []
        in_shot = False

        shot_phases = {BattingPhase.BACKLIFT, BattingPhase.STRIDE,
                       BattingPhase.DOWNSWING, BattingPhase.IMPACT,
                       BattingPhase.FOLLOW_THROUGH}

        for i, p in enumerate(phases):
            if p in shot_phases:
                if not in_shot:
                    # Save previous shot region when starting new one
                    if current_shot:
                        raw_shots.append(current_shot)
                    current_shot = []
                    in_shot = True
                # Store the ACTUAL frame number, not frame_data index
                actual_frame = self.frame_data[i]["frame"]
                current_shot.append((actual_frame, p.value))
            else:
                if in_shot:
                    in_shot = False
                    if current_shot:
                        raw_shots.append(current_shot)
                    current_shot = []

        # Save last region if video ends mid-shot
        if current_shot:
            raw_shots.append(current_shot)

        # Step 2: Merge close regions and filter short ones
        if not raw_shots:
            return

        filtered = [raw_shots[0]]

        for shot in raw_shots[1:]:
            # Gap in ORIGINAL frame numbers
            gap = shot[0][0] - filtered[-1][-1][0]
            if gap < self.eff_min_gap_frames:
                # Merge: extend the previous shot with gap frames
                merged = filtered[-1]
                for g in range(filtered[-1][-1][0] + 1, shot[0][0]):
                    merged.append((g, "transition"))
                merged.extend(shot)
            else:
                # Gap sufficient — keep separate
                filtered.append(shot)

        # Step 3: Apply minimum duration filter (in original frames)
        self.shot_events = []
        for s in filtered:
            if not s:
                continue
            first_frame = s[0][0]
            last_frame = s[-1][0]
            total_frames = last_frame - first_frame + 1
            if total_frames >= self.eff_min_shot_frames:
                self.shot_events.append(s)

    def get_shot_count(self):
        """Get number of detected shots."""
        return len(self.shot_events)

    @staticmethod
    def _is_valid_shot(phases_in_shot, shot_phases_ordered):
        """Check whether a shot has a physically plausible batting phase sequence.

        A valid batting shot must either:
        1. Contain IMPACT (the ball was struck), or
        2. Contain BACKLIFT AND DOWNSWING with backlift occurring before downswing
           (indicating a genuine swing attempt, even if the ball wasn't hit)

        Single-phase shots (e.g. only "backlift") are rejected as noise,
        as are shots with physically impossible phase ordering
        (e.g. downswing before backlift).
        """
        if BattingPhase.IMPACT.value in phases_in_shot:
            return True

        has_backlift = BattingPhase.BACKLIFT.value in phases_in_shot
        has_downswing = BattingPhase.DOWNSWING.value in phases_in_shot

        if not (has_backlift and has_downswing):
            return False

        # Check ordering: backlift must appear before downswing
        backlift_order = None
        downswing_order = None
        for idx, (_, p) in enumerate(shot_phases_ordered):
            if p == BattingPhase.BACKLIFT.value and backlift_order is None:
                backlift_order = idx
            if p == BattingPhase.DOWNSWING.value and downswing_order is None:
                downswing_order = idx

        return backlift_order is not None and downswing_order is not None and backlift_order < downswing_order

    def get_shot_summary(self):
        """Get summary of each detected shot.

        Filters out invalid shots (wicketkeeper noise, partial detections)
        using _is_valid_shot() which checks for plausible batting phase sequences.
        """
        summaries = []
        valid_count = 0
        for i, shot in enumerate(self.shot_events):
            phases_in_shot = set(p for _, p in shot if p != "transition")
            start_frame = shot[0][0]
            end_frame = shot[-1][0]
            duration_frames = end_frame - start_frame + 1
            duration_sec = duration_frames / self.fps if self.fps else 0

            # Determine batter frame ratio
            batter_frames = sum(
                1 for fd in self.frame_data
                if start_frame <= fd["frame"] <= end_frame and fd.get("is_batter", True)
            )
            total_shot_frames = max(1, duration_frames)
            batter_ratio = batter_frames / total_shot_frames

            # Skip if this isn't a valid batting shot
            if not self._is_valid_shot(phases_in_shot, shot):
                continue

            valid_count += 1
            summaries.append({
                "shot_number": valid_count,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": duration_frames,
                "duration_sec": round(duration_sec, 2),
                "phases": list(phases_in_shot),
                "has_impact": BattingPhase.IMPACT.value in phases_in_shot,
                "batter_frame_ratio": round(batter_ratio, 2),
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
