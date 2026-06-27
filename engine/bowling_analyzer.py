"""
Bowling Analyzer — Opportunistic bowling analysis from batter-analysis video.

Camera position:   Non-striker end, behind bowler, looking toward batter.
Source data:       Bowler frames already labelled by PersonTracker as
                   ``bowler_approach``, ``bowler_delivery``, or
                   ``bowler_follow_through``.

Key metrics (all opportunistic — no separate upload or setup):
  1. RUN-UP SPEED         — How fast the bowler approaches the crease.
  2. ARM SPEED             — Angular velocity of the bowling arm at delivery.
  3. RELEASE HEIGHT        — Wrist y-position (normalised) at delivery stride.
  4. BALL SPEED            — Already captured by ball tracker; attributed here.
  5. BOWL TYPE             — Fast / Medium / Spin from combined signals.

Front-on view constraints:
  - Bowler runs TOWARD the camera (opposite of batter).
  - Ball release happens when bowler is closest to camera → largest in frame.
  - Arm action is a circular motion visible from front-on as vertical
    wrist movement.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────

# Arm speed thresholds (rad/s)
ARM_SPEED_FAST_MIN = 12.0        # ≥12 rad/s = fast bowling arm action
ARM_SPEED_MEDIUM_MIN = 8.0       # ≥8 rad/s = medium pace
ARM_SPEED_SPIN_MAX = 8.0         # <8 rad/s = spin

# Release height thresholds (normalised frame height, 0=top, 1=bottom)
RELEASE_HEIGHT_FAST_MAX = 0.45   # Fast bowlers release higher (≤0.45)
RELEASE_HEIGHT_SPIN_MIN = 0.50   # Spinners release lower (≥0.50)

# Ball speed thresholds (km/h) — for bowl type refinement
BALL_SPEED_FAST_MIN = 120.0      # ≥120 km/h → fast
BALL_SPEED_MEDIUM_MIN = 90.0     # ≥90 km/h → medium
BALL_SPEED_SPIN_MAX = 100.0      # ≤100 km/h → spin

# Run-up detection
RUNUP_MIN_FRAMES = 5             # Minimum approach frames for run-up estimate
ARM_SPEED_MIN_FRAMES = 3         # Minimum delivery frames for arm speed calc

# Delivery event clustering
DELIVERY_MIN_GAP_FRAMES = 30     # Minimum frames between separate deliveries


class BowlType:
    """Bowling type classification labels."""
    FAST = "fast"
    FAST_MEDIUM = "fast_medium"
    MEDIUM = "medium"
    MEDIUM_FAST = "medium_fast"   # for UI display as "Medium-Fast"
    OFF_SPIN = "off_spin"
    LEG_SPIN = "leg_spin"
    SPIN = "spin"
    UNKNOWN = "unknown"

    @classmethod
    def all_types(cls) -> List[str]:
        return [cls.FAST, cls.FAST_MEDIUM, cls.MEDIUM, cls.MEDIUM_FAST,
                cls.OFF_SPIN, cls.LEG_SPIN, cls.SPIN, cls.UNKNOWN]

    @classmethod
    def label(cls, bowl_type: str) -> str:
        """Human-readable label for display."""
        labels = {
            cls.FAST: "Fast",
            cls.FAST_MEDIUM: "Fast-Medium",
            cls.MEDIUM: "Medium",
            cls.MEDIUM_FAST: "Medium-Fast",
            cls.OFF_SPIN: "Off Spin",
            cls.LEG_SPIN: "Leg Spin",
            cls.SPIN: "Spin",
            cls.UNKNOWN: "Unknown",
        }
        return labels.get(bowl_type, bowl_type)

    @classmethod
    def icon(cls, bowl_type: str) -> str:
        """Return an icon identifier for the bowl type."""
        icons = {
            cls.FAST: "⚡",
            cls.FAST_MEDIUM: "⚡",
            cls.MEDIUM: "➡️",
            cls.MEDIUM_FAST: "➡️",
            cls.OFF_SPIN: "🔄",
            cls.LEG_SPIN: "🔄",
            cls.SPIN: "🔄",
            cls.UNKNOWN: "❓",
        }
        return icons.get(bowl_type, "❓")


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    elif score >= 0.5:
        return "medium"
    elif score >= 0.25:
        return "low"
    return "estimate"


# ────────────────────────────────────────────────────────────────────────
# Core Analyzer
# ────────────────────────────────────────────────────────────────────────

class BowlingAnalyzer:
    """
    Opportunistic bowling analysis from batter-analysis video data.

    Usage (inside analyser.py post-processing):
        bowling = BowlingAnalyzer(fps=video_fps, camera_view=camera_view)
        result = bowling.analyse(
            bowler_frame_data=collected_bowler_data,
            ball_speed_kmh=ball_speed_kmh,
            frame_height=h,
            frame_width=w,
        )
    """

    def __init__(
        self,
        fps: float = 30,
        batting_hand: str = "right",
        camera_view: str = "front_on",
    ):
        self.fps = fps
        self.batting_hand = batting_hand
        self.camera_view = camera_view

        # Front/back assignments
        if batting_hand == "right":
            self.bowling_arm = "RIGHT"    # Right-arm bowler
            self.non_bowling_arm = "LEFT"
        else:
            self.bowling_arm = "LEFT"     # Left-arm bowler
            self.non_bowling_arm = "RIGHT"

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def analyse(
        self,
        bowler_frame_data: List[Dict[str, Any]],
        ball_speed_kmh: Optional[float] = None,
        frame_height: int = 1080,
        frame_width: int = 1920,
    ) -> Dict[str, Any]:
        """Analyse all bowling data collected during the session.

        Args:
            bowler_frame_data: List of per-frame dicts from bowler frames,
                each containing at minimum:
                  - frame: int
                  - person_label: str (bowler_approach/delivery/follow_through)
                  - landmarks: dict of pose landmarks
                  - (optional) ball data
            ball_speed_kmh: Ball speed from ball tracker (km/h), if available.
            frame_height: Frame height in pixels.
            frame_width: Frame width in pixels.

        Returns:
            dict with keys:
              - num_deliveries_detected (int)
              - deliveries (list[dict]): per-delivery breakdown
              - bowl_type (str): most common type across deliveries
              - bowl_type_label (str): human-readable
              - bowl_type_confidence (str)
              - avg_run_up_speed (float): pixels/sec
              - avg_arm_speed (float): rad/s
              - avg_release_height (float): normalised [0..1]
    `         - has_bowling_data (bool): True if any deliveries were analysed
        """
        if not bowler_frame_data:
            return self._empty_result()

        # 1. Detect delivery events from the frame sequence
        deliveries = self._detect_deliveries(bowler_frame_data)

        # 2. Analyse each delivery (pass ball_speed_kmh so bowl type is accurate)
        for delivery in deliveries:
            if ball_speed_kmh is not None:
                delivery["ball_speed_kmh"] = ball_speed_kmh
                delivery["ball_speed_source"] = "tracker"
            self._analyse_delivery(delivery, frame_height, frame_width)

        # 3. Aggregate
        if not deliveries:
            return self._empty_result()

        bowl_types = [d.get("bowl_type", "unknown") for d in deliveries
                      if d.get("bowl_type", "unknown") != "unknown"]
        # Use nan-safe mean: filter out None/0 for metrics that may not be available
        runup_vals = [d.get("run_up_speed_px_per_sec") for d in deliveries
                      if d.get("run_up_speed_px_per_sec") is not None]
        avg_runup = float(np.mean(runup_vals)) if runup_vals else 0
        arm_vals = [d.get("arm_speed_rad_s") for d in deliveries
                    if d.get("arm_speed_rad_s") is not None]
        avg_arm = float(np.mean(arm_vals)) if arm_vals else 0
        release_vals = [d.get("release_height") for d in deliveries
                        if d.get("release_height") is not None]
        avg_release = float(np.mean(release_vals)) if release_vals else 0

        # Most common bowl type
        if bowl_types:
            most_common = Counter(bowl_types).most_common(1)[0]
            primary_type = most_common[0]
            type_confidence = min(0.9, 0.5 + most_common[1] / max(1, len(bowl_types)) * 0.4)
        else:
            # Fall back to ball-speed-only classification
            primary_type = self._classify_by_ball_speed(ball_speed_kmh)
            type_confidence = 0.4

        return {
            "has_bowling_data": True,
            "num_deliveries_detected": len(deliveries),
            "deliveries": deliveries,
            "bowl_type": primary_type,
            "bowl_type_label": BowlType.label(primary_type),
            "bowl_type_icon": BowlType.icon(primary_type),
            "bowl_type_confidence": _confidence_label(type_confidence),
            "avg_run_up_speed_px_per_s": round(avg_runup, 1),
            "avg_arm_speed_rad_s": round(avg_arm, 2),
            "avg_release_height": round(avg_release, 3),
            "avg_release_height_cm": self._release_height_to_cm(avg_release, frame_height),
        }

    # ────────────────────────────────────────────────────────────────
    # Delivery detection
    # ────────────────────────────────────────────────────────────────

    def _detect_deliveries(
        self, bowler_frame_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Cluster bowler frames into individual deliveries.

        A delivery = approach frames → delivery frame(s) → follow-through frames.
        We detect clusters by finding ``bowler_delivery`` or ``bowler_follow_through``
        and then:
          - Moving forward to collect the rest of the delivery + follow-through
          - Moving backward to collect preceding approach frames

        This handles the real-world ordering: approach frames happen BEFORE
        the first delivery frame in the list.
        """
        deliveries: List[Dict[str, Any]] = []

        for i, fd in enumerate(bowler_frame_data):
            label = fd.get("person_label", "")
            landmarks = fd.get("landmarks", {})

            # Trigger: a delivery or follow-through frame (start of a potential delivery)
            if label not in ("bowler_delivery", "bowler_follow_through"):
                continue

            # Check if this frame is already part of an existing delivery
            # (by checking if any existing delivery contains this frame index)
            frame = fd.get("frame", 0)
            is_duplicate = False
            for d in deliveries:
                all_frames = (d.get("approach_frames", [])
                              + d.get("delivery_frames", [])
                              + d.get("follow_through_frames", []))
                if any(af.get("frame") == frame for af in all_frames):
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            # Start a new delivery
            current: Dict[str, Any] = {
                "delivery_frame": frame,
                "approach_frames": [],
                "delivery_frames": [],
                "follow_through_frames": [],
                "landmarks_delivery": landmarks,
                "landmarks_follow_through": [],
            }

            # Collect all approach frames that precede this trigger frame
            # (scan backward from i-1 to find contiguous approach frames)
            for j in range(i - 1, -1, -1):
                prev = bowler_frame_data[j]
                if prev.get("person_label") == "bowler_approach":
                    current["approach_frames"].insert(0, prev)
                else:
                    break

            # Collect the trigger frame
            if label == "bowler_delivery":
                current["delivery_frames"].append(fd)
                if landmarks:
                    current["landmarks_delivery"] = landmarks
            elif label == "bowler_follow_through":
                current["follow_through_frames"].append(fd)
                if landmarks:
                    current["landmarks_follow_through"].append(landmarks)

            # Collect following frames (delivery + follow-through)
            for j in range(i + 1, len(bowler_frame_data)):
                nxt = bowler_frame_data[j]
                nxt_label = nxt.get("person_label", "")
                if nxt_label == "bowler_delivery":
                    current["delivery_frames"].append(nxt)
                    if nxt.get("landmarks"):
                        current["landmarks_delivery"] = nxt["landmarks"]
                elif nxt_label == "bowler_follow_through":
                    current["follow_through_frames"].append(nxt)
                    if nxt.get("landmarks"):
                        current["landmarks_follow_through"].append(nxt["landmarks"])
                elif nxt_label == "bowler_approach":
                    # New approach = new delivery coming, stop here
                    break
                else:
                    # Non-bowler frame = end of this delivery
                    break

            if self._is_complete_delivery(current):
                deliveries.append(current)

        return deliveries

    @staticmethod
    def _is_complete_delivery(delivery: Dict) -> bool:
        """A delivery must have at least some delivery or follow-through frames."""
        return (len(delivery.get("delivery_frames", [])) > 0 or
                len(delivery.get("follow_through_frames", [])) > 0)

    # ────────────────────────────────────────────────────────────────
    # Per-delivery analysis
    # ────────────────────────────────────────────────────────────────

    def _analyse_delivery(
        self,
        delivery: Dict[str, Any],
        frame_height: int,
        frame_width: int,
    ) -> None:
        """Populate delivery dict with bowling metrics."""
        # Run-up speed (from approach frames)
        delivery["run_up_speed_px_per_sec"] = self._estimate_run_up_speed(
            delivery["approach_frames"]
        )

        # Arm speed at delivery
        arm_speed, arm_angle = self._estimate_arm_speed(
            delivery["delivery_frames"],
            delivery.get("landmarks_follow_through", []),
        )
        delivery["arm_speed_rad_s"] = arm_speed
        delivery["arm_angle_deg"] = arm_angle

        # Release height
        delivery["release_height"] = self._estimate_release_height(
            delivery["landmarks_delivery"], frame_height
        )
        delivery["release_height_cm"] = self._release_height_to_cm(
            delivery["release_height"], frame_height
        )

        # Bowl type classification
        delivery["bowl_type"] = self._classify_bowl_type(
            arm_speed=arm_speed,
            release_height=delivery["release_height"],
            ball_speed_kmh=delivery.get("ball_speed_kmh"),
        )
        delivery["bowl_type_label"] = BowlType.label(delivery["bowl_type"])
        delivery["bowl_type_icon"] = BowlType.icon(delivery["bowl_type"])

        # Bowling hand
        delivery["bowling_arm"] = self.bowling_arm

    # ────────────────────────────────────────────────────────────────
    # Run-up speed
    # ────────────────────────────────────────────────────────────────

    def _estimate_run_up_speed(
        self, approach_frames: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Estimate run-up speed from approach frames.

        From front-on, the bowler runs toward the camera.
        The bowler's size in frame increases over approach frames.
        We use the rate of change of the nose y-position as a proxy
        for approach speed — larger positive delta_y per frame = faster.
        """
        if len(approach_frames) < RUNUP_MIN_FRAMES:
            return None

        # Extract nose y positions
        nose_ys = []
        for fd in approach_frames:
            landmarks = fd.get("landmarks", {})
            nose = landmarks.get("NOSE", {})
            if nose.get("y") is not None:
                nose_ys.append(nose["y"])

        if len(nose_ys) < RUNUP_MIN_FRAMES:
            return None

        # Run-up speed: rate of nose_y change
        # Positive = bowler getting larger (moving toward camera) = y increases
        dy = nose_ys[-1] - nose_ys[0]
        dt = max(1, len(nose_ys)) / self.fps if self.fps > 0 else 1.0

        # dy per second (in normalised coordinates)
        speed_normalised = abs(dy) / dt if dt > 0 else 0

        # Convert to a rough px/s using frame height
        # Typical approach: nose moves from ~0.3 to ~0.6 over the run-up
        # Higher value = faster run-up
        return round(speed_normalised * 1000, 1)  # scaled for readability

    # ────────────────────────────────────────────────────────────────
    # Arm speed
    # ────────────────────────────────────────────────────────────────

    def _estimate_arm_speed(
        self,
        delivery_frames: List[Dict[str, Any]],
        follow_through_frames: List[Dict[str, Any]],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Estimate bowling arm angular speed (rad/s).

        From front-on, the bowling arm goes through a vertical arc.
        We track the angle of the shoulder-wrist vector relative to vertical
        across the delivery + early follow-through frames.

        Returns:
            (arm_speed_rad_s, arm_angle_at_release_deg)
            arm_angle_at_release = angle of bowling arm from vertical (0° = straight up)
        """
        # Collect wrist-y and shoulder-y positions across delivery frames
        angles = []
        timestamps = []

        all_frames = delivery_frames + follow_through_frames[:5]  # first 5 FT frames

        for fd in all_frames:
            landmarks = fd.get("landmarks", {})
            if not landmarks:
                continue
            angle = self._compute_arm_angle(landmarks)
            if angle is not None:
                angles.append(angle)
                timestamps.append(fd.get("frame", 0))

        if len(angles) < ARM_SPEED_MIN_FRAMES:
            return None, None

        # Arm speed = angular velocity (change in angle / time)
        angle_deltas = []
        time_deltas = []
        for i in range(1, len(angles)):
            d_angle = abs(angles[i] - angles[i - 1])
            d_time = (timestamps[i] - timestamps[i - 1]) / self.fps if self.fps > 0 else 0.03
            if d_time > 0:
                angle_deltas.append(d_angle)
                time_deltas.append(d_time)

        if not angle_deltas:
            return None, None

        # Mean angular velocity (deg/s → rad/s)
        mean_deg_s = np.mean([d / t for d, t in zip(angle_deltas, time_deltas)])
        arm_speed_rad_s = mean_deg_s * (math.pi / 180.0)

        # Arm angle at release: use the angle from the latest delivery frame
        if delivery_frames:
            last_delivery_landmarks = delivery_frames[-1].get("landmarks", {})
            arm_angle = self._compute_arm_angle(last_delivery_landmarks)
        else:
            arm_angle = angles[-1] if angles else None

        return round(arm_speed_rad_s, 2), round(arm_angle, 1) if arm_angle else None

    def _compute_arm_angle(self, landmarks: Dict[str, Any]) -> Optional[float]:
        """Compute bowling arm angle from vertical.

        Angle = angle between shoulder-wrist vector and vertical (downward).
        0° = arm straight up (fully vertical)
        90° = arm horizontal
        180° = arm straight down

        For front-on view:
        - Bowling arm shoulder key
        - Bowling arm wrist key
        """
        bowling_shoulder = landmarks.get(f"{self.bowling_arm}_SHOULDER")
        bowling_wrist = landmarks.get(f"{self.bowling_arm}_WRIST")

        if not bowling_shoulder or not bowling_wrist:
            return None

        # Vector: shoulder → wrist
        dx = bowling_wrist.get("x", 0) - bowling_shoulder.get("x", 0)
        dy = bowling_wrist.get("y", 0) - bowling_shoulder.get("y", 0)

        # Angle from vertical (pointing down)
        # In normalised coords, y increases downward
        # Vertical-down vector = (0, 1)
        # Angle between (dx, dy) and (0, 1)
        dot = dy  # (0 * dx + 1 * dy)
        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 0.001:
            return None

        cos_angle = dot / mag
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))
        return angle_deg

    # ────────────────────────────────────────────────────────────────
    # Release height
    # ────────────────────────────────────────────────────────────────

    def _estimate_release_height(
        self,
        delivery_landmarks: Dict[str, Any],
        frame_height: int,
    ) -> Optional[float]:
        """Estimate ball release height from delivery stride landmarks.

        Release height = normalised y-position of the bowling wrist
        at the delivery stride. 0.0 = top of frame, 1.0 = bottom.

        Fast bowlers release higher (lower y value, closer to top).
        Spinners release lower (higher y value, closer to waist height).
        """
        if not delivery_landmarks:
            return None

        bowling_wrist = delivery_landmarks.get(f"{self.bowling_arm}_WRIST")
        if not bowling_wrist:
            return None

        # Normalised release height
        release_y = bowling_wrist.get("y", 0.5)
        return round(release_y, 3)

    @staticmethod
    def _release_height_to_cm(
        release_height: Optional[float],
        frame_height: int,
    ) -> Optional[float]:
        """Convert normalised release height to approximate cm from ground.

        Assumes a typical 180 cm bowler and 240 cm visible frame height
        from front-on (rough estimate — depends on zoom).
        """
        if release_height is None or frame_height <= 0:
            return None
        # Rough conversion: normalised 0→1 = ~240 cm visible height
        # Release at y=0.3 means 0.3 * 240 = 72 cm from top of frame
        # If top of frame is ~10 cm above head, head at ~180 cm,
        # then release height from ground ≈ 240 - (release_y * 240) + offset
        visible_height_cm = 240.0  # rough estimate for typical cricket net
        height_from_top = release_height * visible_height_cm
        height_from_ground = visible_height_cm - height_from_top
        return round(height_from_ground, 1)

    # ────────────────────────────────────────────────────────────────
    # Bowl type classification
    # ────────────────────────────────────────────────────────────────

    def _classify_bowl_type(
        self,
        arm_speed: Optional[float] = None,
        release_height: Optional[float] = None,
        ball_speed_kmh: Optional[float] = None,
    ) -> str:
        """Classify bowling type from available signals.

        Decision logic (from front-on):
          1. High arm speed + high release + fast ball → Fast
          2. Low arm speed + low release + slower ball → Spin
          3. Medium signals → Medium
          4. Ball speed only → medium/medium-fast (fallback)

        Each signal votes, majority wins.
        """
        votes: Dict[str, int] = {}

        # Vote from arm speed
        if arm_speed is not None:
            if arm_speed >= ARM_SPEED_FAST_MIN:
                votes["fast"] = votes.get("fast", 0) + 2
            elif arm_speed >= ARM_SPEED_MEDIUM_MIN:
                votes["medium"] = votes.get("medium", 0) + 1
            else:
                votes["spin"] = votes.get("spin", 0) + 2

        # Vote from release height
        if release_height is not None:
            if release_height <= RELEASE_HEIGHT_FAST_MAX:
                votes["fast"] = votes.get("fast", 0) + 1
            elif release_height >= RELEASE_HEIGHT_SPIN_MIN:
                votes["spin"] = votes.get("spin", 0) + 1
            else:
                votes["medium"] = votes.get("medium", 0) + 1

        # Vote from ball speed
        if ball_speed_kmh is not None:
            if ball_speed_kmh >= BALL_SPEED_FAST_MIN:
                votes["fast"] = votes.get("fast", 0) + 2
            elif ball_speed_kmh >= BALL_SPEED_MEDIUM_MIN:
                votes["medium"] = votes.get("medium", 0) + 1
            else:
                votes["spin"] = votes.get("spin", 0) + 1

        if not votes:
            return BowlType.UNKNOWN

        # Determine winner
        winner = max(votes, key=votes.get)

        # Refine: fast vs fast-medium, spin vs off/leg
        if winner == "fast":
            if ball_speed_kmh is not None and ball_speed_kmh < 130:
                return BowlType.FAST_MEDIUM
            return BowlType.FAST
        elif winner == "medium":
            if ball_speed_kmh is not None:
                if ball_speed_kmh >= 115:
                    return BowlType.MEDIUM_FAST
                elif ball_speed_kmh <= 85:
                    return BowlType.SPIN
            return BowlType.MEDIUM
        elif winner == "spin":
            return BowlType.SPIN

        return BowlType.UNKNOWN

    def _classify_by_ball_speed(self, ball_speed_kmh: Optional[float]) -> str:
        """Fallback bowl type when only ball speed is available."""
        if ball_speed_kmh is None:
            return BowlType.UNKNOWN
        if ball_speed_kmh >= BALL_SPEED_FAST_MIN:
            return BowlType.FAST
        elif ball_speed_kmh >= BALL_SPEED_MEDIUM_MIN:
            return BowlType.MEDIUM
        else:
            return BowlType.SPIN

    # ────────────────────────────────────────────────────────────────
    # Empty result
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        return {
            "has_bowling_data": False,
            "num_deliveries_detected": 0,
            "deliveries": [],
            "bowl_type": "unknown",
            "bowl_type_label": "Unknown",
            "bowl_type_icon": "❓",
            "bowl_type_confidence": "low",
            "avg_run_up_speed_px_per_s": 0,
            "avg_arm_speed_rad_s": 0,
            "avg_release_height": 0,
            "avg_release_height_cm": 0,
        }
