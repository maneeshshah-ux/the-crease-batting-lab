"""
Front-on Shot Classifier — Rule-based cricket shot classification,
including modern/innovative shots.

Camera: front-on (behind bowler, looking toward batter).

Classification uses a decision tree fed by:
  - Foot movement (forward / back / no stride)
  - Swing path angle (straight / across / square)
  - Bat face angle at impact (open / closed / straight)
  - Ball line relative to stumps (off / middle / leg / down leg)
  - Ball length (short / good / full / half-volley / yorker)
  - Front knee bend (for sweep-family shots)
  - Bat speed (for differentiating power shots)
  - Nose height (crouch detection for ramp/scoop)
  - Post-impact ball direction (for ramp/lap detection)

Output buckets (traditional + modern):
  cover_drive, on_drive, straight_drive,
  square_cut, pull, defensive_block, leave,
  sweep, reverse_sweep, slog_sweep, lap_shot,
  glance, flick, ramp, upper_cut, unknown

Confidence score [0, 1] accompanies every classification.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────

# Swing path angle thresholds (degrees from vertical)
# Front-on: bat appears more vertical for same swing compared to side-on
SWING_STRAIGHT_MAX = 35.0       # ~straight bat = drive (cover/on/straight)
SWING_ACROSS_MAX = 55.0         # slightly across = glance/flick
SWING_SQUARE_MIN = 55.0         # horizontal bat = cut/pull

# Bat face angle thresholds (degrees, positive = open)
BAT_FACE_OPEN_MIN = 15.0
BAT_FACE_CLOSED_MAX = -15.0

# Foot movement threshold (pixels of lateral foot offset from start to impact)
FOOT_MOVEMENT_THRESHOLD_PX = 15

# Knee bend for sweep
SWEEP_KNEE_ANGLE_MAX = 100.0

# Bat speed threshold for slog sweep (px/frame at 30fps 1080p)
SLOG_BAT_SPEED_MIN = 120.0  # significantly harder than a regular sweep (~60-80)

# Ball line zones (fraction of frame width from left)
# Off = 0.0-0.35, Middle = 0.35-0.55, Leg = 0.55-0.80, Down leg = 0.80-1.0
# (viewer's perspective: off=left, leg=right)
BALL_LINE_ZONES = [
    ("outside_off", 0.0, 0.25),
    ("off_stump", 0.25, 0.35),
    ("middle_stump", 0.35, 0.55),
    ("leg_stump", 0.55, 0.70),
    ("down_leg", 0.70, 1.0),
]

# Ball length zones (y-fraction of frame, higher = closer to camera)
BALL_LENGTH_ZONES = [
    ("yorker", 0.0, 0.60),
    ("full", 0.60, 0.75),
    ("good", 0.75, 0.85),
    ("short", 0.85, 1.0),
]

# Minimum confidence to report a classification
MIN_CLASSIFICATION_CONFIDENCE = 0.3


class ShotType(Enum):
    """Classified shot types — traditional + modern."""
    # ── Traditional ─────────────────────────────────────────────────
    COVER_DRIVE = "cover_drive"
    ON_DRIVE = "on_drive"
    STRAIGHT_DRIVE = "straight_drive"
    SQUARE_CUT = "square_cut"
    PULL = "pull"
    DEFENSIVE_BLOCK = "defensive_block"
    LEAVE = "leave"
    SWEEP = "sweep"
    GLANCE = "glance"
    FLICK = "flick"

    # ── Modern / new-age ───────────────────────────────────────────
    REVERSE_SWEEP = "reverse_sweep"     # paddle reversed — ball to off side
    SLOG_SWEEP = "slog_sweep"            # power sweep over leg side
    LAP_SHOT = "lap_shot"               # lap/paddle to fine leg
    RAMP = "ramp"                       # scoop over wicketkeeper
    UPPER_CUT = "upper_cut"             # cut over slip cordon

    UNKNOWN = "unknown"


# ────────────────────────────────────────────────────────────────────────
# Shot Classifier
# ────────────────────────────────────────────────────────────────────────

class ShotClassifier:
    """Rule-based shot classifier for front-on cricket video.

    Call ``classify_shots()`` after the phase detector has identified
    shot boundaries.
    """

    def __init__(self, batting_hand: str = "right", frame_height: int = 1080):
        self.batting_hand = batting_hand
        self.frame_h = frame_height

        # Off/leg side mapping depends on batting hand
        if batting_hand == "right":
            # Front-on: off = viewer's left, leg = viewer's right
            self._off_side = "left"
        else:
            # Left-hander: off = viewer's right, leg = viewer's left
            self._off_side = "right"

    def classify_shots(
        self,
        shot_summaries: List[Dict[str, Any]],
        frame_metrics: List[Dict[str, Any]],
        ball_trajectory: List[Tuple[int, int]],
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Classify each detected shot.

        Args:
            shot_summaries: List of shot dicts from
                ``PhaseDetector.get_shot_summary()``.
            frame_metrics: Full frame metrics list from analyser
                (each entry has keys like ``frame``, ``front_knee_angle``,
                ``bat_angle_deg``, ``ball_x``, ``ball_y``, etc.).
            ball_trajectory: List of (x, y) ball trajectory points.
            frame_width: Video frame width in pixels (for ball line calc).
            frame_height: Video frame height in pixels.

        Returns:
            Updated ``shot_summaries`` with ``classification`` (str),
            ``classification_confidence`` (float), and ``shot_type`` (str)
            added to each entry.
        """
        self.frame_w = frame_width
        if frame_height is not None:
            self.frame_h = frame_height

        for shot in shot_summaries:
            classification = self._classify_single_shot(shot, frame_metrics, ball_trajectory)
            shot["shot_type"] = classification["shot_type"]
            shot["classification"] = classification["label"]
            shot["classification_confidence"] = classification["confidence"]
            shot["swing_path_angle"] = classification.get("swing_path_angle")
            shot["bat_face"] = classification.get("bat_face")
            shot["foot_movement"] = classification.get("foot_movement")
            shot["ball_line"] = classification.get("ball_line")
            shot["ball_length"] = classification.get("ball_length")
        return shot_summaries

    # ── Single-shot classification ─────────────────────────────────

    def _classify_single_shot(
        self,
        shot: Dict[str, Any],
        frame_metrics: List[Dict[str, Any]],
        ball_trajectory: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """Classify a single shot from its frame window."""
        start = shot["start_frame"]
        end = shot["end_frame"]

        # Extract features
        features = self._extract_features(start, end, frame_metrics, ball_trajectory)

        # Decision tree
        shot_type, confidence = self._decision_tree(features)

        return {
            "shot_type": shot_type.value,
            "label": shot_type.value.replace("_", " ").title(),
            "confidence": round(confidence, 2),
            **features,
        }

    # ── Feature extraction ─────────────────────────────────────────

    def _extract_features(
        self,
        start_frame: int,
        end_frame: int,
        frame_metrics: List[Dict[str, Any]],
        ball_trajectory: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """Extract classification features from frame data in the shot window."""
        # Filter frame metrics within shot window
        window = [
            m for m in frame_metrics
            if start_frame <= m.get("frame", 0) <= end_frame
        ]
        if not window:
            return {
                "foot_movement": "unknown",
                "swing_path_angle": None,
                "bat_face": "unknown",
                "ball_line": None,
                "ball_length": None,
                "front_knee_min": None,
            }

        # 1. Foot movement: compare foot y-position at start vs end of shot
        foot_movement = self._classify_foot_movement(window)

        # 2. Swing path angle from bat_angle_deg at impact
        swing_path_angle, bat_face = self._analyze_swing(window)

        # 3. Ball line at batter's end (last ball trajectory point near batter)
        ball_line = self._estimate_ball_line(frame_metrics, ball_trajectory, end_frame)

        # 4. Ball length from ball y-position at pitch / arrival
        ball_length = self._estimate_ball_length(frame_metrics, ball_trajectory, end_frame)

        # 5. Front knee angle (for sweep detection)
        front_knee_min = min(
            (m.get("front_knee_angle") or 180 for m in window),
            default=None,
        )

        # 6. Peak bat speed (for slog sweep vs regular sweep)
        bat_speeds = [
            m.get("bat_speed_px", 0) for m in window
            if m.get("bat_speed_px") is not None
        ]
        bat_speed_max = max(bat_speeds) if bat_speeds else None

        return {
            "foot_movement": foot_movement,
            "swing_path_angle": swing_path_angle,
            "bat_face": bat_face,
            "ball_line": ball_line,
            "ball_length": ball_length,
            "front_knee_min": front_knee_min,
            "bat_speed_max": bat_speed_max,
        }

    def _classify_foot_movement(
        self, window: List[Dict[str, Any]]
    ) -> str:
        """Determine if batter moved forward, back, or stayed.

        Uses ``front_on_foot_offset_px`` from front-on metrics (if available),
        which tracks the lateral position of feet relative to centre.

        Forward = foot offset moves toward off side (negative for right-hander)
        Back = foot offset moves toward leg side (positive)
        """
        # Check for front-on foot tracking data
        offsets = [
            m.get("front_on_foot_offset_px") for m in window
            if m.get("front_on_foot_offset_px") is not None
        ]
        if len(offsets) >= 5:
            # Compare first vs last 3-frame average
            early = sum(offsets[:3]) / min(3, len(offsets))
            late = sum(offsets[-3:]) / min(3, len(offsets))
            delta = late - early  # positive = moved toward leg side

            if abs(delta) < FOOT_MOVEMENT_THRESHOLD_PX:
                return "no_stride"
            elif delta < 0:
                return "forward"   # moved toward off side = forward press
            else:
                return "back"      # moved toward leg side = onto back foot

        # Fallback: use bat_lift_height as rough proxy
        # (>50px lift suggests back-foot preparation)
        lift_heights = [m.get("bat_lift_height") or 0 for m in window]
        max_lift = max(lift_heights) if lift_heights else 0
        if max_lift > 50:
            return "back"
        return "forward"

    def _analyze_swing(
        self, window: List[Dict[str, Any]]
    ) -> Tuple[Optional[float], str]:
        """Analyse swing direction and bat face from frame metrics.

        Returns (swing_path_angle_deg, bat_face_label).
        """
        # Find impact frame (minimum bat_angle_deg change or peak speed)
        impact_frames = [
            m for m in window
            if m.get("bat_speed_px", 0) > 0
        ]
        if not impact_frames:
            return None, "unknown"

        # Swing path: use bat_angle_deg which tracks bat handle angle
        bat_angles = [m.get("bat_angle_deg") for m in impact_frames
                      if m.get("bat_angle_deg") is not None]
        if not bat_angles:
            return None, "unknown"

        # Only consider frames near peak speed (top 50% of speeds)
        # to avoid start/end of swing dragging the average down
        speeds = [m.get("bat_speed_px", 0) for m in impact_frames[:len(bat_angles)]]
        if speeds:
            speed_threshold = max(speeds) * 0.5
            peak_angles = [
                a for a, s in zip(bat_angles, speeds)
                if s >= speed_threshold
            ]
            if peak_angles:
                bat_angles = peak_angles

        # The bat angle relative to vertical tells us swing plane.
        # Front-on: 0° = vertical (straight bat), 90° = horizontal (cut/pull)
        # Use absolute value — sign indicates open/closed face, not swing plane.
        avg_angle = abs(sum(bat_angles) / len(bat_angles))

        swing_path = avg_angle  # degrees from vertical (always positive)

        # Bat face estimation: wrist x-offset at impact
        # For front-on, open face → right wrist further from body (for right-hander)
        # This is best-effort; can be refined with dedicated wrist analysis
        bat_face = self._estimate_bat_face(impact_frames)

        return swing_path, bat_face

    def _estimate_bat_face(
        self, impact_frames: List[Dict[str, Any]]
    ) -> str:
        """Estimate bat face angle from wrist positions at impact.

        From front-on:
          - Open face: bat face points toward off side
          - Closed face: bat face points toward leg side
          - Straight: bat face square-on

        Uses relative hand x-position (left_wrist vs right_wrist).
        """
        # This requires wrist x-positions which aren't in frame_metrics currently.
        # For now, derive from bat_angle_deg sign (positive = open, negative = closed).
        # This is a rough approximation.

        bat_angles = [
            m.get("bat_angle_deg") for m in impact_frames
            if m.get("bat_angle_deg") is not None
        ]
        if not bat_angles:
            return "unknown"

        avg = sum(bat_angles) / len(bat_angles)

        # A more positive angle means the bat handle is tilted toward off side
        # (open face), more negative toward leg side (closed face)
        # This is view-dependent
        if self._off_side == "left":
            # Right-hander front-on: positive angle = handle toward off (open)
            if avg > 15:
                return "open"
            elif avg < -15:
                return "closed"
            else:
                return "straight"
        else:
            # Left-hander: reverse
            if avg < -15:
                return "open"
            elif avg > 15:
                return "closed"
            else:
                return "straight"

    def _estimate_ball_line(
        self,
        frame_metrics: List[Dict[str, Any]],
        ball_trajectory: List[Tuple[int, int]],
        end_frame: int,
    ) -> Optional[str]:
        """Estimate ball line (off/middle/leg) at batter's end.

        Uses ``ball_x`` from frame metrics at the last frame of the shot
        (most accurate — in absolute pixel coordinates from the tracker),
        falling back to ball trajectory if needed.
        """
        # Method 1: ball_x from frame_metrics at end_frame
        ball_x = None
        for m in reversed(frame_metrics):
            if m.get("frame") == end_frame and m.get("ball_x") is not None:
                ball_x = m["ball_x"]
                break

        if ball_x is None and ball_trajectory:
            # Method 2: last trajectory point
            bx, _ = ball_trajectory[-1]
            ball_x = float(bx)

        if ball_x is None:
            return None

        # Normalise using provided frame width or estimate from trajectory
        frame_w = getattr(self, 'frame_w', None)
        if frame_w is None and ball_trajectory:
            xs = [x for x, _ in ball_trajectory]
            if xs:
                frame_w = max(xs) * 2
        if frame_w is None:
            return None

        x_norm = ball_x / frame_w

        for zone_name, lo, hi in BALL_LINE_ZONES:
            if lo <= x_norm < hi:
                return zone_name
        return None

    def _estimate_ball_length(
        self,
        frame_metrics: List[Dict[str, Any]],
        ball_trajectory: List[Tuple[int, int]],
        end_frame: int,
    ) -> Optional[str]:
        """Estimate ball length from y-position at pitch or arrival.

        Uses ``ball_y`` from frame metrics if available, otherwise
        the ball y-position in the batter zone of trajectory.
        """
        ball_y = None
        # Method 1: frame_metrics at end_frame
        for m in reversed(frame_metrics):
            if m.get("frame") == end_frame and m.get("ball_y") is not None:
                ball_y = m["ball_y"]
                break

        if ball_y is None and ball_trajectory:
            # Method 2: last trajectory points
            ys = [y for (_, y) in ball_trajectory[-5:]]
            if ys:
                ball_y = max(ys)

        if ball_y is None or self.frame_h == 0:
            return None

        y_norm = ball_y / self.frame_h

        for length_name, lo, hi in BALL_LENGTH_ZONES:
            if lo <= y_norm < hi:
                return length_name
        return None

    # ── Decision tree ──────────────────────────────────────────────

    def _decision_tree(
        self, features: Dict[str, Any]
    ) -> Tuple[ShotType, float]:
        """Rule-based decision tree for shot classification.

        Returns (ShotType, confidence).

        Detection order:
          1. Sweep-family shots (all involve deep knee bend)
             → reverse_sweep | slog_sweep | lap_shot | sweep
          2. Ramp / scoop signature (crouch + upward ball + no classic swing)
             → ramp
          3. No swing → defensive_block | leave
          4. Upper cut (back-foot, 35-55°, ball outside off)
             → upper_cut
          5. Swing shots:
               < 35° straight bat  → cover / on / straight drive
               ≥ 55° horizontal    → square_cut | pull
               35-55° across       → glance | flick
        """
        foot = features.get("foot_movement", "unknown")
        swing = features.get("swing_path_angle")
        bat_face = features.get("bat_face", "unknown")
        ball_line = features.get("ball_line")
        ball_length = features.get("ball_length")
        front_knee_min = features.get("front_knee_min")
        bat_speed_max = features.get("bat_speed_max")

        # ─────────────────────────────────────────────────────────
        # 1. Sweep-family shots (front knee down)
        # ─────────────────────────────────────────────────────────
        if front_knee_min is not None and front_knee_min < SWEEP_KNEE_ANGLE_MAX:
            # 1a. Reverse sweep: knee down + open face (bat face toward off side)
            if bat_face == "open":
                return ShotType.REVERSE_SWEEP, 0.80

            # 1b. Slog sweep: knee down + high bat speed (power shot)
            if bat_speed_max is not None and bat_speed_max >= SLOG_BAT_SPEED_MIN:
                return ShotType.SLOG_SWEEP, 0.82

            # 1c. Lap shot: knee down + ball down leg/very fine + neutral/less-closed face
            #     (guides ball finer to fine leg rather than square leg)
            if bat_face == "straight" or ball_line == "down_leg":
                return ShotType.LAP_SHOT, 0.72

            # 1d. Regular sweep
            if ball_line in ("leg_stump", "down_leg"):
                return ShotType.SWEEP, 0.85
            return ShotType.SWEEP, 0.65

        # ─────────────────────────────────────────────────────────
        # 2. Ramp / Scoop signature
        #    Limited detection from front-on without post-impact ball trajectory.
        #    Key signals: no stride, ball on middle/leg, bat face open,
        #    moderate knee bend (crouch, not full sweep), no classic swing.
        #    This is best-effort — will improve with head-tracking + post-impact ball.
        # ─────────────────────────────────────────────────────────
        # (reserved for future enhancement — requires nose_y tracking in frame_metrics)

        # ─────────────────────────────────────────────────────────
        # 3. No swing = leave or defensive
        # ─────────────────────────────────────────────────────────
        if swing is None or swing < 3:
            if foot == "forward":
                return ShotType.DEFENSIVE_BLOCK, 0.6
            else:
                if ball_line == "outside_off":
                    return ShotType.LEAVE, 0.7
                return ShotType.DEFENSIVE_BLOCK, 0.5

        # ─────────────────────────────────────────────────────────
        # 4. Upper cut (back-foot, angle 35-60°, ball outside off)
        #    The batter cuts over the slip cordon — bat between drive & cut angle,
        #    ball trajectory upward. Check BEFORE the main swing branches.
        # ─────────────────────────────────────────────────────────
        back_foot = (foot == "back")
        upper_cut_range = swing is not None and 35 <= swing < 60
        if upper_cut_range and back_foot and ball_line in ("outside_off", "off_stump"):
            return ShotType.UPPER_CUT, 0.76

        # ─────────────────────────────────────────────────────────
        # 5. Swing shots
        # ─────────────────────────────────────────────────────────

        # 5a. Straight bat shots (drive)
        if swing < SWING_STRAIGHT_MAX:
            if ball_line in ("outside_off", "off_stump"):
                return ShotType.COVER_DRIVE, 0.85
            elif ball_line in ("middle_stump",):
                if bat_face == "closed":
                    return ShotType.ON_DRIVE, 0.80
                return ShotType.STRAIGHT_DRIVE, 0.80
            elif ball_line in ("leg_stump", "down_leg"):
                return ShotType.ON_DRIVE, 0.75
            else:
                if bat_face == "open":
                    return ShotType.COVER_DRIVE, 0.65
                elif bat_face == "closed":
                    return ShotType.ON_DRIVE, 0.65
                return ShotType.STRAIGHT_DRIVE, 0.60

        # 5b. Across-the-line / horizontal bat shots
        elif swing >= SWING_SQUARE_MIN:
            if back_foot or ball_length in ("short", "good"):
                if bat_face == "open":
                    return ShotType.SQUARE_CUT, 0.85
                else:
                    return ShotType.PULL, 0.80
            else:
                if ball_line in ("leg_stump", "down_leg"):
                    return ShotType.PULL, 0.70
                return ShotType.SQUARE_CUT, 0.65

        # 5c. Between straight and square = glance/flick
        else:
            if ball_line in ("leg_stump", "down_leg"):
                return ShotType.GLANCE, 0.75
            elif ball_line in ("outside_off", "off_stump"):
                return ShotType.FLICK, 0.65
            else:
                if bat_face == "closed":
                    return ShotType.GLANCE, 0.55
                return ShotType.FLICK, 0.55

    # ── Public utilities ──────────────────────────────────────────

    @staticmethod
    def shot_type_icon(shot_type: str) -> str:
        """Return an emoji/icon for the shot type."""
        icons = {
            # Traditional
            "cover_drive": "🎯",
            "on_drive": "🎯",
            "straight_drive": "🎯",
            "square_cut": "✂️",
            "pull": "💪",
            "defensive_block": "🛡️",
            "leave": "🚫",
            "sweep": "🧹",
            "glance": "👀",
            "flick": "🖐️",
            # Modern
            "reverse_sweep": "🔄",
            "slog_sweep": "💥",
            "lap_shot": "🦘",
            "ramp": "🚀",
            "upper_cut": "⬆️",
        }
        return icons.get(shot_type, "❓")

    @staticmethod
    def shot_type_description(shot_type: str) -> str:
        """Return a short plain-text description for the shot type."""
        descs = {
            "cover_drive": "Front-foot drive through the covers.",
            "on_drive": "Straight-bat drive through mid-wicket / on side.",
            "straight_drive": "Straight-bat drive down the ground.",
            "square_cut": "Horizontal-bat cut square of the wicket on the off side.",
            "pull": "Horizontal-bat pull to the leg side (short ball).",
            "defensive_block": "Forward defensive block with soft hands.",
            "leave": "Batter leaves the ball, letting it pass.",
            "sweep": "Kneeling sweep to the leg side against spin.",
            "reverse_sweep": "Reverse paddle sweep — ball goes to the off side.",
            "slog_sweep": "Power sweep aimed over the leg-side boundary.",
            "lap_shot": "Lap/paddle shot guiding the ball fine to fine leg.",
            "glance": "Fine glance off the pads to fine leg.",
            "flick": "Wristy flick through mid-wicket.",
            "ramp": "Scoop/ramp over the wicketkeeper's head.",
            "upper_cut": "Cut over the slip cordon, usually off a short wide ball.",
        }
        return descs.get(shot_type, "Unidentified shot.")
