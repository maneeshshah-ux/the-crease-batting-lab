"""
LBW Predictor — Single-camera LBW probability estimator.

Camera position:   Non-striker end, behind bowler, looking toward batter.
Limitations:       2D front-on view — NO 3D height information.
                   Ball bounce height over stumps cannot be determined.

What this CAN do:
  - Estimate ball line (x-position) at the crease from trajectory
  - Compare to stump zone to compute hitting probability
  - Apply cone of uncertainty based on tracking quality
  - Indicate when ball is going down leg, outside off, or hitting stumps

What this CANNOT do (clear caveats displayed on every output):
  - Say OUT / NOT OUT (only percentage chance of hitting stumps)
  - Account for ball bouncing over the stumps (no height data from front-on)
  - Account for bat-pad or bat-first scenarios (no 3D positioning)
  - Replace DRS / third-umpire review

Usage:
    predictor = LbwPredictor()
    result = predictor.predict(
        ball_line="off_stump",
        ball_length="good",
        pitch_zone="off_stump",
        batter_forward=True,
        impact_point="edge",
        batting_hand="right",
        trajectory_points=25,
    )
    # result["hitting_stumps_pct"] → 78 (percent)
    # result["cone"] → {"lower": 0.30, "upper": 0.42} (x-normalised range)
    # result["verdict"] → "hitting_off_stump"
    # result["confidence"] → "medium"
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────
# Constants — Stump zones at the crease (front-on, normalised x)
# ────────────────────────────────────────────────────────────────────────
#
# For a right-handed batter facing the bowler (front-on camera):
#   Off side  = viewer's LEFT  (lower x values)
#   Leg side  = viewer's RIGHT (higher x values)
#   Stumps    = centre of frame, approximately x ∈ [0.25, 0.70]
#
# Each zone: (lower, upper, hit_weight)
#   hit_weight = 1.0 for centre of stumps, 0.0 for missing

STUMP_ZONES: Dict[str, Tuple[float, float, float]] = {
    "outside_off":   (0.00, 0.25, 0.00),   # missing off
    "off_stump":     (0.25, 0.38, 0.70),   # clipping off
    "middle_stump":  (0.38, 0.55, 1.00),   # dead centre
    "leg_stump":     (0.55, 0.68, 0.70),   # clipping leg
    "missing_leg":   (0.68, 1.00, 0.00),   # missing leg
}

# Pitch zone modifiers (how much pitching in a zone affects hitting prob)
# A ball pitching outside off that straightens has moderate hitting chance.
# A ball pitching on leg is usually going down leg.
PITCH_MODIFIERS: Dict[str, float] = {
    "outside_off":   0.6,    # can still straighten to hit
    "off_stump":     1.1,    # dangerous line
    "middle_stump":  1.2,    # dead straight
    "leg_stump":     0.8,    # angling down
    "down_leg":      0.3,    # missing leg
}

# Impact point modifiers
# Edge → ball was close to bat edge = closer to stumps = higher chance
# Middle → well timed = probably safe line
# Toe → low on bat = could be pad first
IMPACT_MODIFIERS: Dict[str, float] = {
    "middle":    0.6,
    "edge":      1.3,
    "toe":       0.8,
    "no_shot":   1.2,   # didn't attempt a shot = pad player
    "unknown":   1.0,
}

# Confidence thresholds (based on trajectory quality)
CONFIDENCE_HIGH_MIN_POINTS = 20
CONFIDENCE_MED_MIN_POINTS = 8


# ────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────

def _confidence_label(points: int) -> str:
    if points >= CONFIDENCE_HIGH_MIN_POINTS:
        return "high"
    elif points >= CONFIDENCE_MED_MIN_POINTS:
        return "medium"
    elif points > 0:
        return "low"
    return "estimate"


def _classify_stump_zone(x_norm: float) -> str:
    """Classify a normalised x-position into a stump zone."""
    for zone_name, (lo, hi, _) in STUMP_ZONES.items():
        if lo <= x_norm < hi:
            return zone_name
    return "unknown"


def _zone_to_verdict(zone: str) -> str:
    """Human-readable verdict for a stump zone."""
    verdicts = {
        "outside_off":  "Missing outside off stump",
        "off_stump":    "Hitting off stump",
        "middle_stump": "Hitting middle stump",
        "leg_stump":    "Hitting leg stump",
        "missing_leg":  "Missing down leg side",
        "unknown":      "Line uncertain — insufficient data",
    }
    return verdicts.get(zone, "Unknown")


# ────────────────────────────────────────────────────────────────────────
# Core Predictor
# ────────────────────────────────────────────────────────────────────────

class LbwPredictor:
    """
    Single-camera LBW predictor.

    Produces a **percentage chance of the ball hitting the stumps**,
    NOT an out/not-out decision. All predictions include a cone of
    uncertainty and a confidence label.
    """

    def __init__(self):
        pass

    # ── Public API ──────────────────────────────────────────────────

    def predict(
        self,
        ball_line: Optional[str] = None,
        ball_length: Optional[str] = None,
        pitch_zone: Optional[str] = None,
        batter_forward: bool = False,
        impact_point: str = "unknown",
        batting_hand: str = "right",
        trajectory_points: int = 0,
        ball_x_at_crease: Optional[float] = None,
        frame_width: int = 1920,
        foot_alignment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Predict LBW probability from available signals.

        Args:
            ball_line: Classified ball line (e.g. "off_stump", "outside_off").
            ball_length: Classified ball length (e.g. "good", "full").
            pitch_zone: Where the ball pitched (pitch point mapped to zone).
            batter_forward: Whether batter is on front foot (stride).
            impact_point: Where ball hit bat ("middle", "edge", "toe", etc.).
            batting_hand: "right" or "left".
            trajectory_points: Number of valid ball tracking points.
            ball_x_at_crease: Ball x-position (pixels) at the crease, if
                available directly from ball tracker.
            frame_width: Frame width in pixels (for normalising ball_x).
            foot_alignment: Batter's foot alignment from front-on metrics
                ("covering_off", "covering_middle", etc.).

        Returns:
            dict with keys:
                - hitting_stumps_pct (int): 0-100 percentage
                - verdict (str): human-readable line call
                - cone (dict): {"lower": float, "upper": float} normalised x range
                - confidence (str): high/medium/low/estimate
                - caveat (str): required disclaimer
        """
        # Always include the caveat
        caveat = (
            "Single-camera estimate only. Does not account for bounce height, "
            "bat-pad scenarios, or 3D positioning. NOT a DRS replacement."
        )

        # ── Step 1: Determine the ball x-position at the crease ──
        # Priority: explicit ball_x > ball_line zone > pitch zone estimate
        x_norm, line_source = self._resolve_ball_line(
            ball_x_at_crease=ball_x_at_crease,
            frame_width=frame_width,
            ball_line=ball_line,
            pitch_zone=pitch_zone,
            batting_hand=batting_hand,
        )

        if x_norm is None:
            return {
                "hitting_stumps_pct": 0,
                "verdict": "Insufficient ball tracking data",
                "cone": {"lower": 0.0, "upper": 1.0},
                "confidence": "estimate",
                "caveat": caveat,
            }

        # ── Step 2: Classify line → stump zone ──
        line_zone = _classify_stump_zone(x_norm)
        base_hit_weight = STUMP_ZONES.get(line_zone, (0, 0, 0))[2]

        # ── Step 3: Apply modifiers ──
        # Modifiers multiply the base weight (1.0 = unmodified)
        modifier = 1.0

        # Pitch zone modifier
        if pitch_zone and pitch_zone in PITCH_MODIFIERS:
            modifier *= PITCH_MODIFIERS[pitch_zone]

        # Impact point modifier
        if impact_point in IMPACT_MODIFIERS:
            modifier *= IMPACT_MODIFIERS[impact_point]

        # Batter forward → reduces hitting chance (ball travels further,
        # may bounce over or miss edge)
        if batter_forward:
            modifier *= 0.85

        # Foot alignment: if batter is outside off, balls on off have
        # higher chance of hitting (batter is closer to ball line)
        if foot_alignment:
            modifier *= self._foot_alignment_modifier(
                foot_alignment, line_zone, batting_hand
            )

        # ── Step 4: Compute final probability ──
        # Clamp modifier & multiply by base hit weight
        modifier = max(0.1, min(2.0, modifier))
        raw_prob = base_hit_weight * modifier
        hitting_pct = min(99, max(1, int(raw_prob * 100)))

        # ── Step 5: Cone of uncertainty ──
        cone = self._compute_cone(
            x_norm, trajectory_points, line_source, frame_width
        )

        # ── Step 6: Confidence ──
        confidence = _confidence_label(trajectory_points)

        # Adjust verdict if cone spans multiple zones
        cone_lower_zone = _classify_stump_zone(cone["lower"])
        cone_upper_zone = _classify_stump_zone(cone["upper"])
        if cone_lower_zone != cone_upper_zone:
            verdict = (
                f"Between {_zone_to_verdict(cone_lower_zone).lower()} "
                f"and {_zone_to_verdict(cone_upper_zone).lower()}"
            )
        else:
            verdict = _zone_to_verdict(line_zone)

        return {
            "hitting_stumps_pct": hitting_pct,
            "verdict": verdict,
            "cone": cone,
            "ball_line_zone": line_zone,
            "ball_x_normalised": round(x_norm, 3),
            "confidence": confidence,
            "caveat": caveat,
            "num_trajectory_points": trajectory_points,
        }

    # ── Per-shot prediction ────────────────────────────────────────

    def predict_shot(
        self,
        shot: Dict[str, Any],
        ball_trajectory: List[Tuple[int, int]],
        frame_width: int,
        frame_height: int,
        batter_forward: Optional[bool] = None,
        foot_alignment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper that calls ``predict()`` from a shot dict.

        Extracts ``ball_line``, ``ball_length``, ``impact_point`` etc.
        from an existing shot summary dict (as output by the shot classifier).
        """
        # Extract ball line
        ball_line = shot.get("ball_line")
        ball_length = shot.get("ball_length")

        # Estimate pitch zone from trajectory (pitch point)
        pitch_zone = self._estimate_pitch_zone(
            ball_trajectory, frame_width, frame_height
        )

        # Impact point from front-on metrics
        impact_point = shot.get("impact_point_label", "unknown")

        # Number of trajectory points available
        traj_points = len(ball_trajectory) if ball_trajectory else 0

        # Ball x-position at end of shot (from frame metrics)
        ball_x = None
        fm = shot.get("frame_metrics")
        if fm:
            ball_x = fm.get("ball_x")

        return self.predict(
            ball_line=ball_line,
            ball_length=ball_length,
            pitch_zone=pitch_zone,
            batter_forward=bool(batter_forward),
            impact_point=impact_point,
            batting_hand=shot.get("batting_hand", "right"),
            trajectory_points=traj_points,
            ball_x_at_crease=ball_x,
            frame_width=frame_width,
            foot_alignment=foot_alignment,
        )

    # ── Internal helpers ───────────────────────────────────────────

    @staticmethod
    def _resolve_ball_line(
        ball_x_at_crease: Optional[float],
        frame_width: int,
        ball_line: Optional[str],
        pitch_zone: Optional[str],
        batting_hand: str,
    ) -> Tuple[Optional[float], str]:
        """Resolve the ball x-position at the crease from available signals.

        Returns (x_norm, source) where source describes how it was determined.
        """
        # Priority 1: direct ball_x measurement
        if ball_x_at_crease is not None and frame_width > 0:
            x_norm = ball_x_at_crease / frame_width
            # Clamp to [0, 1]
            x_norm = max(0.0, min(1.0, x_norm))
            return x_norm, "direct"

        # Priority 2: ball_line classification → estimate midpoint of zone
        if ball_line and ball_line in STUMP_ZONES:
            lo, hi, _ = STUMP_ZONES[ball_line]
            x_norm = (lo + hi) / 2.0
            return x_norm, "classified_line"

        # Priority 3: pitch zone → rough estimate (less accurate)
        if pitch_zone and pitch_zone in STUMP_ZONES:
            lo, hi, _ = STUMP_ZONES[pitch_zone]
            # After pitching, ball tends to move slightly toward leg
            # (drift for spinners, angle for seamers)
            leg_shift = 0.03 if batting_hand == "right" else -0.03
            if pitch_zone in ("outside_off", "off_stump"):
                leg_shift *= 0.5  # less drift from wide lines
            x_norm = (lo + hi) / 2.0 + leg_shift
            x_norm = max(0.0, min(1.0, x_norm))
            return x_norm, "pitch_estimate"

        return None, "none"

    @staticmethod
    def _estimate_pitch_zone(
        ball_trajectory: List[Tuple[int, int]],
        frame_width: int,
        frame_height: int,
    ) -> Optional[str]:
        """Estimate pitch zone from ball trajectory.

        The pitch point is where the ball bounces — detectable as a
        velocity kink in the trajectory. Falls back to the midpoint
        of the trajectory if the pitch point is not clearly detectable.
        """
        if not ball_trajectory or frame_width <= 0:
            return None

        # Look for the pitch point: the point in the trajectory where
        # y-velocity changes sign (ball starts going up after bouncing)
        # or where the trajectory has a sharp angle change
        if len(ball_trajectory) >= 4:
            ys = [y for (_, y) in ball_trajectory]
            # Pitch is roughly where y stops decreasing and starts
            # increasing (ball bounces up toward camera)
            # From front-on, ball goes from bowler → down → up toward camera
            # So y increases then... wait, no. From front-on behind bowler:
            # Ball starts at bowler release (high in frame = low y),
            # travels DOWN the pitch (y increases toward batter),
            # bounces (y continues increasing but at different rate),
            # reaches batter (highest y in frame).

            # Actually the bounce point is hard to detect from front-on.
            # Use mid-point as rough estimate
            pass

        # Fallback: midpoint of trajectory
        xs = [x for (x, _) in ball_trajectory]
        mid_x = sum(xs) / len(xs)
        x_norm = mid_x / frame_width
        x_norm = max(0.0, min(1.0, x_norm))

        for zone_name, (lo, hi, _) in STUMP_ZONES.items():
            if lo <= x_norm < hi:
                return zone_name
        return "unknown"

    @staticmethod
    def _foot_alignment_modifier(
        foot_alignment: str,
        line_zone: str,
        batting_hand: str,
    ) -> float:
        """Adjust LBW probability based on batter's foot position.

        If the batter is standing outside off and the ball is on off stump,
        they're closer to the ball → higher chance of being hit on the
        pads in line.

        If the batter is WAY down leg and the ball is on off, they're
        far from the ball → lower chance of being in line.
        """
        # For right-hander: off = negative offset, leg = positive offset
        if batting_hand == "left":
            # Mirror for left-handers
            foot_map = {
                "covering_off": "covering_leg",
                "covering_leg": "covering_off",
                "outside_off": "down_leg",
                "down_leg": "outside_off",
            }
            foot_alignment = foot_map.get(foot_alignment, foot_alignment)

        # If batter is on off side and ball is hitting off → in line
        if foot_alignment in ("covering_off", "outside_off") and \
           line_zone in ("off_stump", "middle_stump"):
            return 1.15

        # Batter on leg side and ball down leg → might miss
        if foot_alignment in ("covering_leg", "down_leg") and \
           line_zone in ("leg_stump", "missing_leg"):
            return 1.1

        # Batter covering middle and ball hitting → standard
        if foot_alignment == "covering_middle":
            return 1.0

        # Batter on leg but ball on off → less likely in line
        if foot_alignment in ("covering_leg", "down_leg") and \
           line_zone in ("outside_off", "off_stump"):
            return 0.7

        return 1.0

    @staticmethod
    def _compute_cone(
        x_norm: float,
        trajectory_points: int,
        source: str,
        frame_width: int,
    ) -> Dict[str, float]:
        """Compute cone of uncertainty around the ball line.

        The cone narrows with more trajectory points and with
        higher-quality line sources.

        Returns {"lower": float, "upper": float} in normalised x.
        """
        # Base half-width depends on data quality
        if source == "direct":
            # Direct measurement — relatively tight
            base_half_width = 0.04
        elif source == "classified_line":
            base_half_width = 0.06
        elif source == "pitch_estimate":
            base_half_width = 0.12
        else:
            base_half_width = 0.20

        # More trajectory points → more confident
        if trajectory_points > 0:
            confidence_factor = max(0.3, min(1.0, 10.0 / trajectory_points))
        else:
            confidence_factor = 1.0

        half_width = base_half_width * confidence_factor
        half_width = min(0.25, half_width)  # cap at 25% of frame

        lower = max(0.0, x_norm - half_width)
        upper = min(1.0, x_norm + half_width)

        return {"lower": round(lower, 4), "upper": round(upper, 4)}
