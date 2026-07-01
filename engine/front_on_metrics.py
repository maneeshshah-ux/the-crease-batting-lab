"""
Front-on Specific Metrics — Batting metrics unique to the front-on view.

Camera: non-striker end, behind bowler, looking toward batter.

These metrics are supplementary to the general biomechanical metrics in
``metrics.py``.  They capture cricket-specific observations that only
make sense from the front-on perspective:

  1. BAT FACE ANGLE    — Open / closed / straight at impact.
  2. FOOT-STUMP ALIGNMENT — Lateral position of feet relative to stumps.
  3. LATERAL TRIGGER   — Pre-delivery movement across the crease.
  4. HEAD-LINE SYNC    — Head position relative to ball release / impact.
  5. BALANCE DIRECTION — Which way the batter leans at impact.
  6. IMPACT POINT      — Where the ball hits the bat (middle / toe / edge).
  7. SHOULDER ALIGNMENT — Shoulder line relative to pitch at stance.

Each metric includes a confidence label (high / medium / low / estimate).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────
# Confidence helpers
# ────────────────────────────────────────────────────────────────────────

def _conf_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    elif score >= 0.5:
        return "medium"
    elif score >= 0.25:
        return "low"
    return "estimate"


# ────────────────────────────────────────────────────────────────────────
# Bat Face Angle
# ────────────────────────────────────────────────────────────────────────

def estimate_bat_face(
    landmarks: Dict[str, Any],
    batting_hand: str = "right",
    camera_view: str = "front_on",
) -> Dict[str, Any]:
    """Estimate bat face angle from wrist landmark positions.

    From front-on, the bat face angle is inferred from the relative
    orientation of the two hands.  The line connecting the wrists
    approximates the bat handle orientation; the face is perpendicular
    to this line.

    Args:
        landmarks: Pose landmark dict with LEFT_WRIST, RIGHT_WRIST.
        batting_hand: 'right' or 'left'.
        camera_view: 'front_on', 'side_off', 'side_leg', 'angled'.

    Returns:
        dict with keys:
            bat_face_angle (float): degrees, positive = open, negative = closed.
            bat_face_label (str): 'open', 'closed', 'straight'.
            confidence_label (str): 'high' | 'medium' | 'low' | 'estimate'.
    """
    if camera_view != "front_on":
        return {
            "bat_face_angle": None,
            "bat_face_label": "unknown",
            "confidence_label": "estimate",
            "note": "Bat face estimation is front-on specific",
        }

    lw = landmarks.get("LEFT_WRIST")
    rw = landmarks.get("RIGHT_WRIST")
    if not lw or not rw:
        return {
            "bat_face_angle": None,
            "bat_face_label": "unknown",
            "confidence_label": "low",
        }

    # Vector from left wrist (top hand) to right wrist (bottom hand)
    dx = rw.get("x", 0) - lw.get("x", 0)
    dy = rw.get("y", 0) - lw.get("y", 0)

    if abs(dy) < 0.001:
        return {
            "bat_face_angle": 0,
            "bat_face_label": "straight",
            "confidence_label": "low",
        }

    # Angle of wrist line relative to vertical (image y-axis)
    angle_rad = math.atan2(dx, dy)
    angle_deg = math.degrees(angle_rad)

    # For right-hander front-on:
    #   Positive angle = right wrist is to the right of left wrist
    #     → bottom hand further from body → bat face open → off side
    #   Negative angle = right wrist is to the left of left wrist
    #     → bat face closed → leg side
    if batting_hand == "left":
        angle_deg = -angle_deg  # mirror for left-handers

    # Determine label
    if angle_deg > 15:
        label = "open"
    elif angle_deg < -15:
        label = "closed"
    else:
        label = "straight"

    # Confidence: bat face estimation from front-on is moderate
    # because wrist landmarks don't perfectly capture handle rotation
    conf = min(1.0, max(0.3, abs(angle_deg) / 45.0 + 0.4))

    return {
        "bat_face_angle": round(angle_deg, 1),
        "bat_face_label": label,
        "confidence_label": _conf_label(conf),
    }


# ────────────────────────────────────────────────────────────────────────
# Foot-Stump Alignment
# ────────────────────────────────────────────────────────────────────────

def estimate_foot_stump_alignment(
    landmarks: Dict[str, Any],
    frame_width: int,
) -> Dict[str, Any]:
    """Estimate lateral alignment of feet relative to stumps.

    From front-on, the stumps are approximately at frame centre.
    The batter's feet x-positions relative to centre indicate alignment.

    Args:
        landmarks: Pose dict with foot landmarks.
        frame_width: Width of the video frame in pixels.

    Returns:
        dict with keys:
            lateral_offset (float): pixels from centre (negative = off side,
                                    positive = leg side).
            alignment_label (str): 'covering_off', 'covering_middle',
                                   'covering_leg', 'outside_off', 'down_leg'.
            confidence_label (str).
    """
    # Get key foot landmarks
    lf = landmarks.get("LEFT_FOOT_INDEX") or landmarks.get("LEFT_ANKLE")
    rf = landmarks.get("RIGHT_FOOT_INDEX") or landmarks.get("RIGHT_ANKLE")
    if not lf or not rf:
        return {"lateral_offset": None, "alignment_label": "unknown",
                "confidence_label": "low"}

    # Midpoint of feet
    mid_x = (lf.get("x", 0) + rf.get("x", 0)) / 2.0

    # Convert to pixel offset from centre
    centre_x = 0.5
    offset_px = (mid_x - centre_x) * frame_width

    # Stumps are approximately at centre (frame_x ≈ 0.5)
    # Off side = negative (left of centre for right-hander front-on)
    if abs(offset_px) < 15:
        label = "covering_middle"
    elif offset_px < -15:
        label = "covering_off" if offset_px > -40 else "outside_off"
    else:
        label = "covering_leg" if offset_px < 40 else "down_leg"

    return {
        "lateral_offset_px": round(offset_px, 1),
        "alignment_label": label,
        "confidence_label": "medium",
    }


# ────────────────────────────────────────────────────────────────────────
# Lateral Trigger Movement
# ────────────────────────────────────────────────────────────────────────

def estimate_lateral_trigger(
    stance_landmarks: Dict[str, Any],
    impact_landmarks: Dict[str, Any],
    frame_width: int,
) -> Dict[str, Any]:
    """Measure lateral movement (across the crease) from stance to impact.

    Args:
        stance_landmarks: Pose at stance (pre-delivery).
        impact_landmarks: Pose at or near impact.

    Returns:
        dict with keys:
            lateral_movement_px (float).
            direction (str): 'toward_off', 'toward_leg', 'minimal'.
            is_across (bool): True if > 20 px movement toward off.
            confidence_label (str).
    """
    def _foot_mid_x(lms):
        lf = lms.get("LEFT_FOOT_INDEX") or lms.get("LEFT_ANKLE", {})
        rf = lms.get("RIGHT_FOOT_INDEX") or lms.get("RIGHT_ANKLE", {})
        lx = lf.get("x", 0.5) if isinstance(lf, dict) else 0.5
        rx = rf.get("x", 0.5) if isinstance(rf, dict) else 0.5
        return (lx + rx) / 2.0

    start_x = _foot_mid_x(stance_landmarks)
    impact_x = _foot_mid_x(impact_landmarks)

    movement_px = (impact_x - start_x) * frame_width

    if abs(movement_px) < 10:
        direction = "minimal"
    elif movement_px < 0:
        direction = "toward_off"
    else:
        direction = "toward_leg"

    return {
        "lateral_movement_px": round(movement_px, 1),
        "direction": direction,
        "is_across": movement_px < -20,
        "confidence_label": "medium",
    }


# ────────────────────────────────────────────────────────────────────────
# Head-Line Sync
# ────────────────────────────────────────────────────────────────────────

def estimate_head_line_sync(
    landmarks: Dict[str, Any],
    ball_x: Optional[float] = None,
    frame_width: int = 1920,
) -> Dict[str, Any]:
    """Check if the batter's head is in line with the ball / stumps.

    From front-on, head-x relative to the ball line determines if the
    batter is "in line" with the delivery.

    Args:
        landmarks: Pose dict with NOSE landmark.
        ball_x: Ball x-position at batter's end (pixels or None).
        frame_width: Frame width in pixels.

    Returns:
        dict with keys:
            head_offset_px (float): head x from centre.
            head_ball_offset_px (float|None): head x from ball x.
            head_behind_ball (bool|None): True if head is behind the ball line.
            confidence_label (str).
    """
    nose = landmarks.get("NOSE")
    if not nose:
        return {"head_offset_px": None, "confidence_label": "low"}

    head_x = nose.get("x", 0.5) * frame_width
    centre_x = frame_width / 2
    head_offset = head_x - centre_x

    result = {
        "head_offset_px": round(head_offset, 1),
        "confidence_label": "medium",
    }

    if ball_x is not None:
        head_ball_offset = head_x - ball_x
        result["head_ball_offset_px"] = round(head_ball_offset, 1)
        result["head_behind_ball"] = head_ball_offset > 30

    return result


# ────────────────────────────────────────────────────────────────────────
# Balance Direction
# ────────────────────────────────────────────────────────────────────────

def estimate_balance_direction(
    landmarks: Dict[str, Any],
) -> Dict[str, Any]:
    """Estimate which way the batter is leaning at impact.

    From front-on, balance is inferred from the spine angle and
    head position relative to feet.

    Args:
        landmarks: Pose dict with shoulder, hip, and head landmarks.

    Returns:
        dict with keys:
            lean_direction (str): 'forward', 'backward', 'toward_off',
                                  'toward_leg', 'balanced'.
            confidence_label (str).
    """
    nose = landmarks.get("NOSE")
    lhip = landmarks.get("LEFT_HIP")
    rhip = landmarks.get("RIGHT_HIP")
    lshoulder = landmarks.get("LEFT_SHOULDER")
    rshoulder = landmarks.get("RIGHT_SHOULDER")

    if not all([nose, lhip, rhip]):
        return {"lean_direction": "unknown", "confidence_label": "low"}

    # Head y vs hip y (forward lean = head further forward/lower)
    head_y = nose.get("y", 0.5)
    hip_y = (lhip.get("y", 0.5) + rhip.get("y", 0.5)) / 2.0
    head_forward = (hip_y - head_y)  # larger = head is higher = more upright

    # Shoulder tilt (lateral lean)
    if lshoulder and rshoulder:
        shoulder_tilt = lshoulder.get("y", 0.5) - rshoulder.get("y", 0.5)
    else:
        shoulder_tilt = 0

    if abs(head_forward) < 0.05 and abs(shoulder_tilt) < 0.03:
        return {"lean_direction": "balanced", "confidence_label": "medium"}
    elif head_forward > 0.1:
        return {"lean_direction": "forward", "confidence_label": "medium"}
    elif head_forward < -0.05:
        return {"lean_direction": "backward", "confidence_label": "medium"}
    elif shoulder_tilt > 0.03:
        return {"lean_direction": "toward_off", "confidence_label": "low"}
    elif shoulder_tilt < -0.03:
        return {"lean_direction": "toward_leg", "confidence_label": "low"}

    return {"lean_direction": "balanced", "confidence_label": "low"}


# ────────────────────────────────────────────────────────────────────────
# Impact Point on Bat
# ────────────────────────────────────────────────────────────────────────

def estimate_impact_point(
    bat_angle_deg: Optional[float],
    bat_speed_px: float,
    hand_landmarks: Dict[str, Any],
) -> Dict[str, Any]:
    """Estimate where on the bat the ball made contact.

    From front-on:
      - Middle: clean hit, bat speed peaks sharply, balanced hands
      - Toe: ball struck low on bat, bottom hand dominant
      - Edge: ball caught edge, bat wobbles, off-centre hit

    Args:
        bat_angle_deg: Bat angle at impact (from BatAnalyzer).
        bat_speed_px: Bat speed at impact (peak).
        hand_landmarks: Pose dict with wrist landmarks.

    Returns:
        dict with keys:
            impact_point_label (str): 'middle', 'toe', 'edge', 'unknown'.
            confidence_label (str).
    """
    if bat_speed_px < 5:
        return {"impact_point_label": "no_shot", "confidence_label": "high"}

    # This is inherently speculative from front-on.
    # We use a rule of thumb:
    #   - High speed + clean angle → middle
    #   - Low speed + angle change → edge
    #   - Low speed + normal angle → toe

    if bat_angle_deg is None:
        return {"impact_point_label": "unknown", "confidence_label": "low"}

    # Clean hit: bat angle near vertical at impact (0 ± 20°) with good speed
    if abs(bat_angle_deg) < 20 and bat_speed_px > 30:
        return {"impact_point_label": "middle", "confidence_label": "estimate"}

    # Edge: significant angle (bat not straight) or speed drop
    if abs(bat_angle_deg) > 30 and bat_speed_px < 50:
        return {"impact_point_label": "edge", "confidence_label": "estimate"}

    # Toe: straightish bat but low speed
    if bat_speed_px < 30:
        return {"impact_point_label": "toe", "confidence_label": "estimate"}

    return {"impact_point_label": "middle", "confidence_label": "low"}


# ────────────────────────────────────────────────────────────────────────
# Shoulder Alignment
# ────────────────────────────────────────────────────────────────────────

def estimate_shoulder_alignment(
    landmarks: Dict[str, Any],
) -> Dict[str, Any]:
    """Measure shoulder line angle relative to the pitch.

    From front-on, the shoulder line tells us if the batter is:
      - Open (shoulders point toward off) — risk of being bowled through gate
      - Closed (shoulders point toward leg) — risk of LBW
      - Square (shoulders parallel to crease) — ideal

    Args:
        landmarks: Pose dict with shoulder landmarks.

    Returns:
        dict with keys:
            shoulder_angle_deg (float): 0 = square, positive = open,
                                        negative = closed.
            alignment_label (str).
            confidence_label (str).
    """
    ls = landmarks.get("LEFT_SHOULDER")
    rs = landmarks.get("RIGHT_SHOULDER")
    if not ls or not rs:
        return {"shoulder_angle_deg": None, "alignment_label": "unknown",
                "confidence_label": "low"}

    dx = rs.get("x", 0.5) - ls.get("x", 0.5)
    dy = rs.get("y", 0.5) - ls.get("y", 0.5)

    if abs(dx) < 0.001:
        return {"shoulder_angle_deg": 0, "alignment_label": "square",
                "confidence_label": "low"}

    # Angle relative to horizontal (image x-axis)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    # For front-on, 0° = horizontal (parallel to crease)
    # Positive = left shoulder higher (open for right-hander)
    if abs(angle_deg) < 10:
        label = "square"
    elif angle_deg > 0:
        label = "open"
    else:
        label = "closed"

    return {
        "shoulder_angle_deg": round(angle_deg, 1),
        "alignment_label": label,
        "confidence_label": "medium",
    }


# ────────────────────────────────────────────────────────────────────────
# Aggregate all front-on metrics for a frame
# ────────────────────────────────────────────────────────────────────────

def compute_front_on_frame_metrics(
    landmarks: Dict[str, Any],
    frame_width: int,
    frame_height: int,
    batting_hand: str = "right",
    stance_landmarks: Optional[Dict[str, Any]] = None,
    ball_x: Optional[float] = None,
    bat_angle_deg: Optional[float] = None,
    bat_speed_px: float = 0,
) -> Dict[str, Any]:
    """Compute all front-on specific metrics for a single frame.

    Args:
        landmarks: Current frame's pose landmarks.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        batting_hand: 'right' or 'left'.
        stance_landmarks: Landmarks from stance phase (for trigger movement).
        ball_x: Ball x-position at batter's end (optional).
        bat_angle_deg: Bat angle at this frame (from BatAnalyzer).
        bat_speed_px: Bat speed at this frame.

    Returns:
        dict with front_on_* prefixed keys.
    """
    bat_face = estimate_bat_face(landmarks, batting_hand)
    foot_stump = estimate_foot_stump_alignment(landmarks, frame_width)
    head_sync = estimate_head_line_sync(landmarks, ball_x, frame_width)
    balance = estimate_balance_direction(landmarks)
    impact_pt = estimate_impact_point(bat_angle_deg, bat_speed_px, landmarks)
    shoulder = estimate_shoulder_alignment(landmarks)

    result = {
        "front_on_bat_face_angle": bat_face.get("bat_face_angle"),
        "front_on_bat_face_label": bat_face.get("bat_face_label"),
        "front_on_bat_face_confidence": bat_face.get("confidence_label"),

        "front_on_foot_alignment": foot_stump.get("alignment_label"),
        "front_on_foot_offset_px": foot_stump.get("lateral_offset_px"),

        "front_on_head_offset_px": head_sync.get("head_offset_px"),
        "front_on_head_ball_offset_px": head_sync.get("head_ball_offset_px"),
        "front_on_head_behind_ball": head_sync.get("head_behind_ball"),

        "front_on_balance": balance.get("lean_direction"),

        "front_on_impact_point": impact_pt.get("impact_point_label"),

        "front_on_shoulder_angle": shoulder.get("shoulder_angle_deg"),
        "front_on_shoulder_alignment": shoulder.get("alignment_label"),
    }

    # Lateral trigger (requires stance landmarks)
    if stance_landmarks is not None:
        trigger = estimate_lateral_trigger(stance_landmarks, landmarks, frame_width)
        result["front_on_lateral_movement_px"] = trigger.get("lateral_movement_px")
        result["front_on_lateral_direction"] = trigger.get("direction")
        result["front_on_is_across"] = trigger.get("is_across")

    # Depth movement proxy (down the wicket / deep in crease)
    depth = estimate_depth_movement(landmarks)
    result["hip_depth_y"] = depth.get("hip_y")
    result["hip_depth_z"] = depth.get("hip_z")

    return result


# ────────────────────────────────────────────────────────────────────────
# Depth Movement (down the wicket / deep in crease)
# ────────────────────────────────────────────────────────────────────────

def estimate_depth_movement(
    landmarks: Dict[str, Any],
) -> Dict[str, Any]:
    """Estimate batter depth from hip landmark y-position.

    In a front-on view the hip y-coordinate (normalised [0,1]) serves as a
    rough proxy for depth:
      - Higher y (lower in frame) ≈ closer to camera ≈ down the wicket
      - Lower y (higher in frame)  ≈ farther from camera ≈ deep in crease

    The z-coordinate from MediaPipe (negative = toward camera) is stored
    alongside for reference.

    Returns:
        dict with ``hip_y`` (float or None) and ``hip_z`` (float or None).
    """
    lh = landmarks.get("LEFT_HIP")
    rh = landmarks.get("RIGHT_HIP")
    if not lh or not rh:
        return {"hip_y": None, "hip_z": None}

    hip_y = (lh.get("y", 0.5) + rh.get("y", 0.5)) / 2.0
    hip_z = (lh.get("z", 0.0) + rh.get("z", 0.0)) / 2.0

    return {"hip_y": round(hip_y, 4), "hip_z": round(hip_z, 4)}
