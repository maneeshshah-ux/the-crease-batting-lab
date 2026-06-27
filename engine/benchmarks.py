"""
Cricketing Benchmarks — Reference data for player comparisons.

Provides real-world reference points so that raw metrics (pixels, degrees)
translate to "you move your head like Kohli" or "your bat speed is
approaching Rohit Sharma territory."

Sources: Published biomechanics research, coaching manuals, and
publicly available player data from various cricket boards.
"""

# ============================================================
# BAT SPEED (km/h at the toe of the bat)
# Measured at impact. Sources: Cricket Australia biomechanics,
# ECB coaching resources, published sports science papers.
# ============================================================
BAT_SPEED_BENCHMARKS = {
    "club_amateur": {
        "label": "Club / Amateur",
        "avg_kmh": 75,
        "peak_kmh": 95,
        "note": "Typical weekend cricketer. Good technique but lacks explosive power.",
    },
    "club_premier": {
        "label": "Premier / Grade",
        "avg_kmh": 95,
        "peak_kmh": 115,
        "note": "Regular 1st XI player. Decent bat speed through the ball.",
    },
    "state_pro": {
        "label": "State / First-Class",
        "avg_kmh": 110,
        "peak_kmh": 130,
        "note": "Professional. High bat speed generated from core rotation.",
    },
    "international": {
        "label": "International",
        "avg_kmh": 120,
        "peak_kmh": 145,
        "note": "World-class. Elite bat speed through exceptional timing and body mechanics.",
    },
}

# Player-specific bat speed references (peak at impact)
PLAYER_BAT_SPEED = {
    "Rohit Sharma":        {"peak_kmh": 145, "style": "Elegant power — late on the ball, insane wrists"},
    "Virat Kohli":          {"peak_kmh": 135, "style": "Controlled aggression — head still, hands fast"},
    "Steve Smith":         {"peak_kmh": 128, "style": "Unorthodox — all wrists and core, deceptively quick"},
    "David Warner":        {"peak_kmh": 140, "style": "Explosive — uses depth of crease and fast hands"},
    "Joe Root":            {"peak_kmh": 125, "style": "Classical — weighted transfer, smooth through the line"},
    "AB de Villiers":      {"peak_kmh": 150, "style": "Freakish — can change shot in 0.2s, bat speed from anywhere"},
    "Ben Stokes":           {"peak_kmh": 142, "style": "Brute power — strong core, clears front leg"},
}


# ============================================================
# HEAD STABILITY (pixels of head movement per frame)
# The CREASE proprietary metric. Lower = better.
# Based on analysis of hundreds of sessions.
# ============================================================
HEAD_STABILITY_BENCHMARKS = {
    "excellent": {
        "score_range": (80, 100),
        "avg_movement_px": 1.5,
        "note": "Elite. Head virtually still. The ball comes to you.",
    },
    "good": {
        "score_range": (60, 79),
        "avg_movement_px": 3.0,
        "note": "Good club standard. Some movement on aggressive shots.",
    },
    "average": {
        "score_range": (40, 59),
        "avg_movement_px": 5.0,
        "note": "Average. Head moves on back-foot and driving shots.",
    },
    "needs_work": {
        "score_range": (20, 39),
        "avg_movement_px": 8.0,
        "note": "Below average. You're losing sight of the ball through the shot.",
    },
    "poor": {
        "score_range": (0, 19),
        "avg_movement_px": 15.0,
        "note": "Your head is jumping. You'll struggle against pace and movement.",
    },
}

HEAD_STABILITY_PLAYERS = {
    "Virat Kohli":          {"score": 95, "secret": "Watches the ball onto the bat; head stays level through the line."},
    "Kane Williamson":     {"score": 92, "secret": "Exceptionally still head; minimal trigger movement."},
    "Steve Smith":         {"score": 85, "secret": "Despite the fidgeting, head is still at point of contact."},
    "Joe Root":            {"score": 88, "secret": "Classical head position over the ball at all times."},
    "Marnus Labuschagne":  {"score": 78, "secret": "Active trigger but head settles well before impact."},
}


# ============================================================
# KNEE BEND (front knee angle at contact, degrees)
# From cricket biomechanics research.
# 180° = straight leg, 90° = deep knee bend
#
# SIDE-ON ranges (reference standard):
# ============================================================
FRONT_KNEE_BENCHMARKS = {
    "too_straight": {
        "range": (170, 180),
        "note": "Leg is nearly straight. You're tall to the ball — difficult to drive with power.",
    },
    "good_defensive": {
        "range": (145, 169),
        "note": "Good bend for defensive play. Solid base for forward defence.",
    },
    "ideal_drive": {
        "range": (120, 144),
        "note": "Ideal for driving. Low enough to reach the pitch, bent enough to transfer weight.",
    },
    "deep_bend": {
        "range": (90, 119),
        "note": "Deep knee bend — typically for spin or to get very low. Hard to generate power.",
    },
    "extreme": {
        "range": (0, 89),
        "note": "Extreme knee bend. You're almost kneeling. Difficult to recover for the next ball.",
    },
}

# FRONT-ON ranges (bowler's end camera)
# The same physical knee bend projects ~15-20° straighter in front-on 2D:
#   ideal drive (120-144° side-on) → reads as 145-165° front-on
#   good defensive (145-169° side-on) → reads as 160-178° front-on
FRONT_ON_FRONT_KNEE_BENCHMARKS = {
    "too_straight": {
        "range": (176, 180),
        "note": "Leg appears nearly straight from front. Limited power generation through the shot.",
    },
    "good_defensive": {
        "range": (160, 175),
        "note": "Good knee bend visible from front. Solid base.",
    },
    "ideal_drive": {
        "range": (145, 159),
        "note": "Ideal knee bend for driving. Low to the ball without over-bending.",
    },
    "deep_bend": {
        "range": (125, 144),
        "note": "Deep knee bend visible from front. Typically for spin bowling.",
    },
    "extreme": {
        "range": (0, 124),
        "note": "Extreme knee bend — very low stance. May compromise balance against pace.",
    },
}


# ============================================================
# SPINE ANGLE AT CONTACT (degrees from vertical)
# 0° = perfectly upright, 90° = parallel to ground
#
# SIDE-ON ranges (reference standard):
# ============================================================
SPINE_ANGLE_BENCHMARKS = {
    "upright": {
        "range": (0, 10),
        "note": "Very upright. Good for back-foot play but hard to reach full-length balls.",
    },
    "balanced": {
        "range": (11, 22),
        "note": "Ideal. Head over the ball, weight forward, eyes level.",
    },
    "lunging": {
        "range": (23, 35),
        "note": "Leaning forward. You're reaching for the ball — balance compromised.",
    },
    "falling": {
        "range": (36, 90),
        "note": "Head is past the front knee. You're falling into the shot. Vulnerable to movement.",
    },
}

# FRONT-ON spine angle ranges
# Forward lean appears smaller from front-on view
FRONT_ON_SPINE_ANGLE_BENCHMARKS = {
    "upright": {
        "range": (0, 5),
        "note": "Very upright from front. Good for back-foot but hard to reach length balls.",
    },
    "balanced": {
        "range": (6, 16),
        "note": "Ideal forward lean visible from front. Head position looks solid.",
    },
    "lunging": {
        "range": (17, 28),
        "note": "Leaning forward from front view. Balance may be compromised.",
    },
    "falling": {
        "range": (29, 90),
        "note": "Significant forward lean from front. Falling into the shot — vulnerable to movement.",
    },
}


# View-aware lookup
# Angled (~30°) uses intermediate ranges between side and front-on
ANGLED_FRONT_KNEE_BENCHMARKS = {
    "too_straight": {
        "range": (173, 180),
        "note": "Leg appears fairly straight from this angle. Bend the knee more for power.",
    },
    "good_defensive": {
        "range": (152, 172),
        "note": "Good knee bend visible at this angle. Solid base for defensive play.",
    },
    "ideal_drive": {
        "range": (135, 151),
        "note": "Ideal knee bend for driving. Good weight transfer position.",
    },
    "deep_bend": {
        "range": (115, 134),
        "note": "Deep knee bend from this angle. Good for spin, may struggle against pace.",
    },
    "extreme": {
        "range": (0, 114),
        "note": "Extreme knee bend. Very low stance.",
    },
}

ANGLED_SPINE_ANGLE_BENCHMARKS = {
    "upright": {
        "range": (0, 7),
        "note": "Quite upright from this angle. Good for back-foot but reaching forward may be hard.",
    },
    "balanced": {
        "range": (8, 20),
        "note": "Ideal forward lean. Head position looks solid from this angle.",
    },
    "lunging": {
        "range": (21, 32),
        "note": "Leaning forward. Balance may be compromised on driving shots.",
    },
    "falling": {
        "range": (33, 90),
        "note": "Significant forward lean. Falling into the shot.",
    },
}


VIEW_KNEE_BENCHMARKS = {
    "side_off": FRONT_KNEE_BENCHMARKS,
    "side_leg": FRONT_KNEE_BENCHMARKS,
    "front_on": FRONT_ON_FRONT_KNEE_BENCHMARKS,
    "angled": ANGLED_FRONT_KNEE_BENCHMARKS,
    "behind": FRONT_ON_FRONT_KNEE_BENCHMARKS,
}

VIEW_SPINE_BENCHMARKS = {
    "side_off": SPINE_ANGLE_BENCHMARKS,
    "side_leg": SPINE_ANGLE_BENCHMARKS,
    "front_on": FRONT_ON_SPINE_ANGLE_BENCHMARKS,
    "angled": ANGLED_SPINE_ANGLE_BENCHMARKS,
    "behind": FRONT_ON_SPINE_ANGLE_BENCHMARKS,
}


def get_knee_assessment(angle, camera_view="side_off"):
    """
    Assess front knee bend angle, view-aware.

    Args:
        angle: measured knee angle in degrees
        camera_view: "side_off", "side_leg", "front_on", or "behind"
    """
    benchmarks = VIEW_KNEE_BENCHMARKS.get(camera_view, FRONT_KNEE_BENCHMARKS)
    for level, data in benchmarks.items():
        lo, hi = data["range"]
        if lo <= angle <= hi:
            return {
                "level": level,
                "note": data["note"],
                "camera_view": camera_view,
            }
    # Fallback — find nearest
    if angle < benchmarks["extreme"]["range"][0]:
        return {"level": "extreme", "note": benchmarks["extreme"]["note"], "camera_view": camera_view}
    return {"level": "too_straight", "note": benchmarks["too_straight"]["note"], "camera_view": camera_view}


def get_spine_assessment(angle, camera_view="side_off"):
    """
    Assess spine lean angle, view-aware.
    """
    benchmarks = VIEW_SPINE_BENCHMARKS.get(camera_view, SPINE_ANGLE_BENCHMARKS)
    for level, data in benchmarks.items():
        lo, hi = data["range"]
        if lo <= angle <= hi:
            return {
                "level": level,
                "note": data["note"],
                "camera_view": camera_view,
            }
    return {"level": "balanced", "note": benchmarks["balanced"]["note"], "camera_view": camera_view}


def get_bat_speed_benchmark(avg_kmh, peak_kmh):
    """
    Compare a player's bat speed to benchmarks.
    Returns dict with level, comparison text, and nearby player.
    """
    # Determine level
    if peak_kmh >= 130:
        level = "international"
    elif peak_kmh >= 110:
        level = "state_pro"
    elif peak_kmh >= 90:
        level = "club_premier"
    else:
        level = "club_amateur"

    bench = BAT_SPEED_BENCHMARKS[level]

    # Find nearest player
    nearest_player = None
    smallest_diff = float('inf')
    for name, data in PLAYER_BAT_SPEED.items():
        diff = abs(data["peak_kmh"] - peak_kmh)
        if diff < smallest_diff:
            smallest_diff = diff
            nearest_player = (name, data)

    return {
        "level": level,
        "level_label": bench["label"],
        "level_note": bench["note"],
        "avg_vs_benchmark": avg_kmh - bench["avg_kmh"],
        "peak_vs_benchmark": peak_kmh - bench["peak_kmh"],
        "nearest_player": nearest_player,
        "comparison_text": (
            f"Your peak bat speed ({peak_kmh:.0f} km/h) is at {bench['label'].lower()} level. "
            f"{'Above' if peak_kmh > bench['peak_kmh'] else 'Below'} the typical {bench['peak_kmh']:.0f} km/h peak for this level."
        ),
    }


def get_head_stability_assessment(score):
    """
    Assess head stability score and return comparison.
    """
    if score >= 80:
        level = "excellent"
    elif score >= 60:
        level = "good"
    elif score >= 40:
        level = "average"
    elif score >= 20:
        level = "needs_work"
    else:
        level = "poor"

    bench = HEAD_STABILITY_BENCHMARKS[level]

    # Find player comparison
    nearest_player = None
    smallest_diff = float('inf')
    for name, data in HEAD_STABILITY_PLAYERS.items():
        diff = abs(data["score"] - score)
        if diff < smallest_diff:
            smallest_diff = diff
            nearest_player = (name, data)

    return {
        "level": level,
        "level_note": bench["note"],
        "avg_movement_px": bench["avg_movement_px"],
        "nearest_player": nearest_player,
    }



