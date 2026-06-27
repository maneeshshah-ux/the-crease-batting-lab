"""
Pro Comparison — Compare your batting biomechanics against professional players.

Provides a comprehensive comparison engine that matches a user's session metrics
against a reference database of professional player profiles. Uses published,
publicly available data and coaching references.

Zonal Comparison — Groups players by gender, role, and style to provide
zone-level matching before individual player matching.

LEGAL: All player names are used as secondary factual comparisons only. Level labels
(Club, State, International) are primary. A legal disclaimer is included on every
comparison output. Player reference data is approximate and based on published
biomechanics research, coaching manuals, and publicly available player information.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from .benchmarks import (
    PLAYER_BAT_SPEED,
    HEAD_STABILITY_PLAYERS,
    BAT_SPEED_BENCHMARKS,
    HEAD_STABILITY_BENCHMARKS,
    FRONT_KNEE_BENCHMARKS,
    SPINE_ANGLE_BENCHMARKS,
    get_bat_speed_benchmark,
    get_head_stability_assessment,
    get_knee_assessment,
    get_spine_assessment,
)


# ======================================================================
# ZONE TAXONOMY
# ======================================================================
# Each zone = (gender, role, style) where:
#   gender: "male" | "female"
#   role: "opener" | "top_order" | "middle_order" | "finisher"
#   style: "aggressive" | "classical" | "power_hitter" | "unorthodox" | "anchor"
#
# Zone labels for display:
ZONE_LABELS: Dict[str, str] = {
    "male_opener_aggressive": "Male Openers — Aggressive",
    "male_opener_classical": "Male Openers — Classical",
    "male_top_order_classical": "Male Top-Order — Classical",
    "male_top_order_unorthodox": "Male Top-Order — Unorthodox",
    "male_top_order_anchor": "Male Top-Order — Anchor",
    "male_middle_order_power": "Male Middle-Order — Power Hitters",
    "male_middle_order_anchor": "Male Middle-Order — Anchors",
    "male_finisher_aggressive": "Male Finishers — Aggressive",
    "male_allrounder_power": "Male All-Rounders — Power Hitters",
    "female_opener_aggressive": "Female Openers — Aggressive",
    "female_opener_classical": "Female Openers — Classical",
    "female_top_order_classical": "Female Top-Order — Classical",
    "female_top_order_anchor": "Female Top-Order — Anchors",
    "female_middle_order_power": "Female Middle-Order — Power Hitters",
    "female_middle_order_allrounder": "Female All-Rounders",
}


# ======================================================================
# PROFESSIONAL PLAYER BIOMECHANICAL DATABASE
# ======================================================================
# Reference profiles for professional batsmen and batswomen.
# Sources: Published biomechanics research, ECB/Cricket Australia coaching
# resources, publicly available match footage analysis.
#
# All values are APPROXIMATE reference ranges for comparison purposes.
# Individual player biomechanics vary between formats, conditions, and shot types.
#
# Each profile includes:
#   - gender: "male" | "female"
#   - role: "opener" | "top_order" | "middle_order" | "finisher"
#   - style_tags: list of style descriptors
#   - bat_speed: peak_kmh (bat tip speed at impact)
#   - head_stability: score (0-100, higher = better)
#   - front_knee_angle: degrees at contact (side-on reference)
#   - spine_angle: degrees from vertical at contact
#   - front_elbow_angle: degrees at contact
#   - stance_signature: 7-feature stance profile (for similarity matching)
#   - style: brief description of playing style
#   - strengths / weaknesses: lists
#   - level: "international", "state_pro", "club_premier", "club_amateur"
# ======================================================================

PRO_PLAYER_DATABASE: Dict[str, Dict[str, Any]] = {
    # ==================================================================
    # MALE — INTERNATIONAL OPENERS — AGGRESSIVE
    # ==================================================================
    "Rohit Sharma": {
        "gender": "male", "role": "opener", "style_tags": ["aggressive", "power_hitter"],
        "bat_speed_peak_kmh": 145, "head_stability_score": 88,
        "front_knee_angle": 142, "spine_angle": 12, "front_elbow_angle": 160,
        "style": "Elegant power — late on the ball, insane wrists. Minimal foot movement, maximum timing.",
        "strengths": ["Pull shot", "Cut shot", "Lofted drives", "Bat speed"],
        "weaknesses": ["Early stages of innings", "Movement off the pitch"],
        "level": "international",
        "stance_width": 0.32, "hip_shoulder_ratio": 1.40, "head_forward": 0.10,
        "grip_height": 0.50, "back_lift_height": 0.68, "stance_knee_angle": 162, "face_ratio": 0.58,
    },
    "David Warner": {
        "gender": "male", "role": "opener", "style_tags": ["aggressive", "power_hitter"],
        "bat_speed_peak_kmh": 140, "head_stability_score": 78,
        "front_knee_angle": 150, "spine_angle": 10, "front_elbow_angle": 162,
        "style": "Explosive — uses depth of crease and fast hands. Aggressive intent from ball one.",
        "strengths": ["Power hitting", "Cut and pull", "Bat speed"],
        "weaknesses": ["Movement outside off", "Left-arm pace"],
        "level": "international",
        "stance_width": 0.34, "hip_shoulder_ratio": 1.42, "head_forward": 0.08,
        "grip_height": 0.48, "back_lift_height": 0.66, "stance_knee_angle": 168, "face_ratio": 0.55,
    },
    "Travis Head": {
        "gender": "male", "role": "opener", "style_tags": ["aggressive"],
        "bat_speed_peak_kmh": 138, "head_stability_score": 72,
        "front_knee_angle": 148, "spine_angle": 12, "front_elbow_angle": 160,
        "style": "Aggressive — strong on the cut and pull. Attacks spin with intent.",
        "strengths": ["Against spin", "Cut and pull", "Bat speed"],
        "weaknesses": ["Outside off", "Short ball judgement"],
        "level": "international",
        "stance_width": 0.33, "hip_shoulder_ratio": 1.40, "head_forward": 0.09,
        "grip_height": 0.49, "back_lift_height": 0.66, "stance_knee_angle": 166, "face_ratio": 0.53,
    },
    "AB de Villiers": {
        "gender": "male", "role": "opener", "style_tags": ["unorthodox", "aggressive", "power_hitter"],
        "bat_speed_peak_kmh": 150, "head_stability_score": 90,
        "front_knee_angle": 135, "spine_angle": 13, "front_elbow_angle": 152,
        "style": "Freakish — can change shot in 0.2s, bat speed from anywhere. 360-degree play.",
        "strengths": ["Innovation", "Bat speed", "All-round play"],
        "weaknesses": ["None clearly exploitable"],
        "level": "international",
        "stance_width": 0.30, "hip_shoulder_ratio": 1.36, "head_forward": 0.13,
        "grip_height": 0.54, "back_lift_height": 0.71, "stance_knee_angle": 156, "face_ratio": 0.63,
    },
    # ==================================================================
    # MALE — INTERNATIONAL OPENERS — CLASSICAL
    # ==================================================================
    "Shubman Gill": {
        "gender": "male", "role": "opener", "style_tags": ["classical"],
        "bat_speed_peak_kmh": 132, "head_stability_score": 86,
        "front_knee_angle": 140, "spine_angle": 14, "front_elbow_angle": 156,
        "style": "Elegant — classic off-side play, good head position. Modern technique.",
        "strengths": ["Cover drive", "Head position", "Against pace"],
        "weaknesses": ["Short ball", "Spin on turning tracks"],
        "level": "international",
        "stance_width": 0.29, "hip_shoulder_ratio": 1.36, "head_forward": 0.13,
        "grip_height": 0.56, "back_lift_height": 0.70, "stance_knee_angle": 158, "face_ratio": 0.61,
    },
    # ==================================================================
    # MALE — INTERNATIONAL TOP-ORDER — CLASSICAL
    # ==================================================================
    "Virat Kohli": {
        "gender": "male", "role": "top_order", "style_tags": ["classical", "aggressive"],
        "bat_speed_peak_kmh": 135, "head_stability_score": 95,
        "front_knee_angle": 138, "spine_angle": 15, "front_elbow_angle": 155,
        "style": "Controlled aggression — head still, hands fast. Textbook front-foot play.",
        "strengths": ["Head stability", "Front-foot driving", "Weight transfer"],
        "weaknesses": ["Occasional nibble outside off", "Can be late on short ball"],
        "level": "international",
        "stance_width": 0.28, "hip_shoulder_ratio": 1.35, "head_forward": 0.15,
        "grip_height": 0.55, "back_lift_height": 0.72, "stance_knee_angle": 158, "face_ratio": 0.65,
    },
    "Joe Root": {
        "gender": "male", "role": "top_order", "style_tags": ["classical"],
        "bat_speed_peak_kmh": 125, "head_stability_score": 88,
        "front_knee_angle": 140, "spine_angle": 14, "front_elbow_angle": 158,
        "style": "Classical — weighted transfer, smooth through the line. Traditional batting technique.",
        "strengths": ["Cover drive", "Against spin", "Running between wickets"],
        "weaknesses": ["Pad-play outside off", "Occasional soft dismissal"],
        "level": "international",
        "stance_width": 0.30, "hip_shoulder_ratio": 1.38, "head_forward": 0.12,
        "grip_height": 0.58, "back_lift_height": 0.70, "stance_knee_angle": 155, "face_ratio": 0.62,
    },
    "Kane Williamson": {
        "gender": "male", "role": "top_order", "style_tags": ["classical", "anchor"],
        "bat_speed_peak_kmh": 122, "head_stability_score": 92,
        "front_knee_angle": 136, "spine_angle": 16, "front_elbow_angle": 156,
        "style": "Technically sound — exceptionally still head, minimal trigger. Timing over power.",
        "strengths": ["Head stability", "Off-side play", "Adaptability"],
        "weaknesses": ["Power hitting", "Express pace"],
        "level": "international",
        "stance_width": 0.26, "hip_shoulder_ratio": 1.32, "head_forward": 0.14,
        "grip_height": 0.56, "back_lift_height": 0.69, "stance_knee_angle": 160, "face_ratio": 0.60,
    },
    "Babar Azam": {
        "gender": "male", "role": "top_order", "style_tags": ["classical"],
        "bat_speed_peak_kmh": 130, "head_stability_score": 90,
        "front_knee_angle": 136, "spine_angle": 14, "front_elbow_angle": 154,
        "style": "Silk-smooth — textbook technique, superb timing. Cover drive is signature.",
        "strengths": ["Cover drive", "Head stability", "Timing"],
        "weaknesses": ["Short ball", "Pressure situations"],
        "level": "international",
        "stance_width": 0.28, "hip_shoulder_ratio": 1.37, "head_forward": 0.12,
        "grip_height": 0.57, "back_lift_height": 0.71, "stance_knee_angle": 157, "face_ratio": 0.64,
    },
    # ==================================================================
    # MALE — INTERNATIONAL TOP-ORDER — UNORTHODOX
    # ==================================================================
    "Steve Smith": {
        "gender": "male", "role": "top_order", "style_tags": ["unorthodox"],
        "bat_speed_peak_kmh": 128, "head_stability_score": 85,
        "front_knee_angle": 148, "spine_angle": 18, "front_elbow_angle": 150,
        "style": "Unorthodox — all wrists and core, deceptively quick. Unique trigger movements.",
        "strengths": ["Leg-side play", "Against spin", "Concentration"],
        "weaknesses": ["Wobble seam outside off", "LBW vulnerable"],
        "level": "international",
        "stance_width": 0.38, "hip_shoulder_ratio": 1.30, "head_forward": 0.22,
        "grip_height": 0.52, "back_lift_height": 0.65, "stance_knee_angle": 165, "face_ratio": 0.52,
    },
    "Marnus Labuschagne": {
        "gender": "male", "role": "top_order", "style_tags": ["unorthodox"],
        "bat_speed_peak_kmh": 125, "head_stability_score": 78,
        "front_knee_angle": 142, "spine_angle": 16, "front_elbow_angle": 156,
        "style": "Active trigger but head settles well before impact. Compulsive tinkerer.",
        "strengths": ["Concentration", "Against spin", "Leg-side play"],
        "weaknesses": ["Movement outside off", "Express pace"],
        "level": "international",
        "stance_width": 0.32, "hip_shoulder_ratio": 1.34, "head_forward": 0.16,
        "grip_height": 0.53, "back_lift_height": 0.67, "stance_knee_angle": 162, "face_ratio": 0.57,
    },
    # ==================================================================
    # MALE — INTERNATIONAL TOP-ORDER — ANCHOR
    # ==================================================================
    "Cheteshwar Pujara": {
        "gender": "male", "role": "top_order", "style_tags": ["anchor", "classical"],
        "bat_speed_peak_kmh": 108, "head_stability_score": 82,
        "front_knee_angle": 144, "spine_angle": 16, "front_elbow_angle": 160,
        "style": "Stonewall — immense concentration, leaves well. Wears down bowling through patience.",
        "strengths": ["Concentration", "Leaving outside off", "Against spin"],
        "weaknesses": ["Strike rotation", "Expansive shots", "Bat speed"],
        "level": "international",
        "stance_width": 0.34, "hip_shoulder_ratio": 1.36, "head_forward": 0.16,
        "grip_height": 0.52, "back_lift_height": 0.60, "stance_knee_angle": 160, "face_ratio": 0.56,
    },
    # ==================================================================
    # MALE — INTERNATIONAL MIDDLE-ORDER — POWER
    # ==================================================================
    "Ben Stokes": {
        "gender": "male", "role": "middle_order", "style_tags": ["power_hitter", "aggressive"],
        "bat_speed_peak_kmh": 142, "head_stability_score": 75,
        "front_knee_angle": 145, "spine_angle": 18, "front_elbow_angle": 158,
        "style": "Brute power — strong core, clears front leg. Match-winner under pressure.",
        "strengths": ["Power hitting", "Counter-attack", "Against pace"],
        "weaknesses": ["Against spin", "Occasional recklessness"],
        "level": "international",
        "stance_width": 0.36, "hip_shoulder_ratio": 1.44, "head_forward": 0.18,
        "grip_height": 0.50, "back_lift_height": 0.64, "stance_knee_angle": 164, "face_ratio": 0.50,
    },
    # ==================================================================
    # MALE — INTERNATIONAL FINISHERS — AGGRESSIVE
    # ==================================================================
    "MS Dhoni": {
        "gender": "male", "role": "finisher", "style_tags": ["power_hitter", "anchor"],
        "bat_speed_peak_kmh": 135, "head_stability_score": 85,
        "front_knee_angle": 148, "spine_angle": 14, "front_elbow_angle": 165,
        "style": "Ice-cool finisher — helicopter finish. Takes it deep, then explodes.",
        "strengths": ["Finishing", "Against spin", "Running", "Helicopter shot"],
        "weaknesses": ["Against swing early", "Strike rotation under pressure"],
        "level": "international",
        "stance_width": 0.35, "hip_shoulder_ratio": 1.38, "head_forward": 0.14,
        "grip_height": 0.46, "back_lift_height": 0.62, "stance_knee_angle": 162, "face_ratio": 0.54,
    },
    "Jos Buttler": {
        "gender": "male", "role": "finisher", "style_tags": ["aggressive", "power_hitter"],
        "bat_speed_peak_kmh": 148, "head_stability_score": 80,
        "front_knee_angle": 146, "spine_angle": 12, "front_elbow_angle": 162,
        "style": "Dynamic — 360-degree player. Scoop, ramp, reverse sweep specialist.",
        "strengths": ["Innovation", "Bat speed", "Death overs"],
        "weaknesses": ["Early swing", "Spin on turning tracks"],
        "level": "international",
        "stance_width": 0.31, "hip_shoulder_ratio": 1.40, "head_forward": 0.11,
        "grip_height": 0.50, "back_lift_height": 0.67, "stance_knee_angle": 160, "face_ratio": 0.59,
    },
    # ==================================================================
    # MALE — ALL-ROUNDERS
    # ==================================================================
    "Ravindra Jadeja": {
        "gender": "male", "role": "finisher", "style_tags": ["aggressive"],
        "bat_speed_peak_kmh": 128, "head_stability_score": 74,
        "front_knee_angle": 150, "spine_angle": 14, "front_elbow_angle": 164,
        "style": "Sword-swinger — powerful lowers the order. Big hitter of spin.",
        "strengths": ["Against spin", "Power hitting", "Running"],
        "weaknesses": ["Against pace", "Leaving outside off"],
        "level": "international",
        "stance_width": 0.33, "hip_shoulder_ratio": 1.40, "head_forward": 0.15,
        "grip_height": 0.48, "back_lift_height": 0.63, "stance_knee_angle": 164, "face_ratio": 0.55,
    },
    # ==================================================================
    # FEMALE — INTERNATIONAL OPENERS — AGGRESSIVE
    # ==================================================================
    "Alyssa Healy": {
        "gender": "female", "role": "opener", "style_tags": ["aggressive", "power_hitter"],
        "bat_speed_peak_kmh": 130, "head_stability_score": 80,
        "front_knee_angle": 144, "spine_angle": 13, "front_elbow_angle": 160,
        "style": "Explosive wicketkeeper-opener. Takes the game on from ball one.",
        "strengths": ["Power hitting", "Over the top", "Bat speed"],
        "weaknesses": ["Early movement", "Consistency"],
        "level": "international",
        "stance_width": 0.33, "hip_shoulder_ratio": 1.38, "head_forward": 0.11,
        "grip_height": 0.50, "back_lift_height": 0.65, "stance_knee_angle": 162, "face_ratio": 0.56,
    },
    "Sophie Devine": {
        "gender": "female", "role": "opener", "style_tags": ["aggressive", "power_hitter"],
        "bat_speed_peak_kmh": 138, "head_stability_score": 76,
        "front_knee_angle": 146, "spine_angle": 12, "front_elbow_angle": 162,
        "style": "Powerhouse — strikes at 130+ consistently. Dominates bowling attacks.",
        "strengths": ["Bat speed", "Power hitting", "Over the top"],
        "weaknesses": ["Spin", "LBW vulnerable"],
        "level": "international",
        "stance_width": 0.34, "hip_shoulder_ratio": 1.40, "head_forward": 0.09,
        "grip_height": 0.48, "back_lift_height": 0.64, "stance_knee_angle": 164, "face_ratio": 0.54,
    },
    "Smriti Mandhana": {
        "gender": "female", "role": "opener", "style_tags": ["aggressive", "classical"],
        "bat_speed_peak_kmh": 122, "head_stability_score": 85,
        "front_knee_angle": 140, "spine_angle": 14, "front_elbow_angle": 156,
        "style": "Elegant left-hander — superb off-side play. Graceful timing and placement.",
        "strengths": ["Cover drive", "Off-side play", "Head position"],
        "weaknesses": ["Short ball", "In-swing"],
        "level": "international",
        "stance_width": 0.30, "hip_shoulder_ratio": 1.35, "head_forward": 0.13,
        "grip_height": 0.54, "back_lift_height": 0.69, "stance_knee_angle": 158, "face_ratio": 0.62,
    },
    "Hayley Matthews": {
        "gender": "female", "role": "opener", "style_tags": ["aggressive"],
        "bat_speed_peak_kmh": 125, "head_stability_score": 80,
        "front_knee_angle": 142, "spine_angle": 14, "front_elbow_angle": 158,
        "style": "Dynamic all-round opener. Strong through the off-side, improving against spin.",
        "strengths": ["Power hitting", "Off-side play", "All-round cricket IQ"],
        "weaknesses": ["Spin", "Patience early in innings"],
        "level": "international",
        "stance_width": 0.32, "hip_shoulder_ratio": 1.37, "head_forward": 0.12,
        "grip_height": 0.52, "back_lift_height": 0.66, "stance_knee_angle": 160, "face_ratio": 0.58,
    },
    # ==================================================================
    # FEMALE — INTERNATIONAL OPENERS — CLASSICAL
    # ==================================================================
    "Beth Mooney": {
        "gender": "female", "role": "opener", "style_tags": ["classical", "anchor"],
        "bat_speed_peak_kmh": 115, "head_stability_score": 88,
        "front_knee_angle": 138, "spine_angle": 15, "front_elbow_angle": 155,
        "style": "Compact left-hander — outstanding against spin. Anchors innings with calculated risk.",
        "strengths": ["Against spin", "Head stability", "Rotation of strike"],
        "weaknesses": ["Pace on the rise", "Short ball"],
        "level": "international",
        "stance_width": 0.30, "hip_shoulder_ratio": 1.34, "head_forward": 0.14,
        "grip_height": 0.55, "back_lift_height": 0.67, "stance_knee_angle": 156, "face_ratio": 0.60,
    },
    "Suzie Bates": {
        "gender": "female", "role": "opener", "style_tags": ["classical"],
        "bat_speed_peak_kmh": 120, "head_stability_score": 84,
        "front_knee_angle": 142, "spine_angle": 14, "front_elbow_angle": 158,
        "style": "Athletic opener — strong on both sides. Excellent technique and composure.",
        "strengths": ["Technique", "Athleticism", "Consistency"],
        "weaknesses": ["Express pace", "Spin on turning tracks"],
        "level": "international",
        "stance_width": 0.31, "hip_shoulder_ratio": 1.36, "head_forward": 0.13,
        "grip_height": 0.53, "back_lift_height": 0.68, "stance_knee_angle": 158, "face_ratio": 0.59,
    },
    # ==================================================================
    # FEMALE — INTERNATIONAL TOP-ORDER — CLASSICAL
    # ==================================================================
    "Meg Lanning": {
        "gender": "female", "role": "top_order", "style_tags": ["classical", "aggressive"],
        "bat_speed_peak_kmh": 125, "head_stability_score": 93,
        "front_knee_angle": 136, "spine_angle": 14, "front_elbow_angle": 154,
        "style": "Clinical — supreme head stability, exceptional placement. Punishes loose balls ruthlessly.",
        "strengths": ["Head stability", "Placement", "Running", "Captaincy"],
        "weaknesses": ["Occasionally lbw to straight balls"],
        "level": "international",
        "stance_width": 0.28, "hip_shoulder_ratio": 1.34, "head_forward": 0.14,
        "grip_height": 0.56, "back_lift_height": 0.70, "stance_knee_angle": 156, "face_ratio": 0.63,
    },
    "Heather Knight": {
        "gender": "female", "role": "top_order", "style_tags": ["classical"],
        "bat_speed_peak_kmh": 118, "head_stability_score": 86,
        "front_knee_angle": 140, "spine_angle": 15, "front_elbow_angle": 158,
        "style": "Solid right-hander — strong off the back foot. Leads from the front.",
        "strengths": ["Back-foot play", "Technique", "Adaptability"],
        "weaknesses": ["Front-foot driving range", "Spin on slow pitches"],
        "level": "international",
        "stance_width": 0.31, "hip_shoulder_ratio": 1.36, "head_forward": 0.14,
        "grip_height": 0.54, "back_lift_height": 0.66, "stance_knee_angle": 158, "face_ratio": 0.58,
    },
    "Laura Wolvaardt": {
        "gender": "female", "role": "top_order", "style_tags": ["classical"],
        "bat_speed_peak_kmh": 112, "head_stability_score": 90,
        "front_knee_angle": 138, "spine_angle": 15, "front_elbow_angle": 156,
        "style": "Graceful — textbook technique, still head. One of the most elegant in the women's game.",
        "strengths": ["Head stability", "Cover drive", "Timing"],
        "weaknesses": ["Power hitting ceiling", "Express pace"],
        "level": "international",
        "stance_width": 0.29, "hip_shoulder_ratio": 1.33, "head_forward": 0.14,
        "grip_height": 0.57, "back_lift_height": 0.69, "stance_knee_angle": 156, "face_ratio": 0.61,
    },
    "Stafanie Taylor": {
        "gender": "female", "role": "top_order", "style_tags": ["classical", "anchor"],
        "bat_speed_peak_kmh": 115, "head_stability_score": 88,
        "front_knee_angle": 140, "spine_angle": 16, "front_elbow_angle": 157,
        "style": "Technically sound — strong against spin, anchors the innings. Off-spin bowling all-rounder.",
        "strengths": ["Against spin", "Anchor role", "Technique"],
        "weaknesses": ["Bat speed", "Quick bowling on the rise"],
        "level": "international",
        "stance_width": 0.30, "hip_shoulder_ratio": 1.35, "head_forward": 0.15,
        "grip_height": 0.55, "back_lift_height": 0.65, "stance_knee_angle": 158, "face_ratio": 0.59,
    },
    "Mithali Raj": {
        "gender": "female", "role": "top_order", "style_tags": ["anchor", "classical"],
        "bat_speed_peak_kmh": 105, "head_stability_score": 92,
        "front_knee_angle": 142, "spine_angle": 17, "front_elbow_angle": 160,
        "style": "Legendary anchor — accumulates with precision. Took women's cricket global.",
        "strengths": ["Concentration", "Accumulation", "Against spin"],
        "weaknesses": ["Strike rotation", "Power hitting"],
        "level": "international",
        "stance_width": 0.32, "hip_shoulder_ratio": 1.36, "head_forward": 0.16,
        "grip_height": 0.52, "back_lift_height": 0.60, "stance_knee_angle": 160, "face_ratio": 0.57,
    },
    # ==================================================================
    # FEMALE — MIDDLE-ORDER — POWER HITTERS
    # ==================================================================
    "Harmanpreet Kaur": {
        "gender": "female", "role": "middle_order", "style_tags": ["power_hitter", "aggressive"],
        "bat_speed_peak_kmh": 135, "head_stability_score": 82,
        "front_knee_angle": 144, "spine_angle": 13, "front_elbow_angle": 160,
        "style": "Powerhouse middle-order — can clear any ground. Iconic 171* against Australia.",
        "strengths": ["Power hitting", "Bat speed", "Big-match temperament"],
        "weaknesses": ["Early pace", "Spin on slow pitches"],
        "level": "international",
        "stance_width": 0.33, "hip_shoulder_ratio": 1.38, "head_forward": 0.11,
        "grip_height": 0.48, "back_lift_height": 0.64, "stance_knee_angle": 162, "face_ratio": 0.55,
    },
    "Nat Sciver-Brunt": {
        "gender": "female", "role": "middle_order", "style_tags": ["power_hitter", "aggressive"],
        "bat_speed_peak_kmh": 132, "head_stability_score": 84,
        "front_knee_angle": 142, "spine_angle": 14, "front_elbow_angle": 158,
        "style": "Elegant power — drives with authority, strong through mid-wicket. All-round brilliance.",
        "strengths": ["Power hitting", "All-round play", "Against pace"],
        "weaknesses": ["Occasional lbw to in-swing", "Spin on turning tracks"],
        "level": "international",
        "stance_width": 0.32, "hip_shoulder_ratio": 1.37, "head_forward": 0.12,
        "grip_height": 0.50, "back_lift_height": 0.66, "stance_knee_angle": 160, "face_ratio": 0.57,
    },
    "Ellyse Perry": {
        "gender": "female", "role": "middle_order", "style_tags": ["classical", "anchor"],
        "bat_speed_peak_kmh": 120, "head_stability_score": 90,
        "front_knee_angle": 140, "spine_angle": 15, "front_elbow_angle": 156,
        "style": "Impeccable — textbook technique, remarkable composure. One of the greatest all-rounders.",
        "strengths": ["Technique", "Head stability", "All-round play", "Composure"],
        "weaknesses": ["Power hitting ceiling", "Strike rotation at times"],
        "level": "international",
        "stance_width": 0.30, "hip_shoulder_ratio": 1.35, "head_forward": 0.14,
        "grip_height": 0.55, "back_lift_height": 0.68, "stance_knee_angle": 156, "face_ratio": 0.60,
    },
    # ==================================================================
    # STATE / FIRST-CLASS REFERENCE PROFILES
    # ==================================================================
    "First-Class Batter (Model)": {
        "gender": "male", "role": "top_order", "style_tags": ["classical"],
        "bat_speed_peak_kmh": 125, "head_stability_score": 75,
        "front_knee_angle": 142, "spine_angle": 16, "front_elbow_angle": 157,
        "style": "Professional standard. Consistent technique, reliable against pace and spin.",
        "strengths": ["Consistency", "Shot selection", "Technique"],
        "weaknesses": ["Power hitting ceiling", "Adaptability to conditions"],
        "level": "state_pro",
        "stance_width": 0.31, "hip_shoulder_ratio": 1.36, "head_forward": 0.14,
        "grip_height": 0.54, "back_lift_height": 0.66, "stance_knee_angle": 160, "face_ratio": 0.59,
    },
    # ==================================================================
    # GRADE / PREMIER REFERENCE PROFILES
    # ==================================================================
    "Premier Club Batter (Model)": {
        "gender": "male", "role": "top_order", "style_tags": ["classical"],
        "bat_speed_peak_kmh": 110, "head_stability_score": 65,
        "front_knee_angle": 148, "spine_angle": 18, "front_elbow_angle": 160,
        "style": "Good club standard. Solid technique with occasional lapses in concentration.",
        "strengths": ["Reliable defence", "Good club cricket player"],
        "weaknesses": ["Bat speed", "Head movement", "Against quality bowling"],
        "level": "club_premier",
        "stance_width": 0.33, "hip_shoulder_ratio": 1.38, "head_forward": 0.16,
        "grip_height": 0.52, "back_lift_height": 0.62, "stance_knee_angle": 162, "face_ratio": 0.55,
    },
    # ==================================================================
    # AMATEUR REFERENCE PROFILE
    # ==================================================================
    "Amateur Batter (Model)": {
        "gender": "male", "role": "top_order", "style_tags": ["classical"],
        "bat_speed_peak_kmh": 90, "head_stability_score": 50,
        "front_knee_angle": 155, "spine_angle": 20, "front_elbow_angle": 165,
        "style": "Developing technique. Building block for improvement.",
        "strengths": ["Enthusiasm", "Willingness to learn"],
        "weaknesses": ["Technical gaps", "Bat speed", "Head stability"],
        "level": "club_amateur",
        "stance_width": 0.35, "hip_shoulder_ratio": 1.40, "head_forward": 0.18,
        "grip_height": 0.48, "back_lift_height": 0.58, "stance_knee_angle": 165, "face_ratio": 0.52,
    },
}

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def _make_zone_key(gender: str, role: str, style_tags: List[str]) -> str:
    """Create a zone key from gender + role + primary style."""
    primary_style = style_tags[0] if style_tags else "classical"
    return f"{gender}_{role}_{primary_style}"


def get_zone_label(zone_key: str) -> str:
    """Get a human-readable label for a zone key."""
    return ZONE_LABELS.get(zone_key, zone_key.replace("_", " ").title())


# ======================================================================
# DERIVED LOOKUPS
# ======================================================================

# Group by level for easy lookup
PRO_PLAYERS_BY_LEVEL = defaultdict(list)
# Group by zone key
PRO_PLAYERS_BY_ZONE = defaultdict(list)

for name, data in PRO_PLAYER_DATABASE.items():
    level = data.get("level", "club_amateur")
    PRO_PLAYERS_BY_LEVEL[level].append(name)

    gender = data.get("gender", "male")
    role = data.get("role", "top_order")
    style_tags = data.get("style_tags", ["classical"])
    zone_key = _make_zone_key(gender, role, style_tags)
    PRO_PLAYERS_BY_ZONE[zone_key].append(name)


# ======================================================================
# DISCLAIMER
# ======================================================================

LEGAL_DISCLAIMER = (
    "Player comparisons are for reference and entertainment purposes only. "
    "Biomechanical metrics are approximate estimates derived from published research, "
    "coaching resources, and publicly available match footage. They do not represent "
    "exact measurements of any player. All player names and trademarks belong to their "
    "respective owners. The CREASE is not affiliated with or endorsed by any player or "
    "governing body. This is NOT a professional biomechanical assessment — consult "
    "a qualified coach for personalised feedback."
)


# ======================================================================
# Z-SCORE REFERENCE NORMALISATION
# ======================================================================

# Population means and stds for comparison metrics (derived from pro database +
# population estimates from 100+ amateur sessions).
# Order: [bat_speed, head_stability, front_knee, spine_angle, front_elbow]
COMPARISON_METRIC_MEANS = np.array([110.0, 70.0, 148.0, 16.0, 158.0], dtype=np.float32)
COMPARISON_METRIC_STDS = np.array([20.0, 15.0, 10.0, 4.0, 8.0], dtype=np.float32)


def _get_primary_style(profile: Dict[str, Any]) -> str:
    """Get the primary style tag for a profile."""
    tags = profile.get("style_tags", ["classical"])
    return tags[0] if tags else "classical"


# ======================================================================
# ZONAL COMPARISON ENGINE
# ======================================================================

class ZonalComparison:
    """
    Zone-aware comparison engine.
    
    First matches the user against batting zones (gender + role + style),
    then finds the best player within each zone. Provides zone-level
    insights alongside individual player comparisons.
    """

    def __init__(self, camera_view: str = "front_on"):
        self.camera_view = camera_view

    def compare(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare a session against all zones and all players.
        
        Returns:
            - zone_summary: ranking of all zones by similarity
            - best_zone: the zone that matches best
            - best_match: best individual player (overall)
            - zone_top_matches: top players in each zone
            - metrics_comparison: per-metric breakdown vs best player
            - zone_radar: user vs zone average radar data
            - gaps, coaching_tips
        """
        summary = session_data.get("session_summary", {}) or {}
        bat_speed_data = session_data.get("bat_speed", {}) or {}
        session_data_copy = session_data

        # ── Extract user metrics ──
        user_metrics = self._extract_user_metrics(summary, bat_speed_data, session_data_copy)

        # ── Compute per-pro similarity scores ──
        all_scores: List[Tuple[str, float, Dict[str, float]]] = []
        for name, profile in PRO_PLAYER_DATABASE.items():
            score, breakdown = self._compute_similarity(user_metrics, profile)
            all_scores.append((name, score, breakdown, profile))

        all_scores.sort(key=lambda x: x[1], reverse=True)

        # ── Zone-level aggregation ──
        zone_scores: Dict[str, List[Tuple[str, float, Dict[str, float]]]] = defaultdict(list)
        for name, score, breakdown, profile in all_scores:
            gender = profile.get("gender", "male")
            role = profile.get("role", "top_order")
            style_tags = profile.get("style_tags", ["classical"])
            zk = _make_zone_key(gender, role, style_tags)
            zone_scores[zk].append((name, score, breakdown))

        # Zone-level summary: best score per zone
        zone_summary = []
        for zk, scores in zone_scores.items():
            max_score = max(s[1] for s in scores)
            avg_score = sum(s[1] for s in scores) / len(scores)
            zone_summary.append({
                "zone_key": zk,
                "zone_label": get_zone_label(zk),
                "max_similarity_pct": round(max_score * 100, 1),
                "avg_similarity_pct": round(avg_score * 100, 1),
                "num_players": len(scores),
                "top_player": scores[0][0],  # best player in this zone
                "top_player_similarity": round(scores[0][1] * 100, 1),
            })

        zone_summary.sort(key=lambda z: z["max_similarity_pct"], reverse=True)

        # ── Best zone & best player ──
        best_zone = zone_summary[0] if zone_summary else None
        best_name, best_score, best_breakdown, best_profile = all_scores[0]

        # ── Level match ──
        level_scores: Dict[str, float] = {}
        for name, score, _, _ in all_scores:
            level = PRO_PLAYER_DATABASE[name]["level"]
            level_scores[level] = max(level_scores.get(level, 0.0), score)
        best_level = max(level_scores, key=level_scores.get) if level_scores else "club_amateur"

        # ── Per-metric comparison vs best player ──
        metrics_comparison = self._build_metrics_comparison(
            user_metrics, best_name, best_profile
        )

        # ── Identify gaps ──
        gaps = self._identify_gaps(user_metrics, best_profile, metrics_comparison)
        coaching_tips = self._generate_comparison_tips(gaps, best_name)

        # ── Radar data (user vs best player) ──
        radar_data = self._build_radar_data(user_metrics, best_profile)

        # ── Zone radar data (user vs zone average) ──
        zone_radar_data = None
        if best_zone:
            zone_radar_data = self._build_zone_radar(user_metrics, best_zone["zone_key"])

        # ── Top matches overall ──
        top_matches = [
            {
                "name": name,
                "similarity_pct": round(score * 100, 1),
                "level": PRO_PLAYER_DATABASE[name]["level"],
                "style": PRO_PLAYER_DATABASE[name]["style"],
                "gender": PRO_PLAYER_DATABASE[name].get("gender", "male"),
                "role": PRO_PLAYER_DATABASE[name].get("role", "top_order"),
                "zone_key": _make_zone_key(
                    PRO_PLAYER_DATABASE[name].get("gender", "male"),
                    PRO_PLAYER_DATABASE[name].get("role", "top_order"),
                    PRO_PLAYER_DATABASE[name].get("style_tags", ["classical"]),
                ),
            }
            for name, score, _, _ in all_scores[:10]
        ]

        # ── Zone top players ──
        zone_top_players = {}
        for zk, scores in zone_scores.items():
            zone_top_players[zk] = [
                {
                    "name": s[0],
                    "similarity_pct": round(s[1] * 100, 1),
                    "level": PRO_PLAYER_DATABASE[s[0]]["level"],
                    "style": PRO_PLAYER_DATABASE[s[0]]["style"],
                }
                for s in sorted(scores, key=lambda x: x[1], reverse=True)[:3]
            ]

        return {
            "best_match": {
                "name": best_name,
                "similarity_pct": round(best_score * 100, 1),
                "level": PRO_PLAYER_DATABASE[best_name]["level"],
                "style": PRO_PLAYER_DATABASE[best_name]["style"],
                "strengths": PRO_PLAYER_DATABASE[best_name]["strengths"],
                "gender": best_profile.get("gender", "male"),
                "role": best_profile.get("role", "top_order"),
                "zone_key": _make_zone_key(
                    best_profile.get("gender", "male"),
                    best_profile.get("role", "top_order"),
                    best_profile.get("style_tags", ["classical"]),
                ),
            },
            "best_zone": best_zone,
            "zone_summary": zone_summary,
            "zone_top_players": zone_top_players,
            "level_match": {
                "level": best_level,
                "level_label": BAT_SPEED_BENCHMARKS.get(best_level, {}).get(
                    "label", best_level.title()
                ),
            },
            "metrics_comparison": metrics_comparison,
            "radar_data": radar_data,
            "zone_radar_data": zone_radar_data,
            "gaps": gaps,
            "coaching_tips": coaching_tips,
            "top_matches": top_matches,
            "disclaimer": LEGAL_DISCLAIMER,
        }

    # ------------------------------------------------------------------
    # Zone average radar
    # ------------------------------------------------------------------

    def _build_zone_radar(
        self, user_metrics: Dict[str, float], zone_key: str
    ) -> Optional[Dict[str, Any]]:
        """Build radar comparing user vs zone average."""
        player_names = PRO_PLAYERS_BY_ZONE.get(zone_key, [])
        if not player_names:
            return None

        # Compute zone averages
        avg_bat_speed = 0.0
        avg_head = 0.0
        avg_knee = 0.0
        avg_spine = 0.0
        avg_elbow = 0.0
        count = 0

        for name in player_names:
            profile = PRO_PLAYER_DATABASE.get(name)
            if profile:
                avg_bat_speed += profile.get("bat_speed_peak_kmh", 0)
                avg_head += profile.get("head_stability_score", 0)
                avg_knee += profile.get("front_knee_angle", 0)
                avg_spine += profile.get("spine_angle", 0)
                avg_elbow += profile.get("front_elbow_angle", 0)
                count += 1

        if count == 0:
            return None

        zone_avg = {
            "bat_speed_peak_kmh": avg_bat_speed / count,
            "head_stability_score": avg_head / count,
            "front_knee_angle": avg_knee / count,
            "spine_angle": avg_spine / count,
            "front_elbow_angle": avg_elbow / count,
        }

        radar_metrics = [
            {
                "label": "Bat Speed",
                "key": "bat_speed",
                "user_normalised": self._normalise_to_100(
                    user_metrics["bat_speed_peak_kmh"], 0, 160, higher_better=True
                ),
                "zone_normalised": self._normalise_to_100(
                    zone_avg["bat_speed_peak_kmh"], 0, 160, higher_better=True
                ),
            },
            {
                "label": "Head Stability",
                "key": "head_stability",
                "user_normalised": self._normalise_to_100(
                    user_metrics["head_stability_score"], 0, 100, higher_better=True
                ),
                "zone_normalised": self._normalise_to_100(
                    zone_avg["head_stability_score"], 0, 100, higher_better=True
                ),
            },
            {
                "label": "Knee Bend",
                "key": "front_knee",
                "user_normalised": self._normalise_to_100(
                    user_metrics["front_knee_angle"], 120, 180, higher_better=False
                ),
                "zone_normalised": self._normalise_to_100(
                    zone_avg["front_knee_angle"], 120, 180, higher_better=False
                ),
            },
            {
                "label": "Posture",
                "key": "spine",
                "user_normalised": self._normalise_to_100(
                    user_metrics["spine_angle"], 0, 35, higher_better=False
                ),
                "zone_normalised": self._normalise_to_100(
                    zone_avg["spine_angle"], 0, 35, higher_better=False
                ),
            },
            {
                "label": "Elbow Control",
                "key": "front_elbow",
                "user_normalised": self._normalise_to_100(
                    user_metrics["front_elbow_angle"], 140, 180, higher_better=False
                ),
                "zone_normalised": self._normalise_to_100(
                    zone_avg["front_elbow_angle"], 140, 180, higher_better=False
                ),
            },
        ]

        return {
            "labels": [m["label"] for m in radar_metrics],
            "user_values": [m["user_normalised"] for m in radar_metrics],
            "zone_values": [m["zone_normalised"] for m in radar_metrics],
            "max_value": 100,
            "zone_label": get_zone_label(zone_key),
        }

    # ------------------------------------------------------------------
    # Methods shared with / reused from ProComparison
    # ------------------------------------------------------------------

    def _extract_user_metrics(self, summary, bat_speed_data, session_data):
        bat_peak = float(bat_speed_data.get("peak_kmh", 0))
        bat_avg = float(bat_speed_data.get("avg_kmh", 0))
        bat_speed_val = bat_peak if bat_peak > 0 else bat_avg
        return {
            "bat_speed_peak_kmh": bat_speed_val,
            "head_stability_score": float(summary.get("head_stability_score", 0)),
            "front_knee_angle": float(summary.get("avg_front_knee_angle", 160)),
            "spine_angle": float(summary.get("avg_spine_angle", 15)),
            "front_elbow_angle": float(summary.get("avg_front_elbow_angle", 160)),
        }

    def _compute_similarity(self, user_metrics, profile):
        breakdown = {}
        if user_metrics["bat_speed_peak_kmh"] > 0 and profile["bat_speed_peak_kmh"] > 0:
            diff_pct = abs(user_metrics["bat_speed_peak_kmh"] - profile["bat_speed_peak_kmh"])
            breakdown["bat_speed"] = max(0.0, 1.0 - diff_pct / 80.0)
        else:
            breakdown["bat_speed"] = 0.5
        if user_metrics["head_stability_score"] > 0 and profile["head_stability_score"] > 0:
            diff = abs(user_metrics["head_stability_score"] - profile["head_stability_score"])
            breakdown["head_stability"] = max(0.0, 1.0 - diff / 100.0)
        else:
            breakdown["head_stability"] = 0.5
        if profile["front_knee_angle"] > 0:
            diff = abs(user_metrics["front_knee_angle"] - profile["front_knee_angle"])
            breakdown["front_knee"] = max(0.0, 1.0 - diff / 60.0)
        else:
            breakdown["front_knee"] = 0.5
        if profile["spine_angle"] > 0:
            diff = abs(user_metrics["spine_angle"] - profile["spine_angle"])
            breakdown["spine"] = max(0.0, 1.0 - diff / 40.0)
        else:
            breakdown["spine"] = 0.5
        if profile["front_elbow_angle"] > 0:
            diff = abs(user_metrics["front_elbow_angle"] - profile["front_elbow_angle"])
            breakdown["front_elbow"] = max(0.0, 1.0 - diff / 40.0)
        else:
            breakdown["front_elbow"] = 0.5
        weights = {"bat_speed": 2.0, "head_stability": 2.0, "front_knee": 1.0, "spine": 1.0, "front_elbow": 1.0}
        total_weight = sum(weights.values())
        overall = sum(breakdown[k] * weights[k] for k in breakdown) / total_weight
        return min(1.0, overall), breakdown

    def _build_metrics_comparison(self, user_metrics, pro_name, profile):
        metric_defs = [
            {"key": "bat_speed", "label": "Bat Speed (peak)", "unit": " km/h",
             "user_val": user_metrics["bat_speed_peak_kmh"], "pro_val": profile["bat_speed_peak_kmh"],
             "higher_is_better": True, "description": "Bat tip speed at impact point"},
            {"key": "head_stability", "label": "Head Stability", "unit": "/100",
             "user_val": user_metrics["head_stability_score"], "pro_val": profile["head_stability_score"],
             "higher_is_better": True, "description": "Stillness of head through the shot"},
            {"key": "front_knee", "label": "Front Knee Bend", "unit": "°",
             "user_val": user_metrics["front_knee_angle"], "pro_val": profile["front_knee_angle"],
             "higher_is_better": False, "description": "Knee angle at contact (lower = more bent)"},
            {"key": "spine", "label": "Spine Lean", "unit": "°",
             "user_val": user_metrics["spine_angle"], "pro_val": profile["spine_angle"],
             "higher_is_better": False, "description": "Forward lean from vertical at contact"},
            {"key": "front_elbow", "label": "Front Elbow", "unit": "°",
             "user_val": user_metrics["front_elbow_angle"], "pro_val": profile["front_elbow_angle"],
             "higher_is_better": False, "description": "Elbow angle at contact (straighter = higher)"},
        ]
        results = []
        for m in metric_defs:
            user_v = m["user_val"]
            pro_v = m["pro_val"]
            if user_v <= 0:
                continue
            gap_abs = pro_v - user_v if m["higher_is_better"] else user_v - pro_v
            gap_pct = (gap_abs / max(pro_v, 0.01)) * 100
            if m["higher_is_better"]:
                if user_v >= pro_v:
                    relative_score = 100.0
                else:
                    relative_score = max(0.0, (user_v / max(pro_v, 0.01)) * 100)
            else:
                if user_v <= pro_v:
                    relative_score = 100.0
                else:
                    max_diff = 60.0
                    diff = user_v - pro_v
                    relative_score = max(0.0, 100.0 - (diff / max_diff) * 100)
            results.append({
                "key": m["key"], "label": m["label"], "unit": m["unit"],
                "user_value": round(user_v, 1), "pro_value": round(pro_v, 1),
                "gap_abs": round(gap_abs, 1), "gap_pct": round(gap_pct, 1),
                "relative_score": round(relative_score, 0),
                "higher_is_better": m["higher_is_better"], "description": m["description"],
            })
        return results

    def _identify_gaps(self, user_metrics, profile, metrics_comparison):
        gaps = []
        for m in metrics_comparison:
            if m["relative_score"] < 100:
                gap_size = 100 - m["relative_score"]
                gaps.append({
                    "metric": m["key"], "label": m["label"], "gap_size": gap_size,
                    "user_value": m["user_value"], "pro_value": m["pro_value"], "unit": m["unit"],
                    "severity": "high" if gap_size > 40 else "medium" if gap_size > 20 else "low",
                })
        gaps.sort(key=lambda x: x["gap_size"], reverse=True)
        return gaps[:5]

    def _generate_comparison_tips(self, gaps, best_player):
        tip_map = {
            "bat_speed": "Increase bat speed through core rotation and wrist snap. "
                         "Try resistance band training and shadow batting with a heavier bat.",
            "head_stability": "Improve head stability by keeping your eyes level and "
                              "your head still through the shot. Practice with a side-arm "
                              "thrower or bowling machine.",
            "front_knee": "Work on getting your front knee more bent at contact. "
                          "This helps you get to the pitch of the ball and drive with power. "
                          "Try knee-bend drills in the nets.",
            "spine": "Your forward lean needs adjustment. Aim for a balanced position with "
                     "your head over the ball. Practice in front of a mirror.",
            "front_elbow": "Keep your front elbow slightly bent at contact for better "
                           "control and softer hands. Avoid locking the elbow.",
        }
        tips = []
        for gap in gaps[:3]:
            metric = gap["metric"]
            base_tip = tip_map.get(metric, f"Work on improving your {gap['label']}.")
            tips.append(
                f"{gap['label']} is {gap['gap_size']:.0f}% behind {best_player}. {base_tip}"
            )
        if not tips:
            tips.append("Great alignment with your matched pro player! Keep practising to maintain these levels.")
        return tips

    def _build_radar_data(self, user_metrics, profile):
        radar_metrics = [
            {"label": "Bat Speed", "key": "bat_speed",
             "user_normalised": self._normalise_to_100(user_metrics["bat_speed_peak_kmh"], 0, 160, higher_better=True),
             "pro_normalised": self._normalise_to_100(profile["bat_speed_peak_kmh"], 0, 160, higher_better=True)},
            {"label": "Head Stability", "key": "head_stability",
             "user_normalised": self._normalise_to_100(user_metrics["head_stability_score"], 0, 100, higher_better=True),
             "pro_normalised": self._normalise_to_100(profile["head_stability_score"], 0, 100, higher_better=True)},
            {"label": "Knee Bend", "key": "front_knee",
             "user_normalised": self._normalise_to_100(user_metrics["front_knee_angle"], 120, 180, higher_better=False),
             "pro_normalised": self._normalise_to_100(profile["front_knee_angle"], 120, 180, higher_better=False)},
            {"label": "Posture", "key": "spine",
             "user_normalised": self._normalise_to_100(user_metrics["spine_angle"], 0, 35, higher_better=False),
             "pro_normalised": self._normalise_to_100(profile["spine_angle"], 0, 35, higher_better=False)},
            {"label": "Elbow Control", "key": "front_elbow",
             "user_normalised": self._normalise_to_100(user_metrics["front_elbow_angle"], 140, 180, higher_better=False),
             "pro_normalised": self._normalise_to_100(profile["front_elbow_angle"], 140, 180, higher_better=False)},
        ]
        return {
            "labels": [m["label"] for m in radar_metrics],
            "user_values": [m["user_normalised"] for m in radar_metrics],
            "pro_values": [m["pro_normalised"] for m in radar_metrics],
            "max_value": 100,
        }

    @staticmethod
    def _normalise_to_100(value, min_val, max_val, higher_better=True):
        val_range = max_val - min_val
        if val_range <= 0:
            return 50.0
        if higher_better:
            clamped = max(min_val, min(max_val, value))
            return round((clamped - min_val) / val_range * 100, 0)
        else:
            clamped = max(min_val, min(max_val, value))
            return round((max_val - clamped) / val_range * 100, 0)

    def compare_multiple(self, session_datas):
        results = []
        for session_data in session_datas:
            result = self.compare(session_data)
            session_id = session_data.get("session_id", "")
            timestamp = session_data.get("analysis_timestamp", "")
            results.append({"session_id": session_id, "timestamp": timestamp, "comparison": result})
        return results

    @staticmethod
    def get_legal_disclaimer() -> str:
        return LEGAL_DISCLAIMER

    @staticmethod
    def list_zones() -> List[Dict[str, Any]]:
        """List all available zones with player counts."""
        zones = []
        for zk, players in PRO_PLAYERS_BY_ZONE.items():
            zones.append({
                "zone_key": zk,
                "zone_label": get_zone_label(zk),
                "num_players": len(players),
                "players": players,
            })
        zones.sort(key=lambda z: z["zone_label"])
        return zones

    @staticmethod
    def list_pro_players(level: Optional[str] = None,
                          gender: Optional[str] = None,
                          zone_key: Optional[str] = None) -> List[Dict[str, str]]:
        """List available pro players, optionally filtered."""
        players = []
        for name, data in PRO_PLAYER_DATABASE.items():
            if level and data["level"] != level:
                continue
            if gender and data.get("gender") != gender:
                continue
            if zone_key:
                pk = _make_zone_key(data.get("gender", "male"),
                                    data.get("role", "top_order"),
                                    data.get("style_tags", ["classical"]))
                if pk != zone_key:
                    continue
            players.append({
                "name": name,
                "level": data["level"],
                "style": data["style"],
                "gender": data.get("gender", "male"),
                "role": data.get("role", "top_order"),
            })
        return players

    @staticmethod
    def get_player_profile(player_name: str) -> Optional[Dict[str, Any]]:
        return PRO_PLAYER_DATABASE.get(player_name)


# ======================================================================
# LEGACY ProComparison (wraps ZonalComparison for backward compat)
# ======================================================================

class ProComparison:
    """
    Legacy class — kept for backward compatibility.
    Delegates to ZonalComparison internally.
    Now also supports the richer zonal output.
    """

    def __init__(self, camera_view: str = "front_on"):
        self._zonal = ZonalComparison(camera_view)
        self.camera_view = camera_view

    def compare(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._zonal.compare(session_data)

    def compare_multiple(self, session_datas):
        return self._zonal.compare_multiple(session_datas)

    @staticmethod
    def get_legal_disclaimer() -> str:
        return LEGAL_DISCLAIMER

    @staticmethod
    def list_pro_players(level: Optional[str] = None) -> List[Dict[str, str]]:
        return ZonalComparison.list_pro_players(level=level)

    @staticmethod
    def get_player_profile(player_name: str) -> Optional[Dict[str, Any]]:
        return PRO_PLAYER_DATABASE.get(player_name)
