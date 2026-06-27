"""
Pro Comparison — Compare your batting biomechanics against professional players.

Provides a comprehensive comparison engine that matches a user's session metrics
against a reference database of professional player profiles. Uses published,
publicly available data and coaching references.

LEGAL: All player names are used as secondary factual comparisons only. Level labels
(Club, State, International) are primary. A legal disclaimer is included on every
comparison output. Player reference data is approximate and based on published
biomechanics research, coaching manuals, and publicly available player information.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Optional, Tuple

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

from .player_profiler import (
    extract_stance_signature,
    signature_to_vector,
    cosine_similarity,
)


# ======================================================================
# PROFESSIONAL PLAYER BIOMECHANICAL DATABASE
# ======================================================================
# Reference profiles for professional batsmen.
# Sources: Published biomechanics research, ECB/Cricket Australia coaching
# resources, publicly available match footage analysis.
#
# All values are APPROXIMATE reference ranges for comparison purposes.
# Individual player biomechanics vary between formats, conditions, and shot types.
#
# Each profile includes:
#   - bat_speed: peak_kmh (bat tip speed at impact)
#   - head_stability: score (0-100, higher = better)
#   - front_knee_angle: degrees at contact (side-on reference)
#   - spine_angle: degrees from vertical at contact
#   - front_elbow_angle: degrees at contact
#   - stance_signature: 7-feature stance profile (for similarity matching)
#   - style: brief description of playing style
#   - level: "international", "state_pro", "club_premier", "club_amateur"
# ======================================================================

PRO_PLAYER_DATABASE: Dict[str, Dict[str, Any]] = {
    "Virat Kohli": {
        "bat_speed_peak_kmh": 135,
        "head_stability_score": 95,
        "front_knee_angle": 138,
        "spine_angle": 15,
        "front_elbow_angle": 155,
        "style": "Controlled aggression — head still, hands fast. Textbook front-foot play.",
        "strengths": ["Head stability", "Front-foot driving", "Weight transfer"],
        "weaknesses": ["Occasional nibble outside off", "Can be late on short ball"],
        "level": "international",
        "stance_width": 0.28,
        "hip_shoulder_ratio": 1.35,
        "head_forward": 0.15,
        "grip_height": 0.55,
        "back_lift_height": 0.72,
        "stance_knee_angle": 158,
        "face_ratio": 0.65,
    },
    "Rohit Sharma": {
        "bat_speed_peak_kmh": 145,
        "head_stability_score": 88,
        "front_knee_angle": 142,
        "spine_angle": 12,
        "front_elbow_angle": 160,
        "style": "Elegant power — late on the ball, insane wrists. Minimal foot movement, maximum timing.",
        "strengths": ["Pull shot", "Cut shot", "Lofted drives", "Bat speed"],
        "weaknesses": ["Early stages of innings", "Movement off the pitch"],
        "level": "international",
        "stance_width": 0.32,
        "hip_shoulder_ratio": 1.40,
        "head_forward": 0.10,
        "grip_height": 0.50,
        "back_lift_height": 0.68,
        "stance_knee_angle": 162,
        "face_ratio": 0.58,
    },
    "Steve Smith": {
        "bat_speed_peak_kmh": 128,
        "head_stability_score": 85,
        "front_knee_angle": 148,
        "spine_angle": 18,
        "front_elbow_angle": 150,
        "style": "Unorthodox — all wrists and core, deceptively quick. Unique trigger movements.",
        "strengths": ["Leg-side play", "Against spin", "Concentration"],
        "weaknesses": ["Wobble seam outside off", "LBW vulnerable"],
        "level": "international",
        "stance_width": 0.38,
        "hip_shoulder_ratio": 1.30,
        "head_forward": 0.22,
        "grip_height": 0.52,
        "back_lift_height": 0.65,
        "stance_knee_angle": 165,
        "face_ratio": 0.52,
    },
    "Joe Root": {
        "bat_speed_peak_kmh": 125,
        "head_stability_score": 88,
        "front_knee_angle": 140,
        "spine_angle": 14,
        "front_elbow_angle": 158,
        "style": "Classical — weighted transfer, smooth through the line. Traditional batting technique.",
        "strengths": ["Cover drive", "Against spin", "Running between wickets"],
        "weaknesses": ["Pad-play outside off", "Occasional soft dismissal"],
        "level": "international",
        "stance_width": 0.30,
        "hip_shoulder_ratio": 1.38,
        "head_forward": 0.12,
        "grip_height": 0.58,
        "back_lift_height": 0.70,
        "stance_knee_angle": 155,
        "face_ratio": 0.62,
    },
    "Kane Williamson": {
        "bat_speed_peak_kmh": 122,
        "head_stability_score": 92,
        "front_knee_angle": 136,
        "spine_angle": 16,
        "front_elbow_angle": 156,
        "style": "Technically sound — exceptionally still head, minimal trigger. Timing over power.",
        "strengths": ["Head stability", "Off-side play", "Adaptability"],
        "weaknesses": ["Power hitting", "Express pace"],
        "level": "international",
        "stance_width": 0.26,
        "hip_shoulder_ratio": 1.32,
        "head_forward": 0.14,
        "grip_height": 0.56,
        "back_lift_height": 0.69,
        "stance_knee_angle": 160,
        "face_ratio": 0.60,
    },
    "David Warner": {
        "bat_speed_peak_kmh": 140,
        "head_stability_score": 78,
        "front_knee_angle": 150,
        "spine_angle": 10,
        "front_elbow_angle": 162,
        "style": "Explosive — uses depth of crease and fast hands. Aggressive intent from ball one.",
        "strengths": ["Power hitting", "Cut and pull", "Bat speed"],
        "weaknesses": ["Movement outside off", "Left-arm pace"],
        "level": "international",
        "stance_width": 0.34,
        "hip_shoulder_ratio": 1.42,
        "head_forward": 0.08,
        "grip_height": 0.48,
        "back_lift_height": 0.66,
        "stance_knee_angle": 168,
        "face_ratio": 0.55,
    },
    "AB de Villiers": {
        "bat_speed_peak_kmh": 150,
        "head_stability_score": 90,
        "front_knee_angle": 135,
        "spine_angle": 13,
        "front_elbow_angle": 152,
        "style": "Freakish — can change shot in 0.2s, bat speed from anywhere. 360-degree play.",
        "strengths": ["Innovation", "Bat speed", "All-round play"],
        "weaknesses": ["None clearly exploitable"],
        "level": "international",
        "stance_width": 0.30,
        "hip_shoulder_ratio": 1.36,
        "head_forward": 0.13,
        "grip_height": 0.54,
        "back_lift_height": 0.71,
        "stance_knee_angle": 156,
        "face_ratio": 0.63,
    },
    "Ben Stokes": {
        "bat_speed_peak_kmh": 142,
        "head_stability_score": 75,
        "front_knee_angle": 145,
        "spine_angle": 18,
        "front_elbow_angle": 158,
        "style": "Brute power — strong core, clears front leg. Match-winner under pressure.",
        "strengths": ["Power hitting", "Counter-attack", "Against pace"],
        "weaknesses": ["Against spin", "Occasional recklessness"],
        "level": "international",
        "stance_width": 0.36,
        "hip_shoulder_ratio": 1.44,
        "head_forward": 0.18,
        "grip_height": 0.50,
        "back_lift_height": 0.64,
        "stance_knee_angle": 164,
        "face_ratio": 0.50,
    },
    "Marnus Labuschagne": {
        "bat_speed_peak_kmh": 125,
        "head_stability_score": 78,
        "front_knee_angle": 142,
        "spine_angle": 16,
        "front_elbow_angle": 156,
        "style": "Active trigger but head settles well before impact. Compulsive tinkerer.",
        "strengths": ["Concentration", "Against spin", "Leg-side play"],
        "weaknesses": ["Movement outside off", "Express pace"],
        "level": "international",
        "stance_width": 0.32,
        "hip_shoulder_ratio": 1.34,
        "head_forward": 0.16,
        "grip_height": 0.53,
        "back_lift_height": 0.67,
        "stance_knee_angle": 162,
        "face_ratio": 0.57,
    },
    "Babar Azam": {
        "bat_speed_peak_kmh": 130,
        "head_stability_score": 90,
        "front_knee_angle": 136,
        "spine_angle": 14,
        "front_elbow_angle": 154,
        "style": "Silk-smooth — textbook technique, superb timing. Cover drive is signature.",
        "strengths": ["Cover drive", "Head stability", "Timing"],
        "weaknesses": ["Short ball", "Pressure situations"],
        "level": "international",
        "stance_width": 0.28,
        "hip_shoulder_ratio": 1.37,
        "head_forward": 0.12,
        "grip_height": 0.57,
        "back_lift_height": 0.71,
        "stance_knee_angle": 157,
        "face_ratio": 0.64,
    },
    "Travis Head": {
        "bat_speed_peak_kmh": 138,
        "head_stability_score": 72,
        "front_knee_angle": 148,
        "spine_angle": 12,
        "front_elbow_angle": 160,
        "style": "Aggressive — strong on the cut and pull. Attacks spin with intent.",
        "strengths": ["Against spin", "Cut and pull", "Bat speed"],
        "weaknesses": ["Outside off", "Short ball judgement"],
        "level": "international",
        "stance_width": 0.33,
        "hip_shoulder_ratio": 1.40,
        "head_forward": 0.09,
        "grip_height": 0.49,
        "back_lift_height": 0.66,
        "stance_knee_angle": 166,
        "face_ratio": 0.53,
    },
    "Shubman Gill": {
        "bat_speed_peak_kmh": 132,
        "head_stability_score": 86,
        "front_knee_angle": 140,
        "spine_angle": 14,
        "front_elbow_angle": 156,
        "style": "Elegant — classic off-side play, good head position. Modern technique.",
        "strengths": ["Cover drive", "Head position", "Against pace"],
        "weaknesses": ["Short ball", "Spin on turning tracks"],
        "level": "international",
        "stance_width": 0.29,
        "hip_shoulder_ratio": 1.36,
        "head_forward": 0.13,
        "grip_height": 0.56,
        "back_lift_height": 0.70,
        "stance_knee_angle": 158,
        "face_ratio": 0.61,
    },
    # State / First-Class reference profiles
    "First-Class Batter (Model)": {
        "bat_speed_peak_kmh": 125,
        "head_stability_score": 75,
        "front_knee_angle": 142,
        "spine_angle": 16,
        "front_elbow_angle": 157,
        "style": "Professional standard. Consistent technique, reliable against pace and spin.",
        "strengths": ["Consistency", "Shot selection", "Technique"],
        "weaknesses": ["Power hitting ceiling", "Adaptability to conditions"],
        "level": "state_pro",
        "stance_width": 0.31,
        "hip_shoulder_ratio": 1.36,
        "head_forward": 0.14,
        "grip_height": 0.54,
        "back_lift_height": 0.66,
        "stance_knee_angle": 160,
        "face_ratio": 0.59,
    },
    # Grade / Premier reference profiles
    "Premier Club Batter (Model)": {
        "bat_speed_peak_kmh": 110,
        "head_stability_score": 65,
        "front_knee_angle": 148,
        "spine_angle": 18,
        "front_elbow_angle": 160,
        "style": "Good club standard. Solid technique with occasional lapses in concentration.",
        "strengths": ["Reliable defence", "Good club cricket player"],
        "weaknesses": ["Bat speed", "Head movement", "Against quality bowling"],
        "level": "club_premier",
        "stance_width": 0.33,
        "hip_shoulder_ratio": 1.38,
        "head_forward": 0.16,
        "grip_height": 0.52,
        "back_lift_height": 0.62,
        "stance_knee_angle": 162,
        "face_ratio": 0.55,
    },
    # Amateur reference profile
    "Amateur Batter (Model)": {
        "bat_speed_peak_kmh": 90,
        "head_stability_score": 50,
        "front_knee_angle": 155,
        "spine_angle": 20,
        "front_elbow_angle": 165,
        "style": "Developing technique. Building block for improvement.",
        "strengths": ["Enthusiasm", "Willingness to learn"],
        "weaknesses": ["Technical gaps", "Bat speed", "Head stability"],
        "level": "club_amateur",
        "stance_width": 0.35,
        "hip_shoulder_ratio": 1.40,
        "head_forward": 0.18,
        "grip_height": 0.48,
        "back_lift_height": 0.58,
        "stance_knee_angle": 165,
        "face_ratio": 0.52,
    },
}

# Group by level for easy lookup
PRO_PLAYERS_BY_LEVEL = {
    "international": [name for name, data in PRO_PLAYER_DATABASE.items()
                      if data["level"] == "international"],
    "state_pro": [name for name, data in PRO_PLAYER_DATABASE.items()
                  if data["level"] == "state_pro"],
    "club_premier": [name for name, data in PRO_PLAYER_DATABASE.items()
                     if data["level"] == "club_premier"],
    "club_amateur": [name for name, data in PRO_PLAYER_DATABASE.items()
                     if data["level"] == "club_amateur"],
}


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


# ======================================================================
# COMPARISON ENGINE
# ======================================================================

class ProComparison:
    """
    Compare a user's session metrics against professional player profiles.

    Provides:
      - Best matching pro player by overall similarity
      - Per-metric scores and gaps
      - Radar chart data (normalised 0-100)
      - Coaching tips based on gaps
      - Legal disclaimer
    """

    def __init__(self, camera_view: str = "front_on"):
        """
        Args:
            camera_view: 'front_on', 'side_off', 'side_leg', 'angled', 'behind'
        """
        self.camera_view = camera_view

    # ------------------------------------------------------------------
    # Main comparison method
    # ------------------------------------------------------------------

    def compare(
        self,
        session_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare a completed session against pro player profiles.

        Args:
            session_data: Full session analysis result dict (from JSON file)

        Returns:
            Dict with:
              - best_match: nearest pro player profile
              - level_match: best matching level (club/state/international)
              - metrics_comparison: per-metric comparison with scores
              - radar_data: normalised values for radar chart (user + nearest pro)
              - gaps: key areas for improvement
              - coaching_tips: actionable tips
              - disclaimer: legal disclaimer text
        """
        summary = session_data.get("session_summary", {}) or {}
        bat_speed_data = session_data.get("bat_speed", {}) or {}
        shot_summary = session_data.get("shot_summary", []) or []

        # ── Extract user metrics ──
        user_metrics = self._extract_user_metrics(summary, bat_speed_data, session_data)

        # ── Compute per-pro similarity scores ──
        pro_scores: List[Tuple[str, float, Dict[str, Any]]] = []
        for name, profile in PRO_PLAYER_DATABASE.items():
            score, breakdown = self._compute_similarity(user_metrics, profile)
            pro_scores.append((name, score, breakdown))

        # Sort by similarity (descending)
        pro_scores.sort(key=lambda x: x[1], reverse=True)

        # ── Best match player ──
        best_name, best_score, best_breakdown = pro_scores[0]

        # Also find best level match
        level_scores: Dict[str, float] = {}
        for name, score, _ in pro_scores:
            level = PRO_PLAYER_DATABASE[name]["level"]
            level_scores[level] = max(level_scores.get(level, 0.0), score)
        best_level = max(level_scores, key=level_scores.get)  # type: ignore

        # ── Per-metric comparison ──
        metrics_comparison = self._build_metrics_comparison(
            user_metrics, best_name, PRO_PLAYER_DATABASE[best_name]
        )

        # ── Identify biggest gaps ──
        gaps = self._identify_gaps(user_metrics, PRO_PLAYER_DATABASE[best_name], metrics_comparison)
        coaching_tips = self._generate_comparison_tips(gaps, best_name)

        # ── Radar data ──
        radar_data = self._build_radar_data(
            user_metrics, PRO_PLAYER_DATABASE[best_name]
        )

        # ── Top 5 similar players ──
        top_matches = [
            {
                "name": name,
                "similarity_pct": round(score * 100, 1),
                "level": PRO_PLAYER_DATABASE[name]["level"],
                "style": PRO_PLAYER_DATABASE[name]["style"],
            }
            for name, score, _ in pro_scores[:5]
        ]

        return {
            "best_match": {
                "name": best_name,
                "similarity_pct": round(best_score * 100, 1),
                "level": PRO_PLAYER_DATABASE[best_name]["level"],
                "style": PRO_PLAYER_DATABASE[best_name]["style"],
                "strengths": PRO_PLAYER_DATABASE[best_name]["strengths"],
            },
            "level_match": {
                "level": best_level,
                "level_label": BAT_SPEED_BENCHMARKS.get(best_level, {}).get(
                    "label", best_level.title()
                ),
            },
            "metrics_comparison": metrics_comparison,
            "radar_data": radar_data,
            "gaps": gaps,
            "coaching_tips": coaching_tips,
            "top_matches": top_matches,
            "disclaimer": LEGAL_DISCLAIMER,
        }

    # ------------------------------------------------------------------
    # Metric extraction
    # ------------------------------------------------------------------

    def _extract_user_metrics(
        self,
        summary: Dict[str, Any],
        bat_speed_data: Dict[str, Any],
        session_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Extract normalised metrics from a session for comparison.
        """
        # Bat speed: use peak_kmh, fallback to avg or 0
        bat_peak = float(bat_speed_data.get("peak_kmh", 0))
        bat_avg = float(bat_speed_data.get("avg_kmh", 0))
        bat_speed_val = bat_peak if bat_peak > 0 else bat_avg

        metrics = {
            "bat_speed_peak_kmh": bat_speed_val,
            "head_stability_score": float(summary.get("head_stability_score", 0)),
            "front_knee_angle": float(summary.get("avg_front_knee_angle", 160)),
            "spine_angle": float(summary.get("avg_spine_angle", 15)),
            "front_elbow_angle": float(summary.get("avg_front_elbow_angle", 160)),
        }

        return metrics

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def _compute_similarity(
        self,
        user_metrics: Dict[str, float],
        profile: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute similarity between user metrics and a pro profile.
        Returns (overall_similarity_0to1, breakdown_by_metric).
        """
        breakdown = {}

        # 1. Bat speed similarity (gaussian around pro value)
        if user_metrics["bat_speed_peak_kmh"] > 0 and profile["bat_speed_peak_kmh"] > 0:
            diff_pct = abs(user_metrics["bat_speed_peak_kmh"] - profile["bat_speed_peak_kmh"])
            breakdown["bat_speed"] = max(0.0, 1.0 - diff_pct / 80.0)
        else:
            breakdown["bat_speed"] = 0.5

        # 2. Head stability similarity
        if user_metrics["head_stability_score"] > 0 and profile["head_stability_score"] > 0:
            diff = abs(user_metrics["head_stability_score"] - profile["head_stability_score"])
            breakdown["head_stability"] = max(0.0, 1.0 - diff / 100.0)
        else:
            breakdown["head_stability"] = 0.5

        # 3. Front knee angle similarity
        if profile["front_knee_angle"] > 0:
            diff = abs(user_metrics["front_knee_angle"] - profile["front_knee_angle"])
            breakdown["front_knee"] = max(0.0, 1.0 - diff / 60.0)
        else:
            breakdown["front_knee"] = 0.5

        # 4. Spine angle similarity
        if profile["spine_angle"] > 0:
            diff = abs(user_metrics["spine_angle"] - profile["spine_angle"])
            breakdown["spine"] = max(0.0, 1.0 - diff / 40.0)
        else:
            breakdown["spine"] = 0.5

        # 5. Front elbow similarity
        if profile["front_elbow_angle"] > 0:
            diff = abs(user_metrics["front_elbow_angle"] - profile["front_elbow_angle"])
            breakdown["front_elbow"] = max(0.0, 1.0 - diff / 40.0)
        else:
            breakdown["front_elbow"] = 0.5

        # Weighted overall: bat speed ×2, head stability ×2 (key differentiators),
        # others ×1
        weights = {
            "bat_speed": 2.0,
            "head_stability": 2.0,
            "front_knee": 1.0,
            "spine": 1.0,
            "front_elbow": 1.0,
        }
        total_weight = sum(weights.values())
        overall = sum(breakdown[k] * weights[k] for k in breakdown) / total_weight

        # Apply stance signature bonus if available (up to +5%)
        # (We don't have user stance signature extracted here, so skip)

        return min(1.0, overall), breakdown

    # ------------------------------------------------------------------
    # Metrics comparison table builder
    # ------------------------------------------------------------------

    def _build_metrics_comparison(
        self,
        user_metrics: Dict[str, float],
        pro_name: str,
        profile: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Build a per-metric comparison table.
        """
        # Define metrics to display with labels and interpretation
        metric_defs = [
            {
                "key": "bat_speed",
                "label": "Bat Speed (peak)",
                "unit": "km/h",
                "user_val": user_metrics["bat_speed_peak_kmh"],
                "pro_val": profile["bat_speed_peak_kmh"],
                "higher_is_better": True,
                "description": "Bat tip speed at impact point",
                "benchmark_level": BAT_SPEED_BENCHMARKS,
            },
            {
                "key": "head_stability",
                "label": "Head Stability",
                "unit": "/100",
                "user_val": user_metrics["head_stability_score"],
                "pro_val": profile["head_stability_score"],
                "higher_is_better": True,
                "description": "Stillness of head through the shot",
                "benchmark_level": HEAD_STABILITY_BENCHMARKS,
            },
            {
                "key": "front_knee",
                "label": "Front Knee Bend",
                "unit": "°",
                "user_val": user_metrics["front_knee_angle"],
                "pro_val": profile["front_knee_angle"],
                "higher_is_better": False,
                "description": "Knee angle at contact (lower = more bent)",
                "benchmark_level": FRONT_KNEE_BENCHMARKS,
            },
            {
                "key": "spine",
                "label": "Spine Lean",
                "unit": "°",
                "user_val": user_metrics["spine_angle"],
                "pro_val": profile["spine_angle"],
                "higher_is_better": False,
                "description": "Forward lean from vertical at contact",
                "benchmark_level": SPINE_ANGLE_BENCHMARKS,
            },
            {
                "key": "front_elbow",
                "label": "Front Elbow",
                "unit": "°",
                "user_val": user_metrics["front_elbow_angle"],
                "pro_val": profile["front_elbow_angle"],
                "higher_is_better": False,
                "description": "Elbow angle at contact (straighter = higher)",
                "benchmark_level": None,
            },
        ]

        results = []
        for m in metric_defs:
            user_v = m["user_val"]
            pro_v = m["pro_val"]
            if user_v <= 0:
                continue

            # Gap: absolute and percentage
            gap_abs = pro_v - user_v if m["higher_is_better"] else user_v - pro_v
            gap_pct = (gap_abs / max(pro_v, 0.01)) * 100

            # User score relative to pro (0-100)
            if m["higher_is_better"]:
                if user_v >= pro_v:
                    relative_score = 100.0
                else:
                    relative_score = max(0.0, (user_v / max(pro_v, 0.01)) * 100)
            else:
                if user_v <= pro_v:
                    relative_score = 100.0
                else:
                    # Lower is better: further from pro = lower score
                    max_diff = 60.0
                    diff = user_v - pro_v
                    relative_score = max(0.0, 100.0 - (diff / max_diff) * 100)

            results.append({
                "key": m["key"],
                "label": m["label"],
                "unit": m["unit"],
                "user_value": round(user_v, 1),
                "pro_value": round(pro_v, 1),
                "gap_abs": round(gap_abs, 1),
                "gap_pct": round(gap_pct, 1),
                "relative_score": round(relative_score, 0),
                "higher_is_better": m["higher_is_better"],
                "description": m["description"],
            })

        return results

    # ------------------------------------------------------------------
    # Gap identification
    # ------------------------------------------------------------------

    def _identify_gaps(
        self,
        user_metrics: Dict[str, float],
        profile: Dict[str, Any],
        metrics_comparison: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Identify the biggest gaps between user and pro.
        Returns top 3 gaps sorted by magnitude.
        """
        gaps = []
        for m in metrics_comparison:
            if m["relative_score"] < 100:
                gap_size = 100 - m["relative_score"]
                gaps.append({
                    "metric": m["key"],
                    "label": m["label"],
                    "gap_size": gap_size,
                    "user_value": m["user_value"],
                    "pro_value": m["pro_value"],
                    "unit": m["unit"],
                    "severity": "high" if gap_size > 40 else "medium" if gap_size > 20 else "low",
                })

        gaps.sort(key=lambda x: x["gap_size"], reverse=True)
        return gaps[:5]

    def _generate_comparison_tips(
        self,
        gaps: List[Dict[str, Any]],
        best_player: str,
    ) -> List[str]:
        """
        Convert gaps into actionable coaching tips.
        """
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
                f"{gap['label']} is {gap['gap_size']:.0f}% behind {best_player}. "
                f"{base_tip}"
            )

        if not tips:
            tips.append(
                "Great alignment with your matched pro player! "
                "Keep practising to maintain these levels."
            )

        return tips

    # ------------------------------------------------------------------
    # Radar chart data
    # ------------------------------------------------------------------

    def _build_radar_data(
        self,
        user_metrics: Dict[str, float],
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build normalised radar chart data (0-100 scale) for user vs pro.
        """
        # Normalise each metric to 0-100
        radar_metrics = [
            {
                "label": "Bat Speed",
                "key": "bat_speed",
                "user_normalised": self._normalise_to_100(
                    user_metrics["bat_speed_peak_kmh"], 0, 160, higher_better=True
                ),
                "pro_normalised": self._normalise_to_100(
                    profile["bat_speed_peak_kmh"], 0, 160, higher_better=True
                ),
            },
            {
                "label": "Head Stability",
                "key": "head_stability",
                "user_normalised": self._normalise_to_100(
                    user_metrics["head_stability_score"], 0, 100, higher_better=True
                ),
                "pro_normalised": self._normalise_to_100(
                    profile["head_stability_score"], 0, 100, higher_better=True
                ),
            },
            {
                "label": "Knee Bend",
                "key": "front_knee",
                "user_normalised": self._normalise_to_100(
                    user_metrics["front_knee_angle"], 120, 180, higher_better=False
                ),
                "pro_normalised": self._normalise_to_100(
                    profile["front_knee_angle"], 120, 180, higher_better=False
                ),
            },
            {
                "label": "Posture",
                "key": "spine",
                "user_normalised": self._normalise_to_100(
                    user_metrics["spine_angle"], 0, 35, higher_better=False
                ),
                "pro_normalised": self._normalise_to_100(
                    profile["spine_angle"], 0, 35, higher_better=False
                ),
            },
            {
                "label": "Elbow Control",
                "key": "front_elbow",
                "user_normalised": self._normalise_to_100(
                    user_metrics["front_elbow_angle"], 140, 180, higher_better=False
                ),
                "pro_normalised": self._normalise_to_100(
                    profile["front_elbow_angle"], 140, 180, higher_better=False
                ),
            },
        ]

        return {
            "labels": [m["label"] for m in radar_metrics],
            "user_values": [m["user_normalised"] for m in radar_metrics],
            "pro_values": [m["pro_normalised"] for m in radar_metrics],
            "max_value": 100,
        }

    @staticmethod
    def _normalise_to_100(
        value: float,
        min_val: float,
        max_val: float,
        higher_better: bool = True,
    ) -> float:
        """
        Normalise a metric value to 0-100 scale.
        For higher_is_better: value at max_val = 100, at min_val = 0
        For lower_is_better: value at min_val = 100, at max_val = 0
        """
        val_range = max_val - min_val
        if val_range <= 0:
            return 50.0

        if higher_better:
            clamped = max(min_val, min(max_val, value))
            return round((clamped - min_val) / val_range * 100, 0)
        else:
            clamped = max(min_val, min(max_val, value))
            return round((max_val - clamped) / val_range * 100, 0)

    # ------------------------------------------------------------------
    # Batch comparison (multiple sessions)
    # ------------------------------------------------------------------

    def compare_multiple(
        self,
        session_datas: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple sessions and return results.
        Useful for progress tracking over time.
        """
        results = []
        for session_data in session_datas:
            result = self.compare(session_data)
            session_id = session_data.get("session_id", "")
            timestamp = session_data.get("analysis_timestamp", "")
            results.append({
                "session_id": session_id,
                "timestamp": timestamp,
                "comparison": result,
            })
        return results

    # ------------------------------------------------------------------
    # Static utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def get_legal_disclaimer() -> str:
        """Return the full legal disclaimer text."""
        return LEGAL_DISCLAIMER

    @staticmethod
    def list_pro_players(level: Optional[str] = None) -> List[Dict[str, str]]:
        """
        List all available pro players, optionally filtered by level.
        """
        players = []
        for name, data in PRO_PLAYER_DATABASE.items():
            if level and data["level"] != level:
                continue
            players.append({
                "name": name,
                "level": data["level"],
                "style": data["style"],
            })
        return players

    @staticmethod
    def get_player_profile(player_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific player's profile."""
        return PRO_PLAYER_DATABASE.get(player_name)
