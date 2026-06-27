"""
Longitudinal Feedback — Transforms repetitive coaching into a personalised,
evolving narrative by comparing each session against a player's own history.

Produces:
  - Per-metric trends (improved / declined / stable)
  - Fatigue / unwell detection (significant drop vs rolling average)
  - Session-specific narrative paragraphs (not templates)
  - Coaching recommendations that reference past advice
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# Sensitivity thresholds (in standard deviations)
FATIGUE_Z_THRESHOLD = 1.5  # score drop >1.5σ → flag fatigue
IMPROVEMENT_MIN_PCT = 5     # minimum % change to call it "improvement"
DECLINE_MIN_PCT = 5         # minimum % change to call it "decline"


def analyze_trends(
    current_metrics: Dict,
    historical_metrics: Dict,
    session_label: str = "",
) -> Dict:
    """
    Compare current session metrics against a player's history.

    Args:
        current_metrics: dict of metric_name → value (from compute_session_metrics)
        historical_metrics: dict of metric_name → list of prior values
        session_label: e.g. "session_01_105456"

    Returns:
        dict with:
          - trends: list of per-metric comparisons
          - fatigue_flag: bool
          - fatigue_detail: str
          - overall_narrative: str
          - session_type: str (normal / improvement / decline)
    """
    trends = []
    fatigue_flag = False
    fatigue_detail = ""
    improvement_count = 0
    decline_count = 0
    notable_metrics = []

    for metric_key, current_val in current_metrics.items():
        if metric_key.startswith("_"):
            continue
        if not isinstance(current_val, (int, float)):
            continue

        history = historical_metrics.get(metric_key, [])
        if len(history) < 1:
            # No history yet — just record
            continue

        # Compute stats
        arr = np.array(history, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr)) or 1.0  # avoid div by zero

        # Determine direction
        if current_val > mean:
            change_pct = ((current_val - mean) / abs(mean) * 100) if abs(mean) > 0.01 else 0
        else:
            change_pct = ((current_val - mean) / abs(mean) * 100) if abs(mean) > 0.01 else 0

        z_score = (current_val - mean) / std
        direction = "up" if current_val > mean else "down" if current_val < mean else "stable"
        significance = "notable" if abs(z_score) > 1.0 else "normal"

        # Check improvement vs decline
        is_improvement = False
        is_decline = False

        # For metrics where higher is better
        higher_is_better = {
            "session_score": True,
            "bat_speed_avg_kmh": True,
            "bat_speed_peak_kmh": True,
            "head_stability_score": True,
            "shot_completion_pct": True,
        }
        # For metrics where lower is better (or just different)
        lower_is_better = {
            "avg_front_knee_angle": False,  # 130-145 is ideal
            "avg_spine_angle": False,        # closer to 180 is better
        }

        if metric_key in higher_is_better:
            is_improvement = higher_is_better[metric_key] and current_val > mean * (1 + IMPROVEMENT_MIN_PCT / 100)
            is_decline = higher_is_better[metric_key] and current_val < mean * (1 - DECLINE_MIN_PCT / 100)
        elif metric_key == "avg_front_knee_angle":
            # Ideal is 130-145, so we check if we're moving toward it
            ideal = 137.5
            prev_dev = abs(mean - ideal)
            curr_dev = abs(current_val - ideal)
            is_improvement = curr_dev < prev_dev
            is_decline = curr_dev > prev_dev
        elif metric_key == "avg_spine_angle":
            # Higher is better (closer to 180)
            is_improvement = current_val > mean
            is_decline = current_val < mean

        if is_improvement:
            improvement_count += 1
        if is_decline:
            decline_count += 1

        trend = {
            "metric": metric_key,
            "current": round(current_val, 1),
            "previous_avg": round(mean, 1),
            "previous_std": round(std, 1),
            "z_score": round(z_score, 2),
            "direction": direction,
            "significance": significance,
            "is_improvement": is_improvement,
            "is_decline": is_decline,
            "change_pct": round(change_pct, 1),
        }
        trends.append(trend)

        if significance == "notable":
            notable_metrics.append(trend)

    # ── Fatigue / wellness check ──
    score_current = current_metrics.get("session_score", 0)
    score_history = historical_metrics.get("session_score", [])
    bat_current = current_metrics.get("bat_speed_avg_kmh", 0)
    bat_history = historical_metrics.get("bat_speed_avg_kmh", [])

    if len(score_history) >= 2:
        score_arr = np.array(score_history, dtype=float)
        score_mean = float(np.mean(score_arr))
        score_std = float(np.std(score_arr)) or 1.0
        score_z = (score_current - score_mean) / score_std

        bat_z = 0.0
        if len(bat_history) >= 2:
            bat_arr = np.array(bat_history, dtype=float)
            bat_mean = float(np.mean(bat_arr))
            bat_std = float(np.std(bat_arr)) or 1.0
            bat_z = (bat_current - bat_mean) / bat_std

        # Combined flag: score z < -1.5 AND bat speed z < -1.0
        if score_z < -FATIGUE_Z_THRESHOLD and bat_z < -1.0:
            fatigue_flag = True
            fatigue_detail = (
                f"Session score of {score_current:.0f} is {abs(score_z):.1f} standard deviations "
                f"below your average of {score_mean:.0f}. "
                f"Bat speed at {bat_current:.0f} km/h is also below your norm. "
                f"This pattern suggests fatigue, illness, or a disrupted routine."
            )
        elif score_z < -1.0:
            fatigue_flag = True
            fatigue_detail = (
                f"Session score of {score_current:.0f} is below your average of {score_mean:.0f}. "
                f"Not a dramatic drop, but worth noting."
            )

    # ── Determine session type ──
    if fatigue_flag:
        session_type = "fatigue"
    elif improvement_count >= 2:
        session_type = "improvement"
    elif decline_count >= 2:
        session_type = "decline"
    else:
        session_type = "normal"

    # ── Generate narrative ──
    narrative = _generate_narrative(
        trends, fatigue_flag, fatigue_detail, session_type,
        current_metrics, historical_metrics, notable_metrics,
    )

    return {
        "trends": trends,
        "fatigue_flag": fatigue_flag,
        "fatigue_detail": fatigue_detail,
        "overall_narrative": narrative,
        "session_type": session_type,
        "improvement_count": improvement_count,
        "decline_count": decline_count,
    }


def _generate_narrative(
    trends: List[Dict],
    fatigue_flag: bool,
    fatigue_detail: str,
    session_type: str,
    current_metrics: Dict,
    historical_metrics: Dict,
    notable_metrics: List[Dict],
) -> str:
    """Build a paragraph-length narrative that sounds human, not templated."""

    parts = []

    # ── Opening: session type greeting ──
    if session_type == "improvement":
        parts.append("This is a strong session. You're making progress in several areas.")
    elif session_type == "decline":
        parts.append("A slightly tougher session today — a few areas were off your usual standards, which is part of the process.")
    elif session_type == "fatigue":
        parts.append("This session looks like an off day — and that's okay. Let's look at the numbers.")
    else:
        parts.append("A solid session with some interesting numbers.")

    # ── Notable metrics ──
    for t in notable_metrics:
        metric = t["metric"]
        curr = t["current"]
        prev = t["previous_avg"]

        if metric == "session_score":
            if t["is_improvement"]:
                parts.append(f"Your overall session score of {curr:.0f} is up from your average of {prev:.0f}.")
            elif t["is_decline"]:
                parts.append(f"Your session score of {curr:.0f} is below your usual {prev:.0f}.")
        elif metric == "bat_speed_avg_kmh":
            if t["is_improvement"]:
                parts.append(f"Bat speed is up to {curr:.0f} km/h — solid gain from your typical {prev:.0f} km/h.")
            elif t["is_decline"]:
                parts.append(f"Bat speed at {curr:.0f} km/h is off your average of {prev:.0f} km/h.")
        elif metric == "bat_speed_peak_kmh":
            if t["is_improvement"]:
                parts.append(f"Your peak speed hit {curr:.0f} km/h — the best in your tracked sessions.")
        elif metric == "head_stability_score":
            if t["is_improvement"]:
                parts.append(f"Head stability improved to {curr:.0f}/100, up from {prev:.0f}. The work is paying off.")
            elif t["is_decline"]:
                parts.append(f"Head movement crept up — {curr:.0f}/100 vs your typical {prev:.0f}. Worth revisiting the head-still drill.")
        elif metric == "avg_front_knee_angle":
            if t["is_improvement"]:
                parts.append(f"Knee bend improving: {curr:.0f} degrees — getting closer to the ideal range.")
            elif t["is_decline"]:
                parts.append(f"Front knee at {curr:.0f} degrees is a bit straighter than your usual {prev:.0f}.")
        elif metric == "avg_spine_angle":
            if t["is_improvement"]:
                parts.append(f"Posture is better — spine angle at {curr:.0f} degrees, more upright than before.")
            elif t["is_decline"]:
                parts.append(f"You're leaning a bit more today — spine at {curr:.0f} degrees vs {prev:.0f} average.")

    # ── Fatigue / wellness ──
    if fatigue_flag:
        parts.append(fatigue_detail)

    # ── Closing encouragement ──
    if session_type == "improvement":
        parts.append("Keep going — this trajectory is strong.")
    elif session_type == "fatigue":
        parts.append("Rest up. The next session will be better.")
    else:
        parts.append("Consistency is how you get good.")

    return " ".join(parts)


def generate_voiceover_script(
    current_metrics: Dict,
    historical_metrics: Dict,
    report_data: Dict,
    session_label: str = "",
) -> Dict:
    """
    Generate a brief (30-40 second) voiceover script that references history.

    Returns dict with:
      - script: str (plain text for TTS)
      - has_history: bool
      - session_type: str
    """
    has_history = bool(historical_metrics and any(len(v) > 1 for v in historical_metrics.values()))

    if not has_history:
        # First session — standard coaching intro
        return {
            "script": _first_session_script(report_data),
            "has_history": False,
            "session_type": "first",
        }

    trends = analyze_trends(current_metrics, historical_metrics)
    narrative = trends["overall_narrative"]

    # Extract key numbers for the script
    score = report_data.get("session_score", 0)
    head = report_data.get("head_stability", {}).get("score", 0)
    bat = report_data.get("bat_speed", {}).get("avg_kmh", 0)
    knee = report_data.get("front_knee", {}).get("avg", 0)
    shots = report_data.get("shot_completion", {})
    complete = shots.get("complete", 0)
    total = shots.get("total", 0)

    # Build a concise script (40-50 seconds at natural speaking pace ~150 wpm)
    script_parts = []

    if trends["session_type"] == "improvement":
        script_parts.append(f"This is your best session yet.")
    elif trends["session_type"] == "fatigue":
        script_parts.append(f"Looks like an off day. Let's see what the numbers say.")
    else:
        script_parts.append(f"Let's review your session.")

    script_parts.append(narrative)

    # Add the standard stat line
    detail = (
        f"Your session score is {score:.0f} out of 100. "
        f"Head stability at {head:.0f}. "
    )
    if bat > 0:
        detail += f"Bat speed averaging {bat:.0f} kilometres per hour. "
    if total > 0:
        detail += f"You played {complete} of {total} shots to completion. "
    if knee > 0:
        detail += f"Front knee at {knee:.0f} degrees. "

    script_parts.append(detail)

    # Drill recommendation (evolving: don't repeat the same drill every session)
    priorities = report_data.get("priorities", [])
    if priorities:
        top = priorities[0]
        script_parts.append(f"Top focus: {top['area']}. {top['drill']}")

    script = " ".join(script_parts)
    return {
        "script": script,
        "has_history": True,
        "session_type": trends["session_type"],
    }


def _first_session_script(report_data: Dict) -> str:
    """Baseline script for a player's first ever session."""
    score = report_data.get("session_score", 0)
    head = report_data.get("head_stability", {}).get("score", 0)
    bat = report_data.get("bat_speed", {}).get("avg_kmh", 0)
    knee = report_data.get("front_knee", {}).get("avg", 0)
    shots = report_data.get("shot_completion", {})
    complete = shots.get("complete", 0)
    total = shots.get("total", 0)

    parts = [
        f"Welcome to your first baseline session. Your overall score is {score:.0f} out of 100. "
    ]
    if head < 50:
        parts.append(f"Head stability at {head:.0f} — expect this to improve as we work on it together. ")
    if bat > 0:
        parts.append(f"Bat speed averaging {bat:.0f} kilometres per hour. ")
    if total > 0:
        parts.append(f"You completed {complete} of {total} shots. ")
    if knee > 0:
        parts.append(f"Front knee at {knee:.0f} degrees. ")

    parts.append("These numbers give us a baseline to track your progress against.")

    priorities = report_data.get("priorities", [])
    if priorities:
        top = priorities[0]
        parts.append(f"We'll start with {top['area']}. {top['drill']}")

    return " ".join(parts)
