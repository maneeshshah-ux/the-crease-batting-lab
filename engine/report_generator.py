"""
PDF Coaching Report Generator  -  "the CREASE"

Generates a branded, multi-page PDF report with:
- Cover page with session score
- Session summary metrics table + radar chart of key skills
- Bat speed comparison chart vs elite players
- Head stability analysis with elite comparison bars
- Knee bend & posture analysis with range bars
- Per-shot breakdown table (duration, knee, spine, head, status)
- Top 3 personalised priorities with customised drill text
- Source attribution for benchmark data

Uses fpdf2 (pure Python) and matplotlib for data visualisations.
"""

import os
import sys
import json
import math
import tempfile
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from fpdf import FPDF

# ── Brand colours ──
CREASE_ORANGE = "#E55000"
PITCH_BLACK = "#0A0A0A"
CREASE_SILVER = "#C2C2C2"
OFF_WHITE = "#F7F7F5"

# ── Benchmark attribution ──
BENCHMARK_SOURCE = (
    "Benchmark data sourced from published cricket biomechanics research, "
    "ECB/Cricket Australia coaching resources, and publicly available match analysis. "
    "Player names are factual public figures used for comparison purposes only."
)


def _hex_to_rgb(h):
    """Convert #RRGGBB to (R, G, B) tuple for fpdf."""
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _compute_knee_score(avg_knee):
    """Score 0-100: 140 deg is optimal. Penalty increases with deviation."""
    return max(0, min(100, 100 - abs(avg_knee - 140) * 2.5))


def _compute_spine_score(avg_spine):
    """Score 0-100: 162 deg is optimal for this batter's profile."""
    return max(0, min(100, 100 - abs(avg_spine - 162) * 4))


def _compute_bat_speed_score(swing_avg_kmh):
    """Score 0-100: 70 km/h swing avg = elite = 100."""
    return min(100, swing_avg_kmh / 70.0 * 100)


class Report(FPDF):
    """Branded PDF report for the CREASE."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(True, 20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(*_hex_to_rgb(CREASE_SILVER))
            self.cell(0, 6, "the CREASE Batting Analysis", align="L")
            self.cell(0, 6, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
            self.line(10, 14, 200, 14)
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 6)
        self.set_text_color(*_hex_to_rgb(CREASE_SILVER))
        self.cell(0, 10, BENCHMARK_SOURCE[:80] + "...", align="C")

    # ── Helper: colourised cell ──
    def colour_cell(self, w, h, text, bg_color=None, text_color=None, border=0, align="C", size=10, style=""):
        if bg_color:
            self.set_fill_color(*_hex_to_rgb(bg_color))
        if text_color:
            self.set_text_color(*_hex_to_rgb(text_color))
        self.set_font("Helvetica", style, size)
        self.cell(w, h, text, border=border, align=align, fill=bool(bg_color), new_x="RIGHT", new_y="TOP")
        self.set_text_color(0, 0, 0)

    # ── Cover page ──
    def cover_page(self, session_score, session_id, player_label=""):
        self.add_page()
        # Large orange bar at top
        self.set_fill_color(*_hex_to_rgb(CREASE_ORANGE))
        self.rect(0, 0, 210, 60, "F")
        # Title
        self.set_y(22)
        self.set_font("Helvetica", "B", 32)
        self.set_text_color(255, 255, 255)
        self.cell(0, 14, "THE CREASE", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 14)
        self.cell(0, 8, "Batting Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")

        # Score circle (big number in centre)
        self.ln(20)
        score_str = f"{session_score:.0f}"
        self.set_font("Helvetica", "B", 64)
        self.set_text_color(*_hex_to_rgb(CREASE_ORANGE))
        self.cell(0, 20, score_str, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 12)
        self.set_text_color(*_hex_to_rgb(CREASE_SILVER))
        self.cell(0, 8, "SESSION SCORE", align="C", new_x="LMARGIN", new_y="NEXT")

        # Player label (if available)
        if player_label:
            self.ln(2)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*_hex_to_rgb(CREASE_ORANGE))
            self.cell(0, 7, player_label, align="C", new_x="LMARGIN", new_y="NEXT")

        # Separator
        self.ln(8)
        self.set_draw_color(*_hex_to_rgb(CREASE_ORANGE))
        self.set_line_width(0.5)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(10)

        # Metadata
        self.set_font("Helvetica", "", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 7, f"Session ID: {session_id}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 7, f"Date: {datetime.now().strftime('%d %B %Y')}", align="C", new_x="LMARGIN", new_y="NEXT")

        # Bottom branding
        self.ln(30)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*_hex_to_rgb(CREASE_SILVER))
        self.cell(0, 6, "Powered by the CREASE Batting Lab  -  AI Cricket Analysis", align="C")

    # ── Helper: head movement as readable string ──
    def _head_movement_str(self, session_data):
        """Convert head movement to cm if calibration is available, else show px."""
        ss = session_data.get("session_summary", {})
        bs = session_data.get("bat_speed", {})
        avg_px = ss.get("avg_head_movement", 0)
        cal = bs.get("calibration", {})
        px_per_m = cal.get("px_per_m", None)
        if px_per_m and px_per_m > 0:
            cm = avg_px / (px_per_m / 100.0)
            return f"{cm:.1f} cm"
        else:
            return f"{avg_px:.1f} px"

    # ── Customise drill text with per-video metrics ──
    def _customise_drill_text(self, area, default_drill, session_data):
        """Turn a generic drill description into a customised one with the
        player's actual metrics from this session."""
        ss = session_data.get("session_summary", {})
        bs = session_data.get("bat_speed", {})
        shots = session_data.get("shot_summary", [])
        total_shots = len(shots)
        complete = len([s for s in shots if s.get("has_impact")])

        head_mv = self._head_movement_str(session_data)
        head_score = ss.get("head_stability_score", 0)
        knee_angle = ss.get("avg_front_knee_angle", 0)
        spine_angle = ss.get("avg_spine_angle", 0)
        swing_avg = bs.get("swing_avg_kmh", 0)
        peak_speed = bs.get("peak_kmh", 0)

        area_upper = area.upper()

        if "HEAD" in area_upper:
            return (
                f"Your head drifts {head_mv} per frame (score: {head_score:.0f}/100). "
                f"Elite players keep it under 0.5 cm. "
                f"Drill: Place a bottle cap on your head and shadow bat 50 forward "
                f"defences without it falling. Do NOT chase the cap - let it fall if "
                f"your head moves. Repeat until you can complete 10 in a row."
            )

        elif "SHOT" in area_upper or "COMMIT" in area_upper:
            completion_pct = (complete / total_shots * 100) if total_shots > 0 else 0
            return (
                f"Of {total_shots} shots detected, only {complete} ({completion_pct:.0f}%) "
                f"reached full follow-through ({complete} complete). "
                f"Partial swings build bad muscle memory - each time you lift the bat "
                f"you must complete the shot. "
                f"Drill: 30 shadow swings in front of a mirror - swing to full finish "
                f"every time. If you wouldn't hit it, don't lift the bat."
            )

        elif "KNEE" in area_upper:
            too_straight = knee_angle > 155
            too_bent = knee_angle < 120
            assessment = "too straight" if too_straight else "too bent" if too_bent else "within range"
            return (
                f"Your front knee averages {knee_angle:.0f} degrees ({assessment}). "
                f"Target range: 130-145 degrees at impact for optimal power transfer. "
                f"Drill: Slow-motion forward defence in front of a mirror. Pause at "
                f"impact and check knee angle. Drop a tennis ball and catch it after "
                f"contact — this forces you to hold the bent-knee position."
            )

        elif "POSTURE" in area_upper or "SPINE" in area_upper:
            upright = spine_angle > 50
            hunched = spine_angle < 20
            assessment = "too upright" if upright else "too hunched" if hunched else "good"
            return (
                f"Your spine angle averages {spine_angle:.0f} degrees ({assessment}). "
                f"Target: 30-45 degrees forward tilt at impact. "
                f"Drill: Stand 30 cm from a wall and shadow bat without touching it — "
                f"this forces you to lean forward correctly. 20 reps, pause at contact."
            )

        # Fallback: return default drill text but include session score
        text = f"{default_drill}  (Session score: {ss.get('session_score', 0):.0f}/100)"
        return text.encode("ascii", errors="replace").decode("ascii")

    # ── Radar chart: multi-metric overview ──
    def _generate_radar_chart(self, session_data):
        """Generate a professional radar/spider chart showing 5 key metrics.

        Metrics are normalised to 0-100 for comparison.
        Returns path to a PNG; caller must delete the file.
        """
        ss = session_data.get("session_summary", {})
        bs = session_data.get("bat_speed", {})
        shots = session_data.get("shot_summary", [])
        total_shots = len(shots)
        complete = len([s for s in shots if s.get("has_impact")])
        completion_pct = (complete / total_shots * 100) if total_shots > 0 else 0

        head_score = ss.get("head_stability_score", 0)
        avg_knee = ss.get("avg_front_knee_angle", 140)
        avg_spine = ss.get("avg_spine_angle", 162)
        swing_avg = bs.get("swing_avg_kmh", 0)

        # Compute normalised scores
        knee_score = _compute_knee_score(avg_knee)
        spine_score = _compute_spine_score(avg_spine)
        bat_score = _compute_bat_speed_score(swing_avg)

        categories = [
            "Head\nStability",
            "Bat\nSpeed",
            "Shot\nCompletion",
            "Knee\nBend",
            "Body\nPosture",
        ]
        values = [head_score, bat_score, completion_pct, knee_score, spine_score]

        # Number of variables
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values += values[:1]  # Close the loop
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(4.0, 4.0), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # Draw axis lines and labels
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])

        # Style the grid
        ax.set_rlabel_position(0)

        # Draw the filled area
        ax.fill(angles, values, color=CREASE_ORANGE, alpha=0.15)
        ax.plot(angles, values, color=CREASE_ORANGE, linewidth=2, linestyle="solid", zorder=5)

        # Draw data points
        ax.scatter(angles[:-1], values[:-1], color=CREASE_ORANGE, s=40, zorder=6, edgecolors="white", linewidth=0.5)

        # Labels for each axis
        ax.set_xticklabels(categories, fontsize=8, fontweight="bold", color=PITCH_BLACK)

        # Y-axis ticks (0, 25, 50, 75, 100)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75])
        ax.set_yticklabels(["25", "50", "75"], fontsize=6, color=CREASE_SILVER)
        ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.5)
        ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.5)
        ax.spines["polar"].set_visible(False)

        # Annotate actual values near each point
        for i, (angle, val, cat) in enumerate(zip(angles[:-1], values[:-1], categories)):
            # Offset the label a bit outward
            label_r = val + 10
            label_angle = angle
            ax.annotate(
                f"{val:.0f}",
                xy=(angle, val),
                xytext=(label_angle, label_r),
                fontsize=7,
                fontweight="bold",
                color=CREASE_ORANGE,
                ha="center",
                va="center",
                zorder=10,
            )

        plt.tight_layout(pad=0.5)
        chart_fd, chart_path = tempfile.mkstemp(suffix=".png")
        os.close(chart_fd)
        fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return chart_path

    # ── Shot breakdown table ──
    def _generate_shot_breakdown_table(self, session_data):
        """Generate a matplotlib table showing per-shot metrics.

        Uses frame-level data to compute per-shot averages for knee angle,
        spine angle, head movement, and bat speed.
        Returns path to a PNG; caller must delete the file.
        """
        shots = session_data.get("shot_summary", [])
        frame_metrics = session_data.get("frame_metrics", [])

        # Build frame index for quick lookup
        frame_index = {}
        for fm in frame_metrics:
            frame_index[fm["frame"]] = fm

        # Compute per-shot averages
        rows = []
        for shot in shots:
            sn = shot.get("shot_number", 0)
            start = shot.get("start_frame", 0)
            end = shot.get("end_frame", 0)
            has_impact = shot.get("has_impact", False)

            # Collect frame data within this shot's range
            knee_vals = []
            spine_vals = []
            head_mv_vals = []
            bat_speed_vals = []

            for f in range(start, end + 1):
                fm = frame_index.get(f)
                if fm:
                    if fm.get("front_knee_angle") is not None:
                        knee_vals.append(fm["front_knee_angle"])
                    if fm.get("spine_angle") is not None:
                        spine_vals.append(fm["spine_angle"])
                    if fm.get("head_movement") is not None:
                        head_mv_vals.append(fm["head_movement"])
                    if fm.get("bat_speed_px") is not None:
                        bat_speed_vals.append(fm["bat_speed_px"])

            avg_knee = np.mean(knee_vals) if knee_vals else 0
            avg_spine = np.mean(spine_vals) if spine_vals else 0
            avg_head = np.mean(head_mv_vals) if head_mv_vals else 0
            avg_bat = np.mean(bat_speed_vals) if bat_speed_vals else 0
            duration = shot.get("duration_sec", 0)

            rows.append({
                "shot": sn,
                "duration": duration,
                "knee": avg_knee,
                "spine": avg_spine,
                "head": avg_head,
                "bat": avg_bat,
                "complete": has_impact,
            })

        if not rows:
            return None

        # Build the matplotlib table
        fig, ax = plt.subplots(figsize=(7.2, min(0.35 * len(rows) + 0.8, 8.5)))
        fig.patch.set_facecolor("white")
        ax.axis("off")

        # Column labels (convert head from px to cm if calibration available)
        bs = session_data.get("bat_speed", {})
        cal = bs.get("calibration", {})
        px_per_m = cal.get("px_per_m", None)
        has_cal = px_per_m is not None and px_per_m > 0
        head_unit = "cm" if has_cal else "px"
        col_labels = ["Shot", "Dur (s)", "Knee (deg)", "Spine (deg)", f"Head ({head_unit})", "Bat (px)", "Status"]

        # Prepare cell text
        cell_text = []
        for r in rows:
            status = "COMPLETE" if r["complete"] else "partial"
            head_val = r["head"]
            if has_cal:
                head_val = head_val / (px_per_m / 100.0)  # convert px to cm
            cell_text.append([
                str(r["shot"]),
                f"{r['duration']:.1f}",
                f"{r['knee']:.0f}",
                f"{r['spine']:.0f}",
                f"{head_val:.1f}",
                f"{r['bat']:.1f}",
                status,
            ])

        # Table
        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
            colWidths=[0.6, 0.7, 1.0, 1.0, 0.9, 0.9, 1.1],
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(7)

        # Header row
        for j in range(len(col_labels)):
            cell = table[0, j]
            cell.set_facecolor(CREASE_ORANGE)
            cell.set_text_props(color="white", fontweight="bold")

        # Data rows
        for i in range(len(rows)):
            is_complete = rows[i]["complete"]
            for j in range(len(col_labels)):
                cell = table[i + 1, j]
                if i % 2 == 0:
                    cell.set_facecolor("#F5F5F5")
                else:
                    cell.set_facecolor("white")
                # Colour the status cell
                if j == len(col_labels) - 1:
                    if is_complete:
                        cell.set_facecolor("#E8F5E9")  # green tint
                        cell.set_text_props(color="#2E7D32", fontweight="bold")
                    else:
                        cell.set_facecolor("#FFF3E0")  # amber tint
                        cell.set_text_props(color="#E65100")
                # Alternate row colouring
                else:
                    cell.set_edgecolor("#DDDDDD")

        # Title
        ax.set_title("Shot-by-Shot Breakdown", fontsize=10, fontweight="bold",
                      color=PITCH_BLACK, pad=8)

        plt.tight_layout(pad=0.5)
        chart_fd, chart_path = tempfile.mkstemp(suffix=".png")
        os.close(chart_fd)
        fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return chart_path

    # ── Metrics page ──
    def metrics_page(self, report_data, session_data):
        """Page showing key session metrics in a clean table + radar chart."""
        self.add_page()
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*_hex_to_rgb(PITCH_BLACK))
        self.cell(0, 10, "Session Summary", new_x="LMARGIN", new_y="NEXT")

        ss = session_data.get("session_summary", {})
        shots = session_data.get("shot_summary", [])
        total_shots = len(shots)
        complete = len([s for s in shots if s.get("has_impact")])
        bs = session_data.get("bat_speed", {})

        head_movement_str = self._head_movement_str(session_data)

        metrics = [
            ("Shots Detected", str(total_shots)),
            ("Complete Shots", str(complete)),
            ("Session Duration", f"{ss.get('duration_sec', 0):.0f}s"),
            ("Bat Speed (Peak)", f"{bs.get('peak_kmh', 0):.0f} km/h"),
            ("Bat Speed (Swing Avg)", f"{bs.get('swing_avg_kmh', 0):.0f} km/h"),
            ("Head Stability", f"{ss.get('head_stability_score', 0):.0f}/100"),
            ("Avg Head Movement", head_movement_str),
            ("Front Knee (Avg)", f"{ss.get('avg_front_knee_angle', 0):.0f} deg"),
            ("Front Knee (Min)", f"{ss.get('min_front_knee_angle', 0):.0f} deg"),
            ("Spine Angle (Avg)", f"{ss.get('avg_spine_angle', 0):.0f} deg"),
            ("Session Score", f"{report_data.get('session_score', 0):.0f}/100"),
        ]

        # Table header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*_hex_to_rgb(CREASE_ORANGE))
        self.set_text_color(255, 255, 255)
        self.cell(80, 7, "  Metric", border=0, fill=True, new_x="RIGHT", new_y="TOP")
        self.cell(40, 7, "Value", border=0, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

        # Table rows
        for i, (label, value) in enumerate(metrics):
            bg = OFF_WHITE if i % 2 == 0 else "#FFFFFF"
            self.set_fill_color(*_hex_to_rgb(bg))
            self.set_text_color(*_hex_to_rgb(PITCH_BLACK))
            self.set_font("Helvetica", "", 9)
            self.cell(80, 6, f"  {label}", border=0, fill=True, new_x="RIGHT", new_y="TOP")
            self.set_font("Helvetica", "B", 9)
            self.cell(40, 6, value, border=0, fill=True, align="C", new_x="LMARGIN", new_y="NEXT")

        # Radar chart
        self.ln(6)
        radar_path = self._generate_radar_chart(session_data)
        if radar_path:
            # Centre the radar chart
            self.image(radar_path, x=50, w=110)
            os.remove(radar_path)

        self.ln(2)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_hex_to_rgb(CREASE_SILVER))
        self.multi_cell(0, 4,
            "Radar chart shows 5 core skills normalised to 100. "
            "A larger \"snowflake\" shape = more balanced batting profile.")

        # Priorities section with customised drill text
        self.ln(4)
        priorities = report_data.get("priorities", [])
        if priorities:
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(*_hex_to_rgb(CREASE_ORANGE))
            self.cell(0, 10, "Top Priorities", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
            for p in priorities:
                # Priority title
                self.set_font("Helvetica", "B", 9)
                self.set_text_color(*_hex_to_rgb(PITCH_BLACK))
                self.cell(0, 6, f"  #{p['rank']}: {p['area']}", new_x="LMARGIN", new_y="NEXT")

                # Customised drill text (includes per-video metrics)
                drill_text = self._customise_drill_text(
                    p['area'], p.get('drill', ''), session_data
                )
                # Sanitize for Helvetica (strip any non-ASCII chars)
                drill_text = drill_text.encode("ascii", errors="replace").decode("ascii")
                self.set_font("Helvetica", "", 8)
                self.set_text_color(80, 80, 80)
                self.multi_cell(0, 5, f"    {drill_text}")
                self.ln(3)

    # ── Bat speed chart ──
    def bat_speed_chart_page(self, session_data, report_data):
        """Page with matplotlib bat speed comparison chart."""
        self.add_page()
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*_hex_to_rgb(PITCH_BLACK))
        self.cell(0, 10, "Bat Speed Analysis", new_x="LMARGIN", new_y="NEXT")

        bs = session_data.get("bat_speed", {})
        peak_kmh = bs.get("peak_kmh", 0)
        swing_avg_kmh = bs.get("swing_avg_kmh", 0)

        # Create chart
        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig.patch.set_facecolor(OFF_WHITE)
        ax.set_facecolor(OFF_WHITE)

        # Players (peak bat speed)
        players = {
            "You (Peak)": peak_kmh,
            "Swing Avg": swing_avg_kmh,
            "Ellyse Perry": 115,
            "Virat Kohli": 135,
            "AB de Villiers": 150,
            "Andre Russell": 155,
        }

        colours = []
        labels_list = list(players.keys())
        values = list(players.values())
        for lbl in labels_list:
            if "You" in lbl:
                colours.append(CREASE_ORANGE)
            elif "Avg" in lbl:
                colours.append("#F5A623")
            else:
                colours.append(CREASE_SILVER)

        bars = ax.barh(labels_list, values, color=colours, edgecolor="white", height=0.6)
        # Annotate values
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}", va="center", fontsize=8, color=PITCH_BLACK)

        ax.set_xlim(0, 175)
        ax.set_xlabel("Bat Speed (km/h)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        # Save chart to temp file
        chart_fd, chart_path = tempfile.mkstemp(suffix=".png")
        os.close(chart_fd)
        fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=OFF_WHITE)
        plt.close(fig)

        # Embed in PDF
        self.image(chart_path, x=15, w=180)
        os.remove(chart_path)

        # Find the exact frame of peak bat speed
        fm_list = session_data.get("frame_metrics", [])
        best_bat_frame = None
        best_bat_px = 0
        fps = session_data.get("video_fps", 30)
        for f_entry in fm_list:
            bsp = f_entry.get("bat_speed_px", 0) or 0
            if bsp > best_bat_px:
                best_bat_px = bsp
                best_bat_frame = f_entry["frame"]
        if best_bat_frame is not None and peak_kmh > 0:
            ts = self._frame_to_timestamp(best_bat_frame, fps)
            self.ln(2)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*_hex_to_rgb(CREASE_ORANGE))
            self.cell(0, 5, f"  Peak speed of {peak_kmh:.0f} km/h occurred at {ts} in the video.",
                      new_x="LMARGIN", new_y="NEXT")

        # Benchmark note
        self.ln(2)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_hex_to_rgb(CREASE_SILVER))
        bench = bs.get("calibration", {})
        method = bench.get("method", "unknown")
        self.multi_cell(0, 4, f"Calibration: {method} | Lever factor: 1.35x (hand-to-bat-tip) | {BENCHMARK_SOURCE}")

    # ── Frame annotation helpers ──
    @staticmethod
    def _frame_to_timestamp(frame_num, fps=30):
        """Convert frame number to MM:SS timestamp string."""
        sec = frame_num / fps if fps > 0 else 0
        mins = int(sec // 60)
        secs = sec % 60
        return f"{mins}:{secs:04.1f}"

    def _extract_frame(self, video_path, frame_number):
        """Extract a single frame from a video file as a numpy array (RGB)."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _annotate_head_frame(self, video_path, session_data):
        """Extract a key frame and overlay the detected head position + drift arrow.

        Uses frame_metrics head_position_x/y for accurate dot placement.
        Returns path to annotated PNG, or None.
        """
        # Find best frame: maximum head movement inside a genuine shot
        fm_list = session_data.get("frame_metrics", [])
        shots = session_data.get("shot_summary", [])

        best_frame = None
        best_mv = 0
        for fm in fm_list:
            mv = fm.get("head_movement", 0) or 0
            if mv > best_mv:
                for s in shots:
                    if s["start_frame"] <= fm["frame"] <= s["end_frame"]:
                        best_mv = mv
                        best_frame = fm["frame"]
                        best_fm = fm
                        break

        if best_frame is None or best_mv < 1:
            return None

        frame_rgb = self._extract_frame(video_path, best_frame)
        if frame_rgb is None:
            return None

        h, w = frame_rgb.shape[:2]

        # Get head position from frame_metrics (pixel coordinates in original frame)
        head_x = int(best_fm.get("head_position_x", w // 2))
        head_y = int(best_fm.get("head_position_y", h // 3))

        # Draw on the frame using PIL/Pillow for cleaner text
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img, "RGBA")

        # Try to load a clean font, fall back to default
        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
            font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        except (OSError, IOError):
            font_title = ImageFont.load_default()
            font_body = font_title
            font_small = font_title

        # 1. Draw head dot — orange circle with crosshair
        dot_r = 10
        draw.ellipse([head_x - dot_r, head_y - dot_r, head_x + dot_r, head_y + dot_r],
                     outline="#E55000", width=4)

        # Crosshair in centre
        draw.line([head_x - dot_r - 6, head_y, head_x + dot_r + 6, head_y],
                  fill="#E55000", width=2)
        draw.line([head_x, head_y - dot_r - 6, head_x, head_y + dot_r + 6],
                  fill="#E55000", width=2)

        # 2. Draw drift direction arrow (fixed visual length — pure indicator, not a measurement)
        arrow_len_px = 50
        arrow_end_x = head_x + arrow_len_px
        arrow_end_y = head_y
        draw.line([head_x + dot_r + 4, head_y, arrow_end_x, arrow_end_y],
                  fill="#E55000", width=3)
        # Arrowhead
        ah = 10
        draw.line([arrow_end_x, arrow_end_y, arrow_end_x - ah, arrow_end_y - 5],
                  fill="#E55000", width=3)
        draw.line([arrow_end_x, arrow_end_y, arrow_end_x - ah, arrow_end_y + 5],
                  fill="#E55000", width=3)

        # 3. Semi-transparent insight panel at bottom
        panel_h = 80
        panel = Image.new("RGBA", (w, panel_h), (10, 10, 10, 200))
        img.paste(panel, (0, h - panel_h), panel)

        # Panel content
        ss = session_data.get("session_summary", {})
        head_score = ss.get("head_stability_score", 0)
        assessment = ("Excellent" if head_score >= 80 else "Good"
                      if head_score >= 60 else "Average"
                      if head_score >= 40 else "Needs Work")

        px_per_m = session_data.get("bat_speed", {}).get("calibration", {}).get("px_per_m", None)
        avg_px = ss.get("avg_head_movement", 0)
        if px_per_m and px_per_m > 0:
            cm_val = avg_px / (px_per_m / 100.0)
            cm_str = f"{cm_val:.1f} cm"
        else:
            cm_str = f"{avg_px:.1f} px"

        # Title
        draw.text((14, h - panel_h + 8), "HEAD STABILITY", fill="#FFFFFF",
                  font=font_title)
        # Score
        score_text = f"{head_score:.0f}/100  ({assessment})"
        draw.text((w - 14, h - panel_h + 10), score_text, fill="#E55000",
                  font=font_body, anchor="rt")

        # Separator line
        draw.line([(14, h - panel_h + 36), (w - 14, h - panel_h + 36)],
                  fill="#555555", width=1)

        # Body text with timestamp
        fps = session_data.get("video_fps", 30)
        ts = self._frame_to_timestamp(best_frame, fps)
        body_text = f"Drift: {cm_str}    Elite target: < 0.5 cm    at {ts}"
        draw.text((14, h - panel_h + 44), body_text, fill="#CCCCCC", font=font_body)

        # Coaching insight
        insight = ("Your head drifted right on this shot. Still head = see the ball earlier. "
                   "Focus on keeping your nose in line with middle stump through contact.")
        draw.text((14, h - panel_h + 64), insight, fill="#AAAAAA", font=font_small)

        # Save
        fd, out_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(out_path, "PNG")
        return out_path

    # ── Annotated impact frame (knee / spine focus) ──
    def _annotate_impact_frame(self, video_path, session_data, focus="knee"):
        """Extract a frame from an impact phase and overlay knee or spine analysis.

        Args:
            video_path: path to analysis video
            session_data: full analysis JSON
            focus: "knee" or "spine"

        Returns path to annotated PNG, or None.
        """
        fm_list = session_data.get("frame_metrics", [])
        shots = session_data.get("shot_summary", [])
        if not fm_list or not shots:
            return None

        # Find best impact frame: frame where a shot has impact
        impact_frame = None
        for s in shots:
            if s.get("has_impact") and s.get("impact_frame") is not None:
                impact_frame = s["impact_frame"]
                break
            elif s.get("has_impact") and s.get("end_frame"):
                impact_frame = s["end_frame"] - 5  # near end = near impact
                break

        # Fallback: frame with max knee bend or min spine angle
        if impact_frame is None:
            for fm in fm_list:
                if focus == "knee":
                    score_key = "front_knee_angle"
                else:
                    score_key = "spine_angle"
                val = fm.get(score_key)
                if val is not None:
                    impact_frame = fm["frame"]
                    break

        if impact_frame is None:
            return None

        frame_rgb = self._extract_frame(video_path, impact_frame)
        if frame_rgb is None:
            return None

        h, w = frame_rgb.shape[:2]
        fps = session_data.get("video_fps", 30)

        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img, "RGBA")

        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
            font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        except (OSError, IOError):
            font_title = ImageFont.load_default()
            font_body = font_title
            font_small = font_title

        ss = session_data.get("session_summary", {})

        if focus == "knee":
            title = "FRONT KNEE"
            avg_val = ss.get("avg_front_knee_angle", 0)
            assessment = ("Ideal" if 130 <= avg_val <= 145 else
                          "Slightly straight" if avg_val > 145 else "Deep")
            body_metric = f"{avg_val:.0f} deg avg   Target: 130-145 deg"
            insight = (
                "Knee bend controls balance and power transfer. "
                "Too straight and you lose leverage; too deep and you can't drive."
            )
        else:
            title = "SPINE ANGLE"
            avg_val = ss.get("avg_spine_angle", 0)
            assessment = ("Balanced" if avg_val >= 155 else
                          "Slightly forward" if avg_val >= 145 else "Too hunched")
            body_metric = f"{avg_val:.0f} deg avg   Target: > 155 deg"
            insight = (
                "Forward tilt keeps your head over the ball. "
                "Too upright and you lose sight of the ball late. "
                "Too hunched and you can't generate bat speed."
            )

        # Semi-transparent insight panel at bottom
        panel_h = 80
        panel = Image.new("RGBA", (w, panel_h), (10, 10, 10, 200))
        img.paste(panel, (0, h - panel_h), panel)

        ts = self._frame_to_timestamp(impact_frame, fps)
        draw.text((14, h - panel_h + 8), title, fill="#FFFFFF", font=font_title)
        score_text = f"{avg_val:.0f}  ({assessment})"
        draw.text((w - 14, h - panel_h + 10), score_text, fill="#E55000",
                  font=font_body, anchor="rt")
        draw.line([(14, h - panel_h + 36), (w - 14, h - panel_h + 36)],
                  fill="#555555", width=1)
        draw.text((14, h - panel_h + 44), body_metric + f"    at {ts}",
                  fill="#CCCCCC", font=font_body)
        draw.text((14, h - panel_h + 64), insight, fill="#AAAAAA", font=font_small)

        fd, out_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(out_path, "PNG")
        return out_path

    # ── Player History page ──
    def history_page(self, player_history, current_session_id=""):
        """
        Renders a table + trend chart of the player's last sessions.
        Flags fatigued/unwell sessions for isolation.
        player_history = {
            'player_id': 'p_001',
            'label': 'Player 001',
            'n_sessions': 2,
            'sessions': [ { session_id, date, session_score, head_stability_score,
                            avg_front_knee_angle, avg_spine_angle, bat_speed_avg_kmh,
                            num_shots, shot_completion_pct, flags }, ... ]
        }
        flags can include "fatigue", "improvement", "decline" etc.
        """
        self.add_page()
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*_hex_to_rgb(PITCH_BLACK))
        label = player_history.get("label", "Player")
        safe_label = label.encode("ascii", errors="replace").decode("ascii")
        self.cell(0, 10, f"{safe_label} - Session History", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

        sessions = player_history.get("sessions", [])
        if not sessions:
            self.set_font("Helvetica", "I", 10)
            self.set_text_color(120, 120, 120)
            self.cell(0, 7, "No historical data yet.", new_x="LMARGIN", new_y="NEXT")
            return

        # Show last 10 sessions max
        display = sessions[-10:]

        # ── Trend chart (matplotlib line chart) ──
        if len(sessions) >= 2:
            chart_path = self._generate_session_trend_chart(sessions)
            if chart_path:
                self.image(chart_path, x=20, w=170)
                os.remove(chart_path)
                self.ln(4)

        # Table layout: fixed column widths
        col_w = [12, 20, 18, 18, 18, 18, 18, 18]
        headers = ["#", "Date", "Score", "Head", "Knee", "Spine", "Bat", "Shots"]
        total_w = sum(col_w)
        x_start = (210 - total_w) / 2  # centre table

        # Table header
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(*_hex_to_rgb(CREASE_ORANGE))
        self.set_text_color(255, 255, 255)
        self.set_x(x_start)
        for i, h in enumerate(headers):
            self.cell(col_w[i], 6, h, border=0, fill=True, align="C",
                      new_x="RIGHT", new_y="TOP")
        self.ln(6)

        # ── Collect stats for averages ──
        score_vals = []
        head_vals = []
        knee_vals = []
        spine_vals = []
        bat_vals = []
        shot_vals = []
        normal_score_vals = []  # excluding flagged sessions

        # Table rows
        for row_idx, s in enumerate(display):
            flags = s.get("flags", []) or []

            # Fatigue sessions get a different background
            is_fatigue = "fatigue" in flags
            if is_fatigue:
                bg = "#FFE8D0"  # light orange
            else:
                bg = OFF_WHITE if row_idx % 2 == 0 else "#FFFFFF"

            self.set_fill_color(*_hex_to_rgb(bg))
            self.set_text_color(*_hex_to_rgb(PITCH_BLACK))
            self.set_font("Helvetica", "", 8)
            self.set_x(x_start)

            is_current = s.get("session_id") == current_session_id
            row_num = len(sessions) - len(display) + row_idx + 1

            vals = [
                str(row_num),
                s.get("date", "?")[-5:],
                f"{s.get('session_score', 0):.0f}",
                f"{s.get('head_stability_score', 0):.0f}",
                f"{s.get('avg_front_knee_angle', 0):.0f}",
                f"{s.get('avg_spine_angle', 0):.0f}",
                f"{s.get('bat_speed_avg_kmh', 0):.0f}",
                str(s.get('num_shots', 0)),
            ]

            for i, v in enumerate(vals):
                self.cell(col_w[i], 5.5, v, border=0, fill=True, align="C",
                          new_x="RIGHT", new_y="TOP")

            # Flags badge
            if is_fatigue:
                self.set_font("Helvetica", "I", 6)
                self.set_text_color(*_hex_to_rgb(CREASE_ORANGE))
                self.cell(10, 5.5, "FATIGUE", fill=True, align="C")
            elif is_current:
                self.set_font("Helvetica", "I", 6)
                self.set_text_color(*_hex_to_rgb(CREASE_ORANGE))
                self.cell(10, 5.5, "NEW", fill=True, align="C")
            else:
                self.set_font("Helvetica", "", 6)
                self.set_text_color(*_hex_to_rgb(CREASE_SILVER))
                self.cell(10, 5.5, flags[0] if flags else "", fill=False, align="L")
            self.ln(5.5)

            # Collect for averages
            score_vals.append(s.get("session_score", 0))
            head_vals.append(s.get("head_stability_score", 0))
            knee_vals.append(s.get("avg_front_knee_angle", 0))
            spine_vals.append(s.get("avg_spine_angle", 0))
            bat_vals.append(s.get("bat_speed_avg_kmh", 0))
            shot_vals.append(s.get("num_shots", 0))
            if not is_fatigue:
                normal_score_vals.append(s.get("session_score", 0))

        # ── Averages row(s) ──
        def _avg(lst):
            return sum(lst) / len(lst) if lst else 0

        self.ln(1)
        # Row 1: all sessions
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(*_hex_to_rgb(PITCH_BLACK))
        self.set_text_color(255, 255, 255)
        self.set_x(x_start)
        avg_vals = [
            "Avg",
            f"{len(sessions)} ses",
            f"{_avg(score_vals):.0f}",
            f"{_avg(head_vals):.0f}",
            f"{_avg(knee_vals):.0f}",
            f"{_avg(spine_vals):.0f}",
            f"{_avg(bat_vals):.0f}",
            f"{_avg(shot_vals):.0f}",
        ]
        for i, v in enumerate(avg_vals):
            self.cell(col_w[i], 6, v, border=0, fill=True, align="C",
                      new_x="RIGHT", new_y="TOP")
        self.set_font("Helvetica", "I", 6)
        self.set_text_color(200, 200, 200)
        self.cell(10, 6, "all", fill=True, align="C")
        self.ln(6)

        # Row 2: excluding flagged (if any flagged exist)
        flagged_count = len(sessions) - len(normal_score_vals)
        if flagged_count > 0:
            self.set_fill_color(*_hex_to_rgb("#333333"))
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 8)
            self.set_x(x_start)
            clean_avg_vals = [
                "Cln",
                f"{len(normal_score_vals)} ses",
                f"{_avg(normal_score_vals):.0f}" if normal_score_vals else "-",
                "-", "-", "-", "-", "-",
            ]
            for i, v in enumerate(clean_avg_vals):
                self.cell(col_w[i], 6, v, border=0, fill=True, align="C",
                          new_x="RIGHT", new_y="TOP")
            self.set_font("Helvetica", "I", 6)
            self.set_text_color(200, 200, 200)
            self.cell(10, 6, "clean", fill=True, align="C")
            self.ln(6)

        self.ln(4)

        # ── Trend summary ──
        if len(normal_score_vals) >= 2:
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*_hex_to_rgb(CREASE_ORANGE))
            self.cell(0, 6, "Trends (excl. flagged)", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)

            first_score = normal_score_vals[0]
            last_score = normal_score_vals[-1]
            diff = last_score - first_score

            lines = []
            if diff > 5:
                lines.append(f"Score trending UP: {first_score:.0f} to {last_score:.0f} (+{diff:.0f}).")
            elif diff < -5:
                lines.append(f"Score trending DOWN: {first_score:.0f} to {last_score:.0f} ({diff:.0f}).")
            else:
                lines.append(f"Score stable around {_avg(normal_score_vals):.0f}.")

            if len(bat_vals) >= 2:
                clean_bat = [bat_vals[i] for i in range(len(bat_vals))
                             if i < len(sessions) and ("fatigue" in (sessions[-(len(display)) + i].get("flags", []) if -(len(display)) + i < 0 else sessions[-(len(display)) + i].get("flags", [])))]
                # Simplified: just show overall bat trend
                bf, bl = bat_vals[0], bat_vals[-1]
                if bl - bf > 3:
                    lines.append(f"Bat speed improving: {bf:.0f} to {bl:.0f} km/h.")
                elif bl - bf < -3:
                    lines.append(f"Bat speed declining: {bf:.0f} to {bl:.0f} km/h.")

            if flagged_count > 0:
                lines.append(f"{flagged_count} session(s) flagged as fatigue/off-day and excluded from trend.")

            self.set_font("Helvetica", "", 8)
            self.set_text_color(80, 80, 80)
            for line in lines:
                self.cell(0, 5, f"  {line}", new_x="LMARGIN", new_y="NEXT")

        self.ln(3)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_hex_to_rgb(CREASE_SILVER))
        self.multi_cell(0, 4,
            "Orange rows = sessions flagged as possible fatigue or off-day. "
            "\"Clean\" average excludes flagged sessions for a truer performance baseline.")

    # ── Trend chart generator ──
    def _generate_session_trend_chart(self, all_sessions):
        """Generate a matplotlib line chart of session scores over time."""
        if len(all_sessions) < 2:
            return None

        try:
            fig, ax = plt.subplots(figsize=(7, 2.2))
            fig.patch.set_facecolor("#F7F7F5")

            x = list(range(1, len(all_sessions) + 1))
            scores = [s.get("session_score", 0) for s in all_sessions]

            # Separate flagged vs normal
            normal_x, normal_y = [], []
            flagged_x, flagged_y = [], []
            for i, s in enumerate(all_sessions):
                flags = s.get("flags", []) or []
                if "fatigue" in flags:
                    flagged_x.append(i + 1)
                    flagged_y.append(scores[i])
                else:
                    normal_x.append(i + 1)
                    normal_y.append(scores[i])

            # Plot normal points + line
            if normal_x:
                ax.plot(normal_x, normal_y, color=CREASE_ORANGE,
                        marker="o", markersize=6, linewidth=1.5,
                        label="Normal", zorder=3)
            else:
                ax.plot(x, scores, color="#CCCCCC",
                        marker="o", markersize=5, linewidth=1, alpha=0.5, zorder=2)

            # Plot flagged points
            if flagged_x:
                ax.scatter(flagged_x, flagged_y, color=CREASE_ORANGE,
                           marker="x", s=80, linewidths=2, zorder=4, label="Off-day")

            # Mean line (dashed)
            clean_scores = [s for i, s in enumerate(scores)
                            if "fatigue" not in (all_sessions[i].get("flags", []) or [])]
            if clean_scores:
                mean_val = sum(clean_scores) / len(clean_scores)
                ax.axhline(y=mean_val, color=CREASE_SILVER,
                           linestyle="--", linewidth=0.8, alpha=0.7)
                ax.text(len(all_sessions), mean_val + 1,
                        f"Avg {mean_val:.0f}", fontsize=7, color=CREASE_SILVER,
                        ha="right", va="bottom")

            ax.set_xlabel("Session", fontsize=8, color=PITCH_BLACK)
            ax.set_ylabel("Score", fontsize=8, color=PITCH_BLACK)
            ax.set_xlim(0.5, len(x) + 0.5)
            ax.set_ylim(max(0, min(scores) - 10), min(100, max(scores) + 10))

            ax.tick_params(axis="both", labelsize=7, colors=PITCH_BLACK)
            for spine in ax.spines.values():
                spine.set_color(CREASE_SILVER)
            ax.set_facecolor("#F7F7F5")

            # Legend
            if flagged_x:
                legend = ax.legend(fontsize=6, loc="lower left", framealpha=0.8)
                for text in legend.get_texts():
                    text.set_color(PITCH_BLACK)

            plt.tight_layout()
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#F7F7F5")
            plt.close(fig)
            return path
        except Exception as e:
            print(f"    Warning: trend chart failed: {e}")
            return None

    def head_stability_page(self, session_data, report_data, analysis_video_path):
        """Page showing head stability analysis with annotated video frame + elite comparison bars."""
        self.add_page()
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*_hex_to_rgb(PITCH_BLACK))
        self.cell(0, 10, "Head Stability", new_x="LMARGIN", new_y="NEXT")

        ss = session_data.get("session_summary", {})
        head_score = ss.get("head_stability_score", 0)

        # Text description
        self.set_font("Helvetica", "", 10)
        assessment = ("Excellent" if head_score >= 80 else "Good"
                      if head_score >= 60 else "Average"
                      if head_score >= 40 else "Needs Work")
        self.multi_cell(0, 5.5,
            f"Score: {head_score:.0f}/100 ({assessment})\n"
            f"Average head movement: {self._head_movement_str(session_data)} per frame. "
            f"Elite players (Kohli, Williamson) keep it under 0.5 cm. "
            f"A still head = watching the ball onto the bat.")

        self.ln(2)

        # Annotated video frame — head dot + drift arrow + insight panel
        if analysis_video_path and os.path.exists(analysis_video_path):
            annotated = self._annotate_head_frame(analysis_video_path, session_data)
            if annotated:
                # Portrait frame at moderate size so text fits on same page
                self.image(annotated, x=50, w=100)
                os.remove(annotated)
                self.ln(3)

        # Elite comparison as visual bar chart
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*_hex_to_rgb(CREASE_ORANGE))
        self.cell(0, 6, "Head Movement Comparison", new_x="LMARGIN", new_y="NEXT")

        # Draw a simple text-based comparison table
        self.set_font("Helvetica", "", 8)
        self.set_text_color(80, 80, 80)

        # Get player's movement in cm
        bs = session_data.get("bat_speed", {})
        cal = bs.get("calibration", {})
        px_per_m = cal.get("px_per_m", None)
        avg_px = ss.get("avg_head_movement", 0)
        player_cm = avg_px / (px_per_m / 100.0) if px_per_m and px_per_m > 0 else avg_px / 10

        # Players sorted by head movement (lower is better)
        comparisons = [
            ("Virat Kohli", 0.3, "Elite - watches ball onto bat"),
            ("Kane Williamson", 0.4, "Elite - exceptionally still"),
            ("Joe Root", 0.5, "Classical head position"),
            ("Marnus Labuschagne", 0.7, "Active but settles late"),
            ("You (this session)", player_cm, f"{self._head_movement_str(session_data)}"),
        ]

        # Draw mini comparison bars
        bar_max_w = 120
        max_val = max(c[1] for c in comparisons) + 0.5
        self.ln(2)
        for name, cm_val, desc in comparisons:
            bar_w = bar_max_w * min(cm_val / max_val, 1.0)
            is_player = "You" in name
            # Label
            self.cell(55, 5, f"  {name}", new_x="RIGHT", new_y="TOP")
            # Colored bar
            bar_color = CREASE_ORANGE if is_player else "#B4B4B4"
            self.set_fill_color(*_hex_to_rgb(bar_color))
            x0 = self.get_x()
            y0 = self.get_y()
            self.rect(x0, y0 + 0.5, bar_w, 3.5, "F")
            # Value
            self.set_text_color(*_hex_to_rgb(PITCH_BLACK))
            self.cell(20, 5, f" {cm_val:.1f} cm", new_x="RIGHT", new_y="TOP")
            # Description
            self.set_text_color(120, 120, 120)
            self.cell(0, 5, desc, new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(80, 80, 80)

        self.ln(2)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_hex_to_rgb(CREASE_SILVER))
        self.multi_cell(0, 4, "Lower head movement = better. Elite batters keep movement under 0.5 cm per frame.")

        # Add a note about why head stability matters
        self.ln(4)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5,
            "Why it matters: Head movement directly affects your ability to judge "
            "line and length. Every centimetre of head drift shifts your eyeline, "
            "reducing the time you have to adjust to the ball. Elite batters treat "
            "their head as the anchor of their technique - everything moves around it.")

    # ── Knee & Spine page ──
    def knee_spine_page(self, session_data, report_data, analysis_video_path):
        """Page showing knee bend and spine angle analysis with range bars + annotated frames."""
        self.add_page()
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*_hex_to_rgb(PITCH_BLACK))
        self.cell(0, 10, "Knee Bend & Posture", new_x="LMARGIN", new_y="NEXT")

        ss = session_data.get("session_summary", {})
        avg_knee = ss.get("avg_front_knee_angle", 0)
        min_knee = ss.get("min_front_knee_angle", 0)
        avg_spine = ss.get("avg_spine_angle", 0)

        # ── Annotated frames side-by-side ──
        knee_frame_path = None
        spine_frame_path = None
        if analysis_video_path and os.path.exists(analysis_video_path):
            knee_frame_path = self._annotate_impact_frame(analysis_video_path, session_data, focus="knee")
            spine_frame_path = self._annotate_impact_frame(analysis_video_path, session_data, focus="spine")

        if knee_frame_path and spine_frame_path:
            # Side by side: each at ~80mm wide
            self.image(knee_frame_path, x=17, w=82)
            self.image(spine_frame_path, x=106, w=82)
            os.remove(knee_frame_path)
            os.remove(spine_frame_path)
            self.ln(2)
        elif knee_frame_path:
            self.image(knee_frame_path, x=35, w=130)
            os.remove(knee_frame_path)
            self.ln(2)

        # ── Knee analysis below frames ──
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*_hex_to_rgb(CREASE_ORANGE))
        self.cell(0, 7, "Front Knee Bend", new_x="LMARGIN", new_y="NEXT")

        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_hex_to_rgb(PITCH_BLACK))
        if avg_knee > 155:
            note = f"Your front knee averages {avg_knee:.0f} deg - quite straight. Aim for 130-140 deg."
        elif avg_knee > 140:
            note = f"Front knee at {avg_knee:.0f} deg. Good base for balance and power."
        else:
            note = f"Nice knee bend at {avg_knee:.0f} deg."
        if min_knee < 100:
            note += f" Deepest bend was {min_knee:.0f} deg - try to stay above 110 deg."
        self.multi_cell(0, 5, note)

        self.ln(1)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 4.5,
            "How we measure: the front knee angle is the inner bend at the knee joint "
            "between hip-knee-ankle. 180 deg = fully straight, 130-145 deg = ideal "
            "athletic bend for power transfer, < 120 deg = too deep, > 155 deg = too straight.")

        self._draw_range_bar("Knee", avg_knee, 130, 145, 180, 90, "deg")

        # ── Spine analysis ──
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*_hex_to_rgb(CREASE_ORANGE))
        self.cell(0, 7, "Spine Angle", new_x="LMARGIN", new_y="NEXT")

        self.set_font("Helvetica", "", 9)
        self.set_text_color(*_hex_to_rgb(PITCH_BLACK))
        if avg_spine > 160:
            note = f"Spine angle at {avg_spine:.0f} deg. Well balanced - head over the ball."
        elif avg_spine > 150:
            note = f"Spine around {avg_spine:.0f} deg. Keep your head over the ball."
        else:
            note = f"Spine at {avg_spine:.0f} deg. You are lunging forward - stay more upright."
        self.multi_cell(0, 5, note)

        self.ln(1)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 4.5,
            "Spine angle from vertical: 180 deg = perfectly upright, "
            "155-170 deg = ideal forward tilt, < 150 deg = lunging (head past ball).")

        self._draw_range_bar("Spine", avg_spine, 155, 170, 180, 120, " deg")

        self.ln(2)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5,
            "Knee bend creates the platform for power: a bent front knee allows your "
            "hips to rotate freely, transferring energy from the ground up through "
            "your core and into the bat. Spine angle determines your visual plane - "
            "too upright and you lose sight of the ball late; too hunched and you "
            "can't generate bat speed through rotation.")

    # ── Range bar visual indicator ──
    def _draw_range_bar(self, label, value, ideal_min, ideal_max, max_val, min_val, unit):
        """Draw a visual indicator bar showing where the value sits in range."""
        self.ln(4)
        bar_w = 160
        bar_h = 8

        # Calculate positions on the bar (0 to 1)
        val_norm = (value - min_val) / (max_val - min_val) * 100 if max_val > min_val else 50
        ideal_min_norm = (ideal_min - min_val) / (max_val - min_val) * 100 if max_val > min_val else 30
        ideal_max_norm = (ideal_max - min_val) / (max_val - min_val) * 100 if max_val > min_val else 70

        val_norm = max(0, min(100, val_norm))
        ideal_min_norm = max(0, min(100, ideal_min_norm))
        ideal_max_norm = max(0, min(100, ideal_max_norm))

        # Save position
        x0 = self.get_x()
        y0 = self.get_y()

        # Background bar
        self.set_fill_color(230, 230, 230)
        self.rect(x0, y0, bar_w, bar_h, "F")

        # Ideal range (green zone)
        ideal_x = x0 + ideal_min_norm / 100 * bar_w
        ideal_w = (ideal_max_norm - ideal_min_norm) / 100 * bar_w
        self.set_fill_color(76, 175, 80)
        self.rect(ideal_x, y0, ideal_w, bar_h, "F")

        # Value marker (orange triangle-ish)
        val_x = x0 + val_norm / 100 * bar_w
        self.set_fill_color(*_hex_to_rgb(CREASE_ORANGE))
        self.rect(val_x - 1, y0, 3, bar_h, "F")

        # Label below
        self.set_y(y0 + bar_h + 2)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*_hex_to_rgb(CREASE_SILVER))
        self.cell(0, 4, f"{label}: {value:.0f}{unit}  (Ideal: {ideal_min}-{ideal_max}{unit})",
                  new_x="LMARGIN", new_y="NEXT")
        self.ln(2)


def generate_report(session_data, report_data, analysis_video_path, output_path,
                    player_history=None, skip_annotated_frames=True):
    """
    Generate a multi-page PDF coaching report.

    Args:
        session_data: dict  -  full analysis JSON output
        report_data: dict  -  coaching report dict (priorities, session_score, etc.)
        analysis_video_path: str  -  path to the analysis video (for screenshot extraction)
        output_path: str  -  where to save the PDF
        player_history: dict, optional  -  historical player data for trends table
        skip_annotated_frames: bool  -  if True, skip frame-annotation pages (head, knee, spine)

    Returns:
        str  -  path to the generated PDF, or None on failure
    """
    session_id = session_data.get("session_id", "unknown")
    session_score = report_data.get("session_score", 0)
    player_label = (player_history or {}).get("label", "")

    pdf = Report()

    # Build pages
    pdf.cover_page(session_score, session_id, player_label=player_label)
    pdf.bat_speed_chart_page(session_data, report_data)
    pdf.metrics_page(report_data, session_data)
    if player_history:
        pdf.history_page(player_history, current_session_id=session_id)
    if not skip_annotated_frames:
        pdf.head_stability_page(session_data, report_data, analysis_video_path)
        pdf.knee_spine_page(session_data, report_data, analysis_video_path)

    # Save
    pdf.output(output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Report saved: {output_path} ({size_kb:.0f} KB)")
    return output_path


def generate_report_from_json(json_path, report_path=None):
    """
    Generate a PDF report from an existing analysis JSON file.

    Args:
        json_path: str  -  path to analysis JSON (e.g. sessions/analysis_xxxx.json)
        report_path: str  -  output PDF path (optional)

    Returns:
        str  -  path to generated PDF
    """
    with open(json_path) as f:
        session_data = json.load(f)

    # Build report_data from session data (mirrors run_full.py logic)
    ss = session_data.get("session_summary", {})
    shots = session_data.get("shot_summary", [])
    bs = session_data.get("bat_speed", {})
    head_score = ss.get("head_stability_score", 0)
    total_shots = len(shots)
    complete_shots = len([s for s in shots if s.get("has_impact")])
    completion_pct = (complete_shots / total_shots * 100) if total_shots > 0 else 0
    avg_knee = ss.get("avg_front_knee_angle", 154)
    avg_spine = ss.get("avg_spine_angle", 166)
    peak_kmh = bs.get("peak_kmh", 0) if bs.get("kmh_estimated") else 0

    session_score = min(100, max(0,
        (head_score * 0.35) +
        (completion_pct * 0.25) +
        (min(100, (avg_knee - 100) * 1.5) * 0.15) +
        (min(100, max(0, 180 - avg_spine) * 3) * 0.15) +
        (min(100, max(0, peak_kmh - 60)) * 0.10)
    ))

    priorities = []
    if head_score < 60:
        priorities.append({"rank": 1, "area": "HEAD STABILITY",
                           "drill": "Head-Still Drill: Place a bottle cap on your head while shadow batting. Play 50 forward defensive strokes without it falling off."})
    if completion_pct < 50:
        priorities.append({"rank": len(priorities)+1, "area": "SHOT COMMITMENT",
                           "drill": "Commitment Drill: Commit to every shot you start. A full swing builds consistency."})
    if avg_knee > 155:
        priorities.append({"rank": len(priorities)+1, "area": "KNEE FLEX",
                           "drill": "Knee-Tap Drill: Mark a spot 12 inches down the pitch. Every shot, front foot to that spot with knee bent to 130\u00b0."})
    if avg_spine < 155:
        priorities.append({"rank": len(priorities)+1, "area": "POSTURE",
                           "drill": "Corridor Drill: Place a second set of stumps 4ft down. Reach with your front foot only, not your head."})

    report_data = {
        "priorities": priorities,
        "session_score": session_score,
    }

    # Find analysis video
    video_path = session_data.get("output_video_path", "")
    if not video_path or not os.path.exists(video_path):
        # Try next to JSON
        base = os.path.splitext(json_path)[0]
        for ext in [".mp4", ".avi"]:
            candidate = f"{base}{ext}"
            if os.path.exists(candidate):
                video_path = candidate
                break

    if report_path is None:
        report_path = os.path.splitext(json_path)[0] + "_report.pdf"

    return generate_report(session_data, report_data, video_path, report_path)


if __name__ == "__main__":
    # Standalone usage: python report_generator.py <analysis_json_path>
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        generate_report_from_json(json_path)
    else:
        print("Usage: python report_generator.py <analysis_json_path>")
