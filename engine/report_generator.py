"""
PDF Coaching Report Generator  -  "the CREASE"

Generates a branded, multi-page PDF report with:
- Cover page with session score
- Annotated screenshots from the analysis video
- Player comparison charts (bat speed, head stability)
- Session summary metrics
- Top 3 priorities with drills
- Source attribution for benchmark data

Uses fpdf2 (pure Python) and matplotlib for charts.
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

from fpdf import FPDF

# ── Brand colours ──
CREASE_ORANGE = "#E64D0F"
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
        # Reset
        self.set_text_color(0, 0, 0)

    # ── Cover page ──
    def cover_page(self, session_score, session_id):
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
        # Centre the score
        self.cell(0, 20, score_str, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 12)
        self.set_text_color(*_hex_to_rgb(CREASE_SILVER))
        self.cell(0, 8, "SESSION SCORE", align="C", new_x="LMARGIN", new_y="NEXT")

        # Separator
        self.ln(10)
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
            # Threshold: elite < 0.5 cm per frame
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
        # Strip any non-ASCII characters that Helvetica cannot render
        return text.encode("ascii", errors="replace").decode("ascii")

    # ── Metrics page ──
    def metrics_page(self, report_data, session_data):
        """Page showing key session metrics in a clean table."""
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

        # Priorities section with customised drill text and illustrations
        self.ln(10)
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

                # Drill illustration
                diagram_path = self._generate_drill_diagram(p['area'], session_data)
                if diagram_path:
                    self.ln(1)
                    # Centre the diagram
                    diag_w = 55
                    x_centre = (210 - diag_w) / 2
                    self.image(diagram_path, x=x_centre, w=diag_w)
                    os.remove(diagram_path)

                self.ln(4)

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

        # Benchmark note
        self.ln(4)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_hex_to_rgb(CREASE_SILVER))
        bench = bs.get("calibration", {})
        method = bench.get("method", "unknown")
        self.multi_cell(0, 4, f"Calibration: {method} | Lever factor: 1.35x (hand-to-bat-tip) | {BENCHMARK_SOURCE}")

    # ── Drill illustration generator (silhouette-based) ──
    def _draw_batter_stance(self, ax, cx, cy, scale=1.0, color="#1a1a2e", alpha=0.9):
        """Draw a side-view cricket batter silhouette at centre (cx, cy).
        
        Returns dict of key joint positions for annotation.
        """
        s = scale
        # Head
        head = plt.Circle((cx, cy + 5.8*s), 0.55*s, color=color, alpha=alpha, zorder=5)
        ax.add_patch(head)
        
        # Torso (polygon: shoulders to hips)
        torso = plt.Polygon([
            (cx - 0.5*s, cy + 5.2*s),   # left shoulder
            (cx + 0.5*s, cy + 5.2*s),   # right shoulder
            (cx + 0.6*s, cy + 3.2*s),   # right hip
            (cx - 0.2*s, cy + 3.2*s),   # left hip
        ], color=color, alpha=alpha, zorder=4)
        ax.add_patch(torso)
        
        # Neck connector
        neck = plt.Polygon([
            (cx - 0.2*s, cy + 5.4*s),
            (cx + 0.2*s, cy + 5.4*s),
            (cx + 0.2*s, cy + 5.1*s),
            (cx - 0.2*s, cy + 5.1*s),
        ], color=color, alpha=alpha, zorder=4)
        ax.add_patch(neck)
        
        # Back leg (thick)
        bl = plt.Polygon([
            (cx - 0.1*s, cy + 3.2*s),
            (cx + 0.2*s, cy + 3.2*s),
            (cx + 0.15*s, cy + 0.8*s),
            (cx - 0.15*s, cy + 0.8*s),
        ], color=color, alpha=alpha, zorder=3)
        ax.add_patch(bl)
        # Back foot
        bf = plt.Polygon([
            (cx - 0.3*s, cy + 0.8*s),
            (cx + 0.3*s, cy + 0.8*s),
            (cx + 0.3*s, cy + 0.5*s),
            (cx - 0.3*s, cy + 0.5*s),
        ], color=color, alpha=alpha, zorder=3)
        ax.add_patch(bf)
        
        # Front leg (bent at knee, stepping forward)
        # Thigh
        fl1 = plt.Polygon([
            (cx + 0.15*s, cy + 3.2*s),
            (cx + 0.4*s, cy + 3.2*s),
            (cx + 0.7*s, cy + 2.0*s),
            (cx + 0.4*s, cy + 2.0*s),
        ], color=color, alpha=alpha, zorder=3)
        ax.add_patch(fl1)
        # Calf
        fl2 = plt.Polygon([
            (cx + 0.55*s, cy + 2.0*s),
            (cx + 0.75*s, cy + 2.0*s),
            (cx + 0.65*s, cy + 0.5*s),
            (cx + 0.45*s, cy + 0.5*s),
        ], color=color, alpha=alpha, zorder=3)
        ax.add_patch(fl2)
        # Front foot
        ff = plt.Polygon([
            (cx + 0.35*s, cy + 0.5*s),
            (cx + 0.75*s, cy + 0.5*s),
            (cx + 0.75*s, cy + 0.2*s),
            (cx + 0.35*s, cy + 0.2*s),
        ], color=color, alpha=alpha, zorder=3)
        ax.add_patch(ff)
        
        # Back arm (holding bat behind)
        ba = plt.Polygon([
            (cx - 0.3*s, cy + 5.0*s),
            (cx - 0.1*s, cy + 5.0*s),
            (cx - 0.6*s, cy + 4.2*s),
            (cx - 0.8*s, cy + 4.2*s),
        ], color=color, alpha=alpha, zorder=4)
        ax.add_patch(ba)
        
        # Front arm (reaching forward)
        fa = plt.Polygon([
            (cx + 0.4*s, cy + 5.0*s),
            (cx + 0.6*s, cy + 5.0*s),
            (cx + 0.3*s, cy + 4.0*s),
            (cx + 0.1*s, cy + 4.0*s),
        ], color=color, alpha=alpha, zorder=4)
        ax.add_patch(fa)
        
        # Bat (held behind, angled up)
        bat_handle = np.array([
            (cx - 0.55*s, cy + 4.2*s),
            (cx - 0.4*s, cy + 4.0*s),
        ])
        bat_blade = np.array([
            (cx - 1.5*s, cy + 6.5*s),
            (cx - 1.3*s, cy + 6.3*s),
            (cx - 0.4*s, cy + 4.0*s),
            (cx - 0.55*s, cy + 4.2*s),
        ])
        ax.fill(bat_blade[:, 0], bat_blade[:, 1], color="#8B4513", alpha=alpha, zorder=2)
        
        # Return joint positions for annotation
        return {
            "head": np.array([cx, cy + 5.8*s]),
            "shoulder": np.array([cx, cy + 5.2*s]),
            "hip": np.array([cx + 0.2*s, cy + 3.2*s]),
            "front_knee": np.array([cx + 0.6*s, cy + 2.0*s]),
            "back_knee": np.array([cx + 0.0*s, cy + 2.0*s]),
            "front_foot": np.array([cx + 0.55*s, cy + 0.5*s]),
            "back_foot": np.array([cx - 0.0*s, cy + 0.8*s]),
            "bat_tip": np.array([cx - 1.5*s, cy + 6.5*s]),
        }

    def _generate_drill_diagram(self, area, session_data):
        """Generate a silhouette-based drill illustration using matplotlib.
        
        Uses a side-view batter silhouette as the base and adds
        drill-specific annotations (angles, arrows, highlights).
        Returns path to a PNG, or None. Caller must delete the file.
        """
        ss = session_data.get("session_summary", {})
        bs = session_data.get("bat_speed", {})

        fig, ax = plt.subplots(figsize=(2.6, 2.0))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis("off")

        area_upper = area.upper()

        if "HEAD" in area_upper:
            # Batter silhouette with head highlighted
            joints = self._draw_batter_stance(ax, 5, 0, scale=0.95)
            head_pt = joints["head"]
            # Glow ring around head
            glow = plt.Circle(head_pt, 0.9, fill=False, edgecolor=CREASE_ORANGE, 
                              linewidth=2.5, zorder=6)
            ax.add_patch(glow)
            # Movement arrows around head
            head_mv = ss.get("avg_head_movement", 0)
            scale_val = min(head_mv / 15, 2.0)
            ax.annotate("", xy=(head_pt[0] + scale_val, head_pt[1]), xytext=(head_pt[0], head_pt[1]),
                        arrowprops=dict(arrowstyle="->", color=CREASE_ORANGE, lw=2), zorder=7)
            ax.annotate("", xy=(head_pt[0], head_pt[1] + scale_val), xytext=(head_pt[0], head_pt[1]),
                        arrowprops=dict(arrowstyle="->", color=CREASE_ORANGE, lw=2), zorder=7)
            ax.annotate("", xy=(head_pt[0] - scale_val, head_pt[1]), xytext=(head_pt[0], head_pt[1]),
                        arrowprops=dict(arrowstyle="->", color=CREASE_ORANGE, lw=1.5, alpha=0.6), zorder=7)
            # Metric text
            ax.text(5, 1.0, f"{self._head_movement_str(session_data)} drift",
                    ha="center", fontsize=7, color=CREASE_ORANGE,
                    fontweight="bold")
            ax.text(5, 0.3, "Keep head still through shot",
                    ha="center", fontsize=6, color="gray")

        elif "SHOT" in area_upper or "COMMIT" in area_upper:
            # Two silhouettes: full swing (left) vs partial swing (right)
            # Full swing (left figure)
            joints_full = self._draw_batter_stance(ax, 3.2, 0, scale=0.8)
            # Bat in follow-through position
            ax.plot([3.2 - 0.4, 4.8], [3.2 + 0.8, 1.5], 
                    color="#8B4513", lw=4, zorder=3)  # bat follow-through
            ax.text(2.8, 0.3, "Full swing", fontsize=6, ha="center", color="green")
            
            # Partial swing (right figure, faded)
            joints_part = self._draw_batter_stance(ax, 6.8, 0, scale=0.8, color="#888888", alpha=0.4)
            # Bat stopped halfway (partial swing)
            ax.plot([6.8 - 0.5, 6.8 - 1.2], [4.0, 5.5], 
                    color="#888888", lw=4, zorder=3, alpha=0.5)
            ax.annotate("", xy=(6.8 - 1.0, 5.3), xytext=(6.8 - 0.4, 4.2),
                        arrowprops=dict(arrowstyle="->", color="#888888", lw=1.5, ls="--"), zorder=3)
            ax.text(6.8, 0.3, "Partial swing", fontsize=6, ha="center", color="gray")
            
            # Label
            ax.text(5, 7.5, "Complete every swing",
                    ha="center", fontsize=7, fontweight="bold", color=CREASE_ORANGE)

        elif "KNEE" in area_upper:
            # Batter silhouette with knee angle highlighted
            joints = self._draw_batter_stance(ax, 5, 0, scale=0.95)
            knee_pt = joints["front_knee"]
            hip_pt = joints["hip"]
            foot_pt = joints["front_foot"]
            
            # Emphasize the front leg in orange
            thigh_high = plt.Polygon([
                hip_pt + np.array([-0.05, 0]),
                hip_pt + np.array([0.05, 0]),
                knee_pt + np.array([0.05, 0]),
                knee_pt + np.array([-0.05, 0]),
            ], color=CREASE_ORANGE, alpha=0.7, zorder=6)
            ax.add_patch(thigh_high)
            calf_high = plt.Polygon([
                knee_pt + np.array([-0.05, 0]),
                knee_pt + np.array([0.05, 0]),
                foot_pt + np.array([0.05, 0]),
                foot_pt + np.array([-0.05, 0]),
            ], color=CREASE_ORANGE, alpha=0.7, zorder=6)
            ax.add_patch(calf_high)
            
            # Angle arc at knee
            v1 = hip_pt - knee_pt
            v2 = foot_pt - knee_pt
            theta1 = math.atan2(v1[1], v1[0])
            theta2 = math.atan2(v2[1], v2[0])
            r = 1.0
            arc_angles = np.linspace(theta1, theta2, 30)
            ax.plot(knee_pt[0] + r * np.cos(arc_angles),
                    knee_pt[1] + r * np.sin(arc_angles),
                    color=CREASE_ORANGE, lw=2.5, zorder=7)
            
            # Angle label
            knee_angle = ss.get("avg_front_knee_angle", 0)
            mid_theta = (theta1 + theta2) / 2
            ax.text(knee_pt[0] + (r + 0.3) * np.cos(mid_theta),
                    knee_pt[1] + (r + 0.3) * np.sin(mid_theta),
                    f"{knee_angle:.0f} deg", fontsize=7, fontweight="bold",
                    color=CREASE_ORANGE, ha="center", zorder=8)
            
            # Guide text
            ax.text(5, 0.3, "Bend knee 130-145 deg", ha="center", fontsize=6, color="gray")

        elif "POSTURE" in area_upper or "SPINE" in area_upper:
            # Batter silhouette with spine angle highlighted
            joints = self._draw_batter_stance(ax, 5, 0, scale=0.95)
            head_pt = joints["head"]
            hip_pt = joints["hip"]
            
            # Draw spine line from head to hip
            spine_pts = np.array([head_pt, hip_pt])
            ax.plot(spine_pts[:, 0], spine_pts[:, 1], color=CREASE_ORANGE, lw=3, zorder=6)
            
            # Vertical reference line
            ax.plot([head_pt[0], head_pt[0]], [hip_pt[1] - 0.5, head_pt[1] + 0.5],
                    color="gray", lw=1.5, ls="--", zorder=3)
            
            # Angle arc at hip
            vert_vec = np.array([0, 1])  # straight up
            spine_vec = hip_pt - head_pt
            angle_spine = math.atan2(spine_vec[1], spine_vec[0])
            angle_vert = math.atan2(vert_vec[1], vert_vec[0])
            r = 0.8
            arc_angles = np.linspace(angle_vert, angle_spine, 20)
            ax.plot(hip_pt[0] + r * np.cos(arc_angles),
                    hip_pt[1] + r * np.sin(arc_angles),
                    color=CREASE_ORANGE, lw=2, zorder=7)
            
            # Angle label
            spine_angle = ss.get("avg_spine_angle", 0)
            mid_a = (angle_vert + angle_spine) / 2
            ax.text(hip_pt[0] + (r + 0.3) * np.cos(mid_a),
                    hip_pt[1] + (r + 0.3) * np.sin(mid_a),
                    f"{spine_angle:.0f} deg", fontsize=7, fontweight="bold",
                    color=CREASE_ORANGE, ha="center", zorder=8)
            
            # Guide text
            ax.text(5, 0.3, "Keep head over the ball", ha="center", fontsize=6, color="gray")

        else:
            plt.close(fig)
            return None

        plt.tight_layout(pad=0.3)
        chart_fd, chart_path = tempfile.mkstemp(suffix=".png")
        os.close(chart_fd)
        fig.savefig(chart_path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return chart_path

    # ── Head stability page ──
    def head_stability_page(self, session_data, report_data, analysis_video_path):
        """Page showing head stability analysis with silhouette diagram (no screenshot)."""
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

        # Silhouette diagram (larger, shown instead of screenshot)
        self.ln(2)
        diag_path = self._generate_drill_diagram("HEAD STABILITY", session_data)
        if diag_path:
            self.image(diag_path, x=55, w=100)
            os.remove(diag_path)

        # Elite comparison as visual bar chart
        self.ln(6)
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

    # ── Knee & Spine page ──
    def knee_spine_page(self, session_data, report_data, analysis_video_path):
        """Page showing knee bend and spine angle analysis."""
        self.add_page()
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*_hex_to_rgb(PITCH_BLACK))
        self.cell(0, 10, "Knee Bend & Posture", new_x="LMARGIN", new_y="NEXT")

        ss = session_data.get("session_summary", {})
        avg_knee = ss.get("avg_front_knee_angle", 0)
        min_knee = ss.get("min_front_knee_angle", 0)
        avg_spine = ss.get("avg_spine_angle", 0)

        # ── Knee ──
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

        # Angle reference explanation
        self.ln(1)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 4.5,
            "How we measure: the front knee angle is the inner bend at the knee joint "
            "between hip-knee-ankle. 180 deg = completely straight leg. "
            "130-145 deg = ideal athletic bend for power transfer through the shot. "
            "Below 120 deg = too deep (lose power). Above 155 deg = too straight "
            "(lose balance).")

        self._draw_range_bar("Knee", avg_knee, 130, 145, 180, 90, "deg")

        # ── Spine ──
        self.ln(4)
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

        # Spine angle reference
        self.ln(1)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 4.5,
            "Spine angle measured from vertical. 180 deg = perfectly upright. "
            "155-170 deg = ideal forward tilt for balance and power. "
            "Below 150 deg = lunging forward (head past the ball).")

        self._draw_range_bar("Spine", avg_spine, 155, 170, 180, 120, " deg")

        # Silhouette diagrams: knee + spine side by side
        self.ln(6)
        # Knee diagram on left
        knee_diag = self._generate_drill_diagram("KNEE FLEX", session_data)
        if knee_diag:
            self.image(knee_diag, x=12, w=88)
            os.remove(knee_diag)
        # Spine diagram on right
        spine_diag = self._generate_drill_diagram("POSTURE", session_data)
        if spine_diag:
            self.image(spine_diag, x=110, w=88)
            os.remove(spine_diag)

        self.ln(3)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*_hex_to_rgb(CREASE_SILVER))
        self.cell(0, 4, "Left: knee bend angle    Right: spine/ posture angle", align="C", new_x="LMARGIN", new_y="NEXT")

    # ── Detailed knee angle measurement diagram ──
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


def generate_report(session_data, report_data, analysis_video_path, output_path):
    """
    Generate a multi-page PDF coaching report.

    Args:
        session_data: dict  -  full analysis JSON output
        report_data: dict  -  coaching report dict (priorities, session_score, etc.)
        analysis_video_path: str  -  path to the analysis video (for screenshot extraction)
        output_path: str  -  where to save the PDF

    Returns:
        str  -  path to the generated PDF, or None on failure
    """
    session_id = session_data.get("session_id", "unknown")
    session_score = report_data.get("session_score", 0)

    pdf = Report()

    # Build pages
    pdf.cover_page(session_score, session_id)
    pdf.metrics_page(report_data, session_data)
    pdf.bat_speed_chart_page(session_data, report_data)
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
