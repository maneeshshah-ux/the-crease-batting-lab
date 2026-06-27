"""
Scorecard Image Generator — shareable session summary graphic.

Produces a ready-to-post PNG image (1080x1080 Instagram-square or
1080x1350 portrait) with session stats, shot breakdown, bragging
rights, and CREASE branding.

Used by:
  - Social share (one-tap post to Instagram/WhatsApp/Twitter)
  - OG meta image for shared session links
  - Downloadable report card
"""

from __future__ import annotations

import math
import os
import textwrap
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# ── Brand constants ─────────────────────────────────────────────────────
BRAND_COLOR = "#E55000"
DARK_BG = "#0D0D0D"
CARD_BG = "#1A1A1A"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#AAAAAA"
TEXT_MUTED = "#666666"
ACCENT_GREEN = "#4CAF50"
ACCENT_AMBER = "#FFC107"
ACCENT_RED = "#F44336"

# Canvas sizes
CANVAS_SIZES = {
    "square": (1080, 1080),
    "portrait": (1080, 1350),
    "og": (1200, 630),  # Open Graph
}

# Shot type to emoji mapping (same as the app uses)
SHOT_EMOJI: Dict[str, str] = {
    "cover_drive": "\U0001F3AF",
    "on_drive": "\U0001F3AF",
    "straight_drive": "\U0001F3AF",
    "square_cut": "\u2702\ufe0f",
    "pull": "\U0001F4AA",
    "defensive_block": "\U0001F6E1\ufe0f",
    "leave": "\U0001F6AB",
    "sweep": "\U0001F9F9",
    "reverse_sweep": "\U0001F504",
    "slog_sweep": "\U0001F4A5",
    "lap_shot": "\U0001F998",
    "ramp": "\U0001F680",
    "upper_cut": "\u2B06\ufe0f",
    "glance": "\U0001F440",
    "flick": "\U0001F590\ufe0f",
    "unknown": "\u2753",
}


class ScorecardImage:
    """Generate shareable session scorecard images.

    Usage:
        card = ScorecardImage()
        img = card.create(session_data, fmt="square")
        img.save("scorecard.png")
    """

    def __init__(self):
        self._font_cache: Dict[str, ImageFont.FreeTypeFont] = {}

    # ── Public API ──────────────────────────────────────────────────────

    def create(
        self,
        session: Dict[str, Any],
        fmt: str = "square",
        output_path: Optional[str] = None,
    ) -> Image.Image:
        """Build a scorecard image for the given session.

        Args:
            session: Session dict from the analysis pipeline.
            fmt: ``"square"`` (IG post), ``"portrait"`` (IG story),
                 or ``"og"`` (social link preview).
            output_path: If given, saves the image to this path.

        Returns:
            PIL Image ready to share or save.
        """
        canvas_size = CANVAS_SIZES.get(fmt, CANVAS_SIZES["square"])
        img = Image.new("RGB", canvas_size, DARK_BG)
        draw = ImageDraw.Draw(img)

        self._draw_background(img, draw, canvas_size)
        self._draw_header(draw, canvas_size, session)
        self._draw_stats_row(draw, canvas_size, session)
        self._draw_shot_list(draw, canvas_size, session)
        self._draw_bragging_rights(draw, canvas_size, session)
        self._draw_footer(draw, canvas_size, session)

        if output_path:
            img.save(output_path, "PNG")

        return img

    # ── Layout ──────────────────────────────────────────────────────────

    def _draw_background(self, img: Image.Image, draw: ImageDraw, size: Tuple[int, int]):
        """Subtle gradient or pattern background."""
        w, h = size
        # Simple radial-gradient approximation: overlay dark vignette
        overlay = Image.new("RGB", size, DARK_BG)
        img.paste(overlay)

        # Top accent stripe
        for y in range(4):
            draw.rectangle([(0, y), (w, y + 1)], fill=BRAND_COLOR)

    def _draw_header(self, draw: ImageDraw, size: Tuple[int, int], session: Dict[str, Any]):
        """Brand wordmark (Lato 100/900) + tagline + session title."""
        w, _ = size
        y = 28

        # Brand wordmark: "the" (thin) + "CREASE" (black)
        thin_font = self._font(30, bold=False)
        bold_font = self._font(36, bold=True)
        tag_font = self._font(14, bold=False)

        draw.text((40, y), "the", fill=BRAND_COLOR, font=thin_font)
        draw.text((40 + 48, y), "CREASE", fill=TEXT_PRIMARY, font=bold_font)

        # Tagline
        draw.text((40, y + 40), "\u201cKnow your game.\u201d",
                  fill=TEXT_MUTED, font=tag_font)

        # Session label or date
        label = session.get("session_label") or ""
        ts = session.get("analysis_timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                date_str = dt.strftime("%d %b %Y")
            except Exception:
                date_str = ts[:10] if len(ts) >= 10 else ""
        else:
            date_str = ""

        sub_font = self._font(16)
        meta_parts = [p for p in [label, date_str] if p]
        if meta_parts:
            draw.text((40, y + 66), " \u00b7 ".join(meta_parts),
                      fill=TEXT_SECONDARY, font=sub_font)

        # Divider line
        draw.line([(40, y + 100), (w - 40, y + 100)], fill=CARD_BG, width=1)

    def _draw_stats_row(self, draw: ImageDraw, size: Tuple[int, int], session: Dict[str, Any]):
        """Key metrics in a horizontal row."""
        w, _ = size
        y = 150

        shots = session.get("num_shots_detected", 0)
        dur = session.get("duration_sec", 0)
        top_speed = session.get("bat_speed", {})
        speed_val = top_speed.get("swing_avg_kmh", 0) if isinstance(top_speed, dict) else top_speed

        metrics = [
            ("Shots", str(shots)),
            ("Duration", f"{dur:.0f}s" if dur else "--"),
            ("Top Speed", f"{speed_val:.0f} km/h" if speed_val else "--"),
            ("Hand", session.get("batting_hand", "--").title()),
        ]

        # If there are shots with classifications, count unique shot types
        shot_summary = session.get("shot_summary", [])
        shot_types = {s.get("shot_type") for s in shot_summary if s.get("shot_type") and s["shot_type"] != "unknown"}
        metrics.append(("Shot Types", str(len(shot_types))))

        n = len(metrics)
        if n == 0:
            return

        col_w = (w - 80) // n
        stat_font = self._font(26, bold=True)
        label_font = self._font(14)

        for i, (label, value) in enumerate(metrics):
            x = 40 + i * col_w + (col_w // 2)
            draw.text((x, y), value, fill=BRAND_COLOR, font=stat_font, anchor="mt")
            draw.text((x, y + 34), label, fill=TEXT_MUTED, font=label_font, anchor="mt")

        # Divider
        draw.line([(40, y + 72), (w - 40, y + 72)], fill=CARD_BG, width=1)

    def _draw_shot_list(self, draw: ImageDraw, size: Tuple[int, int], session: Dict[str, Any]):
        """List of detected shots with icons."""
        w, _ = size
        y = 250

        # Section heading
        heading_font = self._font(16, bold=True)
        draw.text((40, y), "Shot Breakdown", fill=TEXT_SECONDARY, font=heading_font)
        y += 28

        shot_summary = session.get("shot_summary", [])
        shots_to_show = shot_summary[:8]  # max 8 in grid

        if not shots_to_show:
            draw.text((40, y), "No shots detected", fill=TEXT_MUTED,
                      font=self._font(14))
            return

        # 2-column grid
        col_w = (w - 80) // 2
        row_h = 36
        item_font = self._font(15)
        small_font = self._font(12)

        for i, shot in enumerate(shots_to_show):
            col = i % 2
            row = i // 2
            x = 40 + col * col_w
            sy = y + row * row_h

            shot_type = shot.get("shot_type", "unknown")
            conf = shot.get("classification_confidence", 0) or 0
            emoji = SHOT_EMOJI.get(shot_type, "\u2753")
            label = shot_type.replace("_", " ").title()

            # Confidence color
            conf_color = ACCENT_GREEN if conf > 0.7 else (ACCENT_AMBER if conf > 0.5 else ACCENT_RED)

            draw.text((x, sy), f"{emoji}  {label}", fill=TEXT_PRIMARY, font=item_font)
            draw.text((x + 320, sy), f"{conf * 100:.0f}%",
                      fill=conf_color, font=small_font)

    def _draw_bragging_rights(self, draw: ImageDraw, size: Tuple[int, int], session: Dict[str, Any]):
        """Bragging rights / comparative stats."""
        w, h = size
        shot_summary = session.get("shot_summary", [])

        if not shot_summary:
            return

        # Find best shot
        best_shot = max(
            shot_summary,
            key=lambda s: s.get("classification_confidence", 0),
            default=None,
        )
        best_type = best_shot.get("shot_type", "").replace("_", " ").title() if best_shot else ""
        best_conf = (best_shot.get("classification_confidence", 0) or 0) * 100 if best_shot else 0

        # Count exciting shots
        exciting = {"slog_sweep", "reverse_sweep", "pull", "upper_cut", "ramp", "cover_drive"}
        exciting_count = sum(
            1 for s in shot_summary
            if s.get("shot_type") in exciting
        )

        # Shot diversity
        unique_types = {
            s.get("shot_type") for s in shot_summary
            if s.get("shot_type") and s["shot_type"] != "unknown"
        }
        diversity = len(unique_types)

        lines = []
        if best_type and best_conf > 50:
            lines.append(f"Best shot: {best_type} ({best_conf:.0f}% confidence)")
        if exciting_count >= 2:
            lines.append(f"{exciting_count} exciting shots in this session")
        if diversity >= 4:
            lines.append(f"Played {diversity} different shot types — great variety!")
        elif diversity <= 2:
            lines.append("Try adding more shots to your repertoire")

        if not lines:
            return

        y = h - 160
        section_font = self._font(14)
        item_font = self._font(16, bold=True)

        draw.text((40, y), "Session Highlights", fill=TEXT_SECONDARY, font=section_font)
        y += 24

        for line in lines:
            draw.text((40, y), f"\u2605  {line}", fill=TEXT_PRIMARY, font=item_font)
            y += 28

    def _draw_footer(self, draw: ImageDraw, size: Tuple[int, int], session: Dict[str, Any]):
        """Bottom brand bar with tagline, URL and disclaimer."""
        w, h = size
        bar_y = h - 56

        # Bar background
        draw.rectangle([(0, bar_y), (w, h)], fill=CARD_BG)

        # Left: brand wordmark (Lato style: thin 'the' + bold 'CREASE')
        thin_font = self._font(10, bold=False)
        bold_font = self._font(14, bold=True)
        draw.text((40, bar_y + 8), "the", fill=BRAND_COLOR, font=thin_font)
        draw.text((40 + 18, bar_y + 8), "CREASE", fill=TEXT_PRIMARY, font=bold_font)

        # Tagline
        draw.text((40, bar_y + 28), "\u201cKnow your game.\u201d",
                  fill=TEXT_MUTED, font=self._font(10))

        # Right: URL
        share_token = session.get("share_token", "")
        url = f"thecrease.app/s/{share_token}" if share_token else "thecrease.app"
        draw.text((w - 40, bar_y + 12), url,
                  fill=TEXT_MUTED, font=self._font(11), anchor="rt")

        # Very bottom: disclaimer
        disc_font = self._font(8)
        draw.text((40, h - 12),
                  "AI-powered analysis \u00b7 Comparison with professional players for educational purposes only.",
                  fill=TEXT_MUTED, font=disc_font)

    # ── Font helpers ────────────────────────────────────────────────────

    def _font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Load a brand font, falling back to system default.

        Priority:
          1. Lato (headings, brand) — macOS bundled or Google Fonts download
          2. Inter (body/UI)
          3. System fallback
        """
        cache_key = f"{size}_{bold}"
        if cache_key not in self._font_cache:
            paths = []
            import platform as pf
            if pf.system() == "Darwin":
                if bold:
                    paths = [
                        "/System/Library/Fonts/Supplemental/Lato-Bold.ttf",
                        "/Library/Fonts/Lato-Bold.ttf",
                    ]
                else:
                    paths = [
                        "/System/Library/Fonts/Supplemental/Lato-Regular.ttf",
                        "/Library/Fonts/Lato-Regular.ttf",
                    ]
            else:
                paths = [
                    "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
                    "/usr/share/fonts/opentype/lato/Lato-Regular.otf",
                ]
                if bold:
                    paths = [
                        "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
                        "/usr/share/fonts/opentype/lato/Lato-Bold.otf",
                    ]
            loaded = False
            for p in paths:
                try:
                    self._font_cache[cache_key] = ImageFont.truetype(p, size)
                    loaded = True
                    break
                except (OSError, IOError):
                    continue
            if not loaded:
                try:
                    # Fallback: try system sans-serif
                    if pf.system() == "Darwin":
                        fallback = "/System/Library/Fonts/Helvetica.ttc" if not bold else "/System/Library/Fonts/Helvetica-Bold.ttf"
                        self._font_cache[cache_key] = ImageFont.truetype(fallback, size)
                    else:
                        self._font_cache[cache_key] = ImageFont.truetype(
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size
                        )
                except (OSError, IOError):
                    self._font_cache[cache_key] = ImageFont.load_default()
        return self._font_cache[cache_key]

    # ── Open Graph image shortcut ───────────────────────────────────────

    @staticmethod
    def make_og_image(session: Dict[str, Any]) -> BytesIO:
        """Convenience: return OG image as BytesIO for Flask response."""
        card = ScorecardImage()
        img = card.create(session, fmt="og")
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf
