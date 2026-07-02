"""
Highlight Reel Generator — auto-clip the best moments from a session.

Scans all detected shots, ranks them by impact quality (bat speed,
shot type rarity, ball speed), and exports 5-15 second clips as
sharable MP4 files with CREASE watermark overlay.

Output:
  - One or more short MP4 clips in the session output directory
  - Metadata dict with clip paths, scores, and shot descriptions
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


# Shot type "prestige" scores — rarer/more exciting shots rank higher
SHOT_PRESTIGE: Dict[str, float] = {
    "slog_sweep": 1.4,
    "reverse_sweep": 1.3,
    "ramp": 1.5,
    "upper_cut": 1.3,
    "lap_shot": 1.2,
    "cover_drive": 1.2,
    "straight_drive": 1.1,
    "pull": 1.1,
    "sweep": 1.0,
    "on_drive": 1.0,
    "flick": 0.9,
    "glance": 0.8,
    "square_cut": 0.9,
    "defensive_block": 0.3,
    "leave": 0.2,
    "unknown": 0.5,
}


class HighlightReel:
    """Generate shareable highlight clips from a completed analysis.

    Usage:
        reel = HighlightReel()
        clips = reel.generate(session_data, output_dir)
        # clips = [{"path": "...", "score": 85, "label": "Slog Sweep", ...}, ...]
    """

    def __init__(
        self,
        clip_duration_sec: float = 6.0,  # 3s before, 3s after impact
        max_clips: int = 5,
        min_quality_score: float = 40.0,
        target_fps: int = 30,
    ):
        self.clip_duration = clip_duration_sec
        self.max_clips = max_clips
        self.min_quality = min_quality_score
        self.target_fps = target_fps

    # ── Public API ──────────────────────────────────────────────────────

    def generate(
        self,
        session: Dict[str, Any],
        output_dir: str | Path,
        video_path: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Generate highlight clips from a completed session.

        Args:
            session: Session dictionary from the analysis pipeline.
            output_dir: Directory to write clip files to.
            video_path: Override path to the raw source video. If None,
                        uses ``session["video_path"]``.

        Returns:
            List of clip dicts::

                [{
                    "path": "/path/to/clip_01.mp4",
                    "filename": "clip_01.mp4",
                    "score": 85.0,
                    "shot_type": "cover_drive",
                    "label": "Cover Drive",
                    "start_frame": 240,
                    "end_frame": 420,
                    "description": "Cover Drive — 85% quality",
                }, ...]
        """
        source = video_path or session.get("video_path")
        if not source or not os.path.isfile(source):
            log.warning("HighlightReel: source video not found at %s", source)
            return []

        shots = session.get("shot_summary", [])
        if not shots:
            log.info("HighlightReel: no shots to clip")
            return []

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Rank shots by quality score
        ranked = self._rank_shots(shots, session)

        # Take top N above threshold
        candidates = [s for s in ranked if s["score"] >= self.min_quality][:self.max_clips]

        clips: List[Dict[str, Any]] = []
        for i, shot in enumerate(candidates, 1):
            clip_filename = f"highlight_{i:02d}.mp4"
            clip_path = str(output_path / clip_filename)

            ok = self._render_clip(
                source_video=source,
                output_path=clip_path,
                start_sec=shot["start_sec"],
                end_sec=shot["end_sec"],
                shot_label=shot["label"],
                shot_confidence=shot["confidence"],
                session_label=session.get("session_label", ""),
            )
            if ok:
                clips.append({
                    "path": clip_path,
                    "filename": clip_filename,
                    "score": round(shot["score"], 1),
                    "shot_type": shot["shot_type"],
                    "label": shot["label"],
                    "start_frame": shot["start_frame"],
                    "end_frame": shot["end_frame"],
                    "duration_sec": round(shot["end_sec"] - shot["start_sec"], 1),
                    "description": f"{shot['label']} — {shot['score']:.0f}% quality",
                })

        return clips

    # ── Shot ranking ────────────────────────────────────────────────────

    def _rank_shots(
        self,
        shots: List[Dict[str, Any]],
        session: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Score each shot for highlight-worthiness."""
        # Context stats for relative comparisons
        all_speeds = [
            s.get("bat_speed_px", 0) or 0
            for s in shots
        ]
        max_speed = max(all_speeds) if all_speeds else 1

        ranked = []
        for shot in shots:
            st = shot.get("shot_type", "unknown")
            conf = shot.get("classification_confidence", 0.5) or 0.5
            bat_speed = shot.get("bat_speed_px", 0) or 0

            # Prestige bonus for exciting shots
            prestige = SHOT_PRESTIGE.get(st, 0.5)

            # Speed relative to session max
            speed_ratio = bat_speed / max_speed if max_speed > 0 else 0.5

            # Confidence bonus
            conf_bonus = conf  # 0-1

            # Composite score 0-100
            raw = (prestige * 30) + (speed_ratio * 35) + (conf_bonus * 35)
            score = min(raw, 100.0)

            # Clip timing
            start_frame = max(0, shot.get("impact_frame", 0) - int(self.clip_duration * self.target_fps / 2))
            end_frame = start_frame + int(self.clip_duration * self.target_fps)
            fps = session.get("video_fps") or session.get("fps", 30)
            ranked.append({
                "shot_type": st,
                "label": st.replace("_", " ").title(),
                "confidence": conf,
                "score": score,
                "bat_speed_px": bat_speed,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_sec": start_frame / fps,
                "end_sec": end_frame / fps,
                "impact_frame": shot.get("impact_frame", 0),
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    # ── Clip rendering ─────────────────────────────────────────────────

    def _render_clip(
        self,
        source_video: str,
        output_path: str,
        start_sec: float,
        end_sec: float,
        shot_label: str,
        shot_confidence: float,
        session_label: str,
    ) -> bool:
        """Use ffmpeg to cut a segment and overlay CREASE watermark.

        Two-pass approach:
          1. ffmpeg: fast seek + trim to temp clip
          2. OpenCV: overlay watermark bar on temp clip
          3. ffmpeg: re-encode watermarked clip to final output
        """
        try:
            duration = end_sec - start_sec
            if duration <= 0.5:
                log.warning("Clip too short (%.2fs), skipping", duration)
                return False

            # Step 1: Trim source with ffmpeg
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                trimmed_path = tmp.name

            # Get ffmpeg binary
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg = get_ffmpeg_exe()

            trim_cmd = [
                ffmpeg, "-y",
                "-ss", str(max(0, start_sec)),
                "-i", source_video,
                "-t", str(duration),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22",
                "-pix_fmt", "yuv420p",
                "-an",
                trimmed_path,
            ]
            subprocess.run(trim_cmd, capture_output=True, check=True)

            # Step 2: Overlay watermark using OpenCV
            self._apply_watermark_to_video(trimmed_path, output_path,
                                            shot_label, shot_confidence,
                                            session_label)

            # Clean up temp
            if os.path.exists(trimmed_path):
                os.unlink(trimmed_path)

            return os.path.isfile(output_path) and os.path.getsize(output_path) > 1024

        except Exception as e:
            log.error("Failed to render highlight clip: %s", e)
            return False

    def _apply_watermark_to_video(
        self,
        input_path: str,
        output_path: str,
        shot_label: str,
        shot_confidence: float,
        session_label: str,
    ):
        """Apply CREASE branding overlay to every frame of the clip."""
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Brand colours
        ORANGE = (0, 80, 229)    # BGR -> #E55000
        DARK_BG = (10, 10, 10)
        WHITE = (255, 255, 255)
        GREY = (120, 120, 120)

        bar_h = int(height * 0.07)
        pct = int(shot_confidence * 100)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # Semi-transparent overlay bar at bottom
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - bar_h), (w, h), DARK_BG, -1)
            cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

            # Left: brand wordmark "the" + "CREASE" (thin/bold style via font)
            cv2.putText(frame, "the", (12, h - int(bar_h * 0.42)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1)
            cv2.putText(frame, "CREASE", (40, h - int(bar_h * 0.42)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, WHITE, 1)

            # Tagline
            cv2.putText(frame, '"Know your game."',
                        (12, h - int(bar_h * 0.15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, GREY, 1)

            # Middle: shot label + confidence
            label_text = f"{shot_label}  |  {pct}%"
            cv2.putText(frame, label_text, (int(w * 0.25), h - int(bar_h * 0.42)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)

            # Right: URL
            cv2.putText(frame, "thecrease.app", (w - 150, h - int(bar_h * 0.42)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREY, 1)

            # Top-right corner bug
            cv2.putText(frame, "CREASE", (w - 85, 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, ORANGE, 1)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

    # ── Static utilities ───────────────────────────────────────────────

    @staticmethod
    def best_clip(clips: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Return the highest-scoring clip."""
        return max(clips, key=lambda c: c["score"]) if clips else None

    @staticmethod
    def top_n(clips: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
        """Return the top N clips by score."""
        return sorted(clips, key=lambda c: c["score"], reverse=True)[:n]
