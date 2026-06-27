"""
Voiceover — Natural-sounding coaching commentary using edge-tts (Microsoft neural TTS).

Sounds like a real coach chatting to you, not a robot reading a report.
"""

import os
import json
import subprocess
import tempfile
import asyncio


FFMPEG_PATH = os.path.join(os.path.dirname(__file__), "..", "ffmpeg")
if not os.path.exists(FFMPEG_PATH):
    FFMPEG_PATH = "ffmpeg"


def _ensure_ffmpeg():
    if not os.path.exists(FFMPEG_PATH):
        import shutil
        sys_ffmpeg = shutil.which("ffmpeg")
        if sys_ffmpeg:
            return sys_ffmpeg
        raise FileNotFoundError("ffmpeg not found")
    return FFMPEG_PATH


async def _generate_speech_async(text, output_path, voice="en-AU-WilliamNeural"):
    """
    Generate speech using edge-tts (Microsoft neural TTS).

    Voices:
        en-AU-WilliamNeural — male, Australian (good for cricket coach)
        en-AU-NatashaNeural — female, Australian
        en-GB-RyanNeural   — male, British
    """
    print(f"  Generating natural voiceover ({len(text)} chars)...")
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(output_path)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  Voiceover saved: {size_kb:.0f} KB")
        return True
    except Exception as e:
        print(f"  edge-tts failed: {e}")
        print("  Falling back to gTTS...")
        return False


def text_to_speech(text, output_path, voice="en-AU-WilliamNeural"):
    """Generate speech (edge-tts preferred, fallback to gTTS)."""
    # Try edge-tts first
    try:
        asyncio.run(_generate_speech_async(text, output_path, voice))
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return output_path
    except Exception:
        pass

    # Fallback to gTTS
    print("  Using gTTS fallback...")
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", tld="com.au", slow=False)
        tts.save(output_path)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  Voiceover saved: {size_kb:.0f} KB")
    except Exception as e:
        print(f"  gTTS also failed: {e}")
        raise

    return output_path


def generate_coaching_script(session_data, report_data):
    """
    Generates a concise coaching debrief — sounds like a real coach
    giving quick feedback after a practice session.
    Kept intentionally short to fit within typical video durations.
    """
    shots = session_data.get("shot_summary", [])
    ss = session_data.get("session_summary", {})
    bat_speed = session_data.get("bat_speed", {})

    total_shots = len(shots)
    complete_shots = len([s for s in shots if s.get("has_impact")])

    head_score = ss.get("head_stability_score", 0)
    avg_movement_px = ss.get("avg_head_movement", 0)
    avg_knee = ss.get("avg_front_knee_angle", 154)
    min_knee = ss.get("min_front_knee_angle", 154)
    avg_spine = ss.get("avg_spine_angle", 166)
    session_score = report_data.get("session_score", 50)

    # Convert head movement to cm if calibration available
    cal = bat_speed.get("calibration", {})
    px_per_m = cal.get("px_per_m", None)
    if px_per_m and px_per_m > 0:
        avg_movement_cm = avg_movement_px / (px_per_m / 100.0)
        movement_str = f"{avg_movement_cm:.1f} centimetres"
    else:
        movement_str = f"{avg_movement_px:.1f} pixels"

    # Bat speed with player comparison
    if bat_speed and bat_speed.get("kmh_estimated"):
        peak_kmh = bat_speed.get("peak_kmh", 0)
        avg_kmh = bat_speed.get("speed_kmh", 0)
        if peak_kmh > 20:
            # Find nearest benchmark
            benchmarks = [
                ("Ellyse Perry", 115, "women's international"),
                ("Virat Kohli", 130, "men's international"),
                ("AB de Villiers", 150, "world class"),
            ]
            comparison = None
            for pname, pspeed, plevel in benchmarks:
                if peak_kmh <= pspeed:
                    comparison = (pname, pspeed, plevel)
                    break
            if comparison:
                pname, pspeed, plevel = comparison
                bat_line = (f"Bat speed peaked at {peak_kmh:.0f} kilometres an hour. "
                           f"For context, {pname} generates {pspeed} at {plevel} level.")
            else:
                bat_line = f"Bat speed topped out at {peak_kmh:.0f} kilometres an hour."
        else:
            bat_line = ""
    else:
        bat_line = ""

    # Head stability
    if head_score < 40:
        head_line = (f"Your head moves {movement_str} on average. "
                     f"Top players keep it under half a centimetre. "
                     f"Watch the ball all the way and keep your head still.")
    elif head_score < 60:
        head_line = f"Head stability is improving, but still {movement_str} of movement. Keep your head still through the shot."
    else:
        head_line = f"Head stability is solid at {head_score:.0f} out of 100."

    # Knee bend
    if avg_knee > 155:
        knee_line = f"Front knee averages {avg_knee:.0f} degrees — a bit straight. Aim for 130 to 140."
    elif avg_knee > 140:
        knee_line = f"Front knee at {avg_knee:.0f} degrees. Good base."
    else:
        knee_line = f"Nice knee bend at {avg_knee:.0f} degrees."
    if min_knee < 100:
        knee_line += f" Deepest bend was {min_knee:.0f} — try to keep it above 110 for balance."

    # Spine angle
    if avg_spine > 160:
        spine_line = f"Spine angle at {avg_spine:.0f} degrees. Well balanced."
    elif avg_spine > 150:
        spine_line = f"Spine around {avg_spine:.0f} degrees. Keep your head over the ball."
    else:
        spine_line = f"Spine at {avg_spine:.0f}. You are lunging forward — stay more upright."

    # Shot completion
    if total_shots > 0:
        pct = (complete_shots / total_shots) * 100
        if pct < 50:
            shot_line = (f"Shot completion was {pct:.0f} percent. "
                        f"Commit to your shots — a full swing is better than a half-hearted one.")
        else:
            shot_line = f"Shot completion at {pct:.0f} percent. Good commitment."
    else:
        shot_line = ""

    # Priorities
    priorities = report_data.get("priorities", [])
    priority_lines = []
    for p in priorities[:2]:
        area = p['area'].title().replace("_", " ")
        drill = p.get("drill", "")
        # Only use first sentence of drill
        drill_short = drill.split(".")[0] if drill else ""
        priority_lines.append(f"Focus on {area}. {drill_short}.")

    # ── Build concise script ──
    parts = [
        f"Welcome to your CREASE batting analysis.",

        head_line,
        knee_line,
        spine_line,
    ]

    if bat_line:
        parts.append(bat_line)

    parts.append(shot_line)

    if priority_lines:
        parts.append("Two things to work on.")
        parts.extend(priority_lines)

    parts.append(f"Session score {session_score:.0f} out of 100. Keep at it.")

    script = " ".join(parts)
    return script


def _get_duration_sec(path):
    """Get media duration in seconds using ffmpeg itself (more portable than ffprobe)."""
    ffmpeg = _ensure_ffmpeg()
    try:
        result = subprocess.run(
            [ffmpeg, "-i", path, "-f", "null", "-"],
            capture_output=True, text=True, timeout=30
        )
        # ffmpeg prints duration in stderr: "Duration: HH:MM:SS.MS"
        import re
        match = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", result.stderr)
        if match:
            h, m, s, ms = map(int, match.groups())
            return h * 3600 + m * 60 + s + ms / 100.0
    except Exception:
        pass
    return 0


def _estimate_section_timestamps(session_data, report_data, total_audio_sec):
    """Estimate start/end timestamps for each topic section in the voiceover.
    
    Uses character count ratio of each section and a constant speech rate
    to estimate when each topic will be discussed.
    """
    ss = session_data.get("session_summary", {})
    bs = session_data.get("bat_speed", {})
    shots = session_data.get("shot_summary", [])
    total_shots = len(shots)
    complete = len([s for s in shots if s.get("has_impact")])
    pct = (complete / total_shots * 100) if total_shots > 0 else 0
    head_score = ss.get("head_stability_score", 0)
    avg_knee = ss.get("avg_front_knee_angle", 0)
    avg_spine = ss.get("avg_spine_angle", 0)
    peak_speed = bs.get("peak_kmh", 0)
    swing_avg = bs.get("swing_avg_kmh", 0)
    session_score = report_data.get("session_score", 50)

    cal = bs.get("calibration", {})
    px_per_m = cal.get("px_per_m", None)
    head_mv_px = ss.get("avg_head_movement", 0)
    head_cm = head_mv_px / (px_per_m / 100.0) if px_per_m and px_per_m > 0 else None
    head_str = f"{head_cm:.1f} cm" if head_cm else f"{head_mv_px:.1f} px"

    # Build ordered list of topic sections  
    sections = []

    # Welcome (short)
    sections.append(("welcome", "", 1.5))

    # Head stability
    sections.append(("head", f"HEAD MOVEMENT  {head_str}", 3.5))

    # Knee bend
    sections.append(("knee", f"KNEE  {avg_knee:.0f} deg", 3.0))

    # Spine angle
    sections.append(("spine", f"SPINE  {avg_spine:.0f} deg", 2.5))

    # Bat speed
    if peak_speed > 20:
        sections.append(("bat_speed", f"BAT SPEED  {peak_speed:.0f} km/h peak", 3.0))

    # Shot completion
    if total_shots > 0:
        sections.append(("shot", f"SHOTS  {complete}/{total_shots} ({pct:.0f}%)", 3.0))

    # Session score
    sections.append(("score", f"SCORE  {session_score:.0f}/100", 2.0))

    # Calculate cumulative timestamps from section durations
    # Each section's visual starts at its cumulative time.
    # The visual appears until the NEXT section starts (or for min_duration).
    cursor = 2.0  # Leave 2s before first visual
    for i, (name, label, dur) in enumerate(sections):
        sections[i] = (name, label, dur, cursor, cursor + dur)
        cursor += dur

    return sections


def _add_topic_text_overlays(video_path, session_data, report_data,
                              audio_duration=None,
                              output_path=None):
    """Add topic-relevant text overlays to the video at timestamps
    matching the voiceover script, using ffmpeg drawtext filters.
    
    This replaces the freeze-frame approach with memory-efficient
    ffmpeg streaming overlays.
    """
    if audio_duration is None:
        audio_duration = 60.0

    # Get estimated timestamps
    sections = _estimate_section_timestamps(session_data, report_data, audio_duration)

    # Build ffmpeg filter: chain drawtext filters for each section
    filter_parts = []

    for name, label, duration, start_sec, end_sec in sections:
        if not label:
            continue
        # Short labels show at top centre
        y_pos = "h*0.12"
        fontsize = 28
        # Text is safe: no colons or quotes in overlay labels
        f_str = (
            f"drawtext=text='{label}'"
            f":fontsize={fontsize}"
            f":fontcolor=white"
            f":box=1:boxcolor=black@0.55:boxborderw=10"
            f":x=(w-text_w)/2"
            f":y={y_pos}"
            f":enable='between(t,{start_sec:.1f},{end_sec:.1f})'"
        )
        filter_parts.append(f_str)

    if not filter_parts:
        # No overlays needed
        return video_path

    # Chain filters: [0:v]drawtext1=...[v1]; [v1]drawtext2=...[v2]; ...
    chain = ""
    prev = "[0:v]"
    for i, fp in enumerate(filter_parts):
        label = f"[v{i}]" if i < len(filter_parts) - 1 else "[vout]"
        chain += f"{prev}{fp}{label};"
        prev = label
    chain = chain.rstrip(";")

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_overlay{ext}"

    ffmpeg = _ensure_ffmpeg()
    cmd = [
        ffmpeg, "-i", video_path,
        "-filter_complex", chain,
        "-map", "[vout]",
        "-c:v", "libx264",
        "-b:v", "1000k",  # Higher bitrate to preserve quality
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-y",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  Overlay ffmpeg error: {result.stderr[:300]}")
        return video_path

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Added {len(filter_parts)} topic overlays to video: {size_mb:.1f} MB")
    return output_path


def compress_and_add_audio(video_path, audio_path, output_path=None,
                           video_bitrate="500k", audio_bitrate="64k"):
    """Compress video with H.264 and overlay voiceover audio.

    Handles three cases:
    1. Audio longer than video  → freeze the last frame (tpad) so coach finishes speaking
    2. Video longer than audio  → let video play full length; audio ends naturally
    3. Roughly equal            → standard -shortest (trim minor excess)
    """
    ffmpeg = _ensure_ffmpeg()

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_final{ext}"

    # Get durations
    video_dur = _get_duration_sec(video_path)
    audio_dur = _get_duration_sec(audio_path)

    print(f"  Video: {video_dur:.0f}s, Audio: {audio_dur:.0f}s", end="")

    if audio_dur > video_dur + 1:
        extra = audio_dur - video_dur
        print(f"  -> Freezing last frame for {extra:.0f}s")
        cmd = [
            ffmpeg,
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={extra:.2f}[v]",
            "-map", "[v]",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-b:v", video_bitrate,
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-shortest",
            "-movflags", "+faststart",
            "-y",
            output_path,
        ]
    elif video_dur > audio_dur + 1:
        print(f"  -> Full video, audio ends naturally")
        # No -shortest: output = video length, audio goes silent when it ends
        cmd = [
            ffmpeg,
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-b:v", video_bitrate,
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-movflags", "+faststart",
            "-y",
            output_path,
        ]
    else:
        print(f"  -> Matching durations")
        cmd = [
            ffmpeg,
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-b:v", video_bitrate,
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-movflags", "+faststart",
            "-y",
            output_path,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  FFmpeg error: {result.stderr[:500]}")
        # Fallback: simple compress without audio
        fallback = output_path.replace("_coached.", "_noaudio.")
        cmd2 = [ffmpeg, "-i", video_path, "-c:v", "libx264", "-b:v", video_bitrate,
                "-preset", "medium", "-movflags", "+faststart", "-y", fallback]
        subprocess.run(cmd2, capture_output=True, text=True)
        return fallback

    out_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    in_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"  Done! {out_size_mb:.1f} MB (was {in_size_mb:.1f} MB, "
          f"{'compressed ' + str(int((1 - out_size_mb/in_size_mb)*100)) + '%' if in_size_mb > 0 else 'N/A'})")

    return output_path


def compress_video_only(video_path, output_path=None, video_bitrate="500k"):
    ffmpeg = _ensure_ffmpeg()
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_compressed{ext}"
    cmd = [ffmpeg, "-i", video_path, "-c:v", "libx264", "-b:v", video_bitrate,
           "-preset", "medium", "-movflags", "+faststart", "-y", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FFmpeg error: {result.stderr[:300]}")
        return video_path
    out_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    in_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"  Compressed: {out_size_mb:.1f} MB (was {in_size_mb:.1f} MB)")
    return output_path


def generate_full_video(video_path, session_data, report_data,
                        output_path=None, video_bitrate="500k",
                        voice="en-GB-RyanNeural",
                        coaching_script=None, player_context=None):
    """Full pipeline: script -> speech -> topic overlays -> compress + audio.

    Args:
        video_path: Path to input (already-annotated) video
        session_data: Full analysis result dict
        report_data: Coaching report dict
        output_path: Where to save the final video
        video_bitrate: Target bitrate (e.g. '500k')
        voice: edge-tts voice name
        coaching_script: Pre-generated script (if None, uses default)
        player_context: Dict with 'player_id', 'label', 'session_type', etc.
    """
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_coached{ext}"

    print("Step 1: Writing conversational coaching script...")
    if coaching_script:
        script = coaching_script
        print(f"  Using custom script ({len(coaching_script)} chars)")
    else:
        script = generate_coaching_script(session_data, report_data)

    print("Step 2: Converting to natural speech...")
    audio_fd, audio_path = tempfile.mkstemp(suffix=".mp3")
    os.close(audio_fd)
    text_to_speech(script, audio_path, voice=voice)

    # Get audio duration to time the overlays
    audio_dur = _get_duration_sec(audio_path)

    print("Step 3: Adding topic-relevant text overlays to video...")
    overlaid_path = _add_topic_text_overlays(
        video_path, session_data, report_data,
        audio_duration=audio_dur
    )

    print("Step 4: Compressing and adding audio...")
    result = compress_and_add_audio(
        overlaid_path, audio_path, output_path, video_bitrate=video_bitrate
    )

    # Cleanup temp files
    try:
        os.remove(audio_path)
        if overlaid_path != video_path and os.path.exists(overlaid_path):
            os.remove(overlaid_path)
    except:
        pass

    return result
