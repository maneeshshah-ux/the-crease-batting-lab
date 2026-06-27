"""
Batch analyser — runs the full CREASE pipeline on multiple videos.
Processes each video sequentially, generating:
  1. Analysis JSON + video (pose, phases, ball, bat)
  2. Coaching report (benchmarks, priorities, session score)
  3. Voiceover + compressed video
  4. PDF coaching report

Usage:
  python batch_run.py                          # runs all 6 session videos
  python batch_run.py --videos 1,3,5           # runs specific videos by index
  python batch_run.py --skip-analysis          # skips MediaPipe (uses existing JSON)

Output naming: session_<NN>_<HHMMSS>_data.json / _coached.mp4 / _report.pdf
"""

import sys, os, json, time, argparse, re

sys.path.insert(0, os.path.dirname(__file__))

from engine.analyser import BattingAnalyser
from engine.benchmarks import (
    get_bat_speed_benchmark,
    get_head_stability_assessment,
    get_knee_assessment,
    get_spine_assessment,
)
from engine.voiceover import generate_full_video
from engine.report_generator import generate_report
from engine.player_profiler import extract_stance_signature, signature_to_vector, cosine_similarity
from engine.player_registry import (
    find_or_create_player,
    compute_session_metrics,
    list_players,
    _load_registry,
)
from engine.longitudinal_feedback import (
    analyze_trends,
    generate_voiceover_script,
)

# ── Configuration ──
VIDEO_DIR = os.path.join(os.path.dirname(__file__), "Batting Videos")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "sessions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# (index, time_tag, filename, duration_label)
SESSION_VIDEOS = [
    ("01", "105456", "WhatsApp Video 2026-06-22 at 10.54.56.mp4", "01:42"),
    ("02", "105509", "WhatsApp Video 2026-06-22 at 10.55.09.mp4", "01:23"),
    ("03", "105616", "WhatsApp Video 2026-06-22 at 10.56.16.mp4", "01:06"),
    ("04", "105617", "WhatsApp Video 2026-06-22 at 10.56.17.mp4", "00:43"),
    ("05", "105618", "WhatsApp Video 2026-06-22 at 10.56.18.mp4", "01:44"),
    ("06", "105619", "WhatsApp Video 2026-06-22 at 10.56.19.mp4", "01:14"),
]

VOICE = os.environ.get("VOICE", "en-GB-RyanNeural")
VIDEO_BITRATE = os.environ.get("VIDEO_BITRATE", "500k")


def _sanitise(text: str) -> str:
    """Remove or replace non-ASCII characters for ffmpeg drawtext."""
    return text.encode("ascii", errors="replace").decode("ascii")


def _build_player_history(player, current_session_id="", current_flags=None):
    """Build a player_history dict for the PDF report from registry data."""
    if not player:
        return None

    current_flags = current_flags or []
    session_ids = player.get("session_ids", [])
    hist = player.get("historical_metrics", {})

    # Build session list
    sessions_out = []
    for sid in session_ids:
        # Try to extract metrics for this session from the flat historical_metrics
        # Since we store vals as lists per metric, we need to find the index
        idx = session_ids.index(sid) if sid in session_ids else -1
        is_current = sid == current_session_id

        if idx >= 0:
            def _get_metric(key, i):
                vals = hist.get(key, [])
                return vals[i] if i < len(vals) else 0

            # Parse date from session_id format "session_NN_HHMMSS"
            if len(sid) == 17 and sid.startswith("session_"):
                time_part = sid[11:]  # e.g. "105509"
                h, m = time_part[:2], time_part[2:4]
                date_str = f"Jun 22 {h}:{m}"
            else:
                date_str = sid[:10] if len(sid) >= 10 else sid

            session_entry = {
                "session_id": sid,
                "date": date_str,
                "head_stability_score": _get_metric("head_stability_score", idx),
                "avg_front_knee_angle": _get_metric("avg_front_knee_angle", idx),
                "avg_spine_angle": _get_metric("avg_spine_angle", idx),
                "bat_speed_avg_kmh": _get_metric("bat_speed_avg_kmh", idx),
                "num_shots": _get_metric("num_shots", idx),
                "shot_completion_pct": _get_metric("shot_completion_pct", idx),
                "flags": current_flags if is_current else [],
            }
            sessions_out.append(session_entry)

    return {
        "player_id": player.get("id", ""),
        "label": player.get("label", player.get("name", f"Player {player.get('id', '?')}")),
        "n_sessions": len(session_ids),
        "sessions": sessions_out,
    }


def process_video(video_path, session_prefix, skip_analysis=False):
    """
    Run the full pipeline on a single video, WITH player recognition + longitudinal feedback.

    Args:
        video_path: path to source video
        session_prefix: e.g. "session_01_105456"
        skip_analysis: if True, load most recent analysis JSON instead of running MediaPipe
    """
    video_name = os.path.basename(video_path)
    print(f"\n{'=' * 70}")
    print(f"  [{session_prefix}] {video_name}")
    print(f"{'=' * 70}\n")

    t_start = time.time()

    # Structured output paths
    json_out = os.path.join(OUTPUT_DIR, f"{session_prefix}_data.json")
    analysis_video_out = os.path.join(OUTPUT_DIR, f"{session_prefix}_analysis.mp4")
    coached_video_out = os.path.join(OUTPUT_DIR, f"{session_prefix}_coached.mp4")
    pdf_out = os.path.join(OUTPUT_DIR, f"{session_prefix}_report.pdf")

    # ── Step 1: Analysis ──
    if not skip_analysis:
        print("  Step 1/4: Video analysis (pose, phases, bat, ball)...")
        a = BattingAnalyser(batting_hand="right", ball_color="red")

        def progress(f, t, s):
            if f % 300 == 0:
                pct = f / t * 100 if t else 0
                print(f"    [{pct:5.1f}%] frame {f}/{t}")

        result = a.analyse_video(
            video_path,
            output_dir=OUTPUT_DIR,
            generate_video=True,
            progress_callback=progress,
        )
        analysis_time = (time.time() - t_start) / 60
        print(f"    Done in {analysis_time:.1f} min")

        # Rename outputs to structured names
        orig_analysis_video = result.get("output_video_path", "")
        if orig_analysis_video and os.path.exists(orig_analysis_video):
            os.rename(orig_analysis_video, analysis_video_out)
            result["output_video_path"] = analysis_video_out

        orig_json = result.get("result_path", "")
        if orig_json and os.path.exists(orig_json):
            os.rename(orig_json, json_out)
            result["result_path"] = json_out
        else:
            with open(json_out, "w") as f:
                json.dump(result, f, indent=2, default=str)
    else:
        # Load existing structured JSON
        if os.path.exists(json_out):
            json_path = json_out
        else:
            json_files = [
                f for f in os.listdir(OUTPUT_DIR)
                if f.endswith("_data.json") and not f.startswith("coaching_")
            ]
            if not json_files:
                json_files = [
                    f for f in os.listdir(OUTPUT_DIR)
                    if f.endswith(".json") and not f.startswith("coaching_")
                ]
            if not json_files:
                print("  ERROR: No existing analysis JSON found. Run without --skip-analysis.")
                return False
            json_path = os.path.join(
                OUTPUT_DIR,
                max(json_files, key=lambda f: os.path.getmtime(os.path.join(OUTPUT_DIR, f))),
            )

        print(f"  Step 1/4: Skipping analysis, loading {os.path.basename(json_path)}...")
        with open(json_path) as f:
            result = json.load(f)
        print(f"    Loaded session: {result.get('session_id', 'unknown')}")

        if not os.path.exists(analysis_video_out):
            orig_vid = result.get("output_video_path", "")
            if orig_vid and os.path.exists(orig_vid):
                os.rename(orig_vid, analysis_video_out)
                result["output_video_path"] = analysis_video_out

    # ── Step 2: Coaching Report + Player Recognition ──
    print("\n  Step 2/4: Coaching report & player profiling...")
    shots = result.get("shot_summary", [])
    ss = result.get("session_summary", {})
    bat_speed = result.get("bat_speed", {})

    total_shots = len(shots)
    complete_shots = len([s for s in shots if s.get("has_impact")])
    completion_pct = (complete_shots / total_shots * 100) if total_shots > 0 else 0

    head_score = ss.get("head_stability_score", 0)
    avg_knee = ss.get("avg_front_knee_angle", 0)
    min_knee = ss.get("min_front_knee_angle", 0)
    avg_spine = ss.get("avg_spine_angle", 0)
    min_spine = ss.get("min_spine_angle", 0)
    swing_avg_kmh = bat_speed.get("swing_avg_kmh", 0)
    peak_kmh = bat_speed.get("peak_kmh", 0)

    head_assessment = get_head_stability_assessment(head_score)
    knee_assessment = get_knee_assessment(avg_knee)
    spine_assessment = get_spine_assessment(avg_spine)

    priorities = []
    if head_score < 60:
        drift_cm = head_assessment.get("avg_movement_px", 0)
        priorities.append({
            "rank": 1, "area": "HEAD STABILITY",
            "drill": _sanitise(
                f"Head-Still Drill: Place a bottle cap on your head while shadow batting. "
                f"Your head drifts {drift_cm:.1f}cm on average. "
                f"Play 50 defensive strokes without it falling off. Target: score 60+ (currently {head_score:.0f})."
            ),
        })
    if completion_pct < 50:
        priorities.append({
            "rank": len(priorities) + 1, "area": "SHOT COMMITMENT",
            "drill": _sanitise(
                f"Only {complete_shots} of {total_shots} shots were completed ({completion_pct:.0f}%). "
                f"Commit fully to every shot you start. Target: 70%+ completion."
            ),
        })
    if avg_knee > 155:
        priorities.append({
            "rank": len(priorities) + 1, "area": "KNEE FLEX",
            "drill": _sanitise(
                f"Your front knee averages {avg_knee:.0f} degrees ({knee_assessment['level'].replace('_', ' ').title()}). "
                f"Bend deeper to 130-145 degrees. "
                f"Knee-Tap Drill: front foot to a marked spot 12 inches down the pitch with knee at 130 degrees."
            ),
        })
    if avg_spine < 155:
        priorities.append({
            "rank": len(priorities) + 1, "area": "POSTURE",
            "drill": _sanitise(
                f"Your spine lean averages {avg_spine:.0f} degrees ({spine_assessment['note']}). "
                f"Keep your head above your front knee. "
                f"Corridor Drill: place a second set of stumps 4 feet down and reach with your front foot only."
            ),
        })

    session_score = min(100, max(0,
        (head_score * 0.35) +
        (completion_pct * 0.25) +
        (min(100, (avg_knee - 100) * 1.5) * 0.15) +
        (min(100, max(0, 180 - avg_spine) * 3) * 0.15) +
        (min(100, max(0, peak_kmh - 60)) * 0.10)
    ))

    report_data = {
        "bat_speed": {"avg_kmh": swing_avg_kmh, "peak_kmh": peak_kmh},
        "head_stability": {"score": head_score, "assessment": head_assessment},
        "shot_completion": {
            "total": total_shots, "complete": complete_shots,
            "completion_pct": completion_pct,
        },
        "front_knee": {"avg": avg_knee, "min": min_knee, "assessment": knee_assessment},
        "spine": {"avg": avg_spine, "min": min_spine, "assessment": spine_assessment},
        "priorities": priorities,
        "session_score": session_score,
    }

    print(f"    Score: {session_score:.0f}/100"
          f"  |  Shots: {complete_shots}/{total_shots} ({completion_pct:.0f}%)"
          f"  |  Head: {head_score:.0f}  Knee: {avg_knee:.0f}  Spine: {avg_spine:.0f}"
          f"  |  Bat: {swing_avg_kmh:.0f} km/h")

    # ── Player Recognition ──
    print("\n  Player profiling...")
    stance_sig = result.get("stance_signature", {})
    if stance_sig and stance_sig.get("_confidence", 0) >= 0.3:
        print(f"    Stance confidence: {stance_sig['_confidence']:.2f} "
              f"(from {stance_sig.get('_n_stance_frames', 0)} stance frames, "
              f"{stance_sig.get('_n_backlift_frames', 0)} backlift frames)")

        # Compute session metrics for the registry
        session_metrics = compute_session_metrics(result, report_data)

        # Find or create player
        player, is_new = find_or_create_player(
            stance_sig, session_prefix,
            session_metrics=session_metrics,
            match_threshold=0.80,  # slightly lower for 7-feature matching
        )
        player_id = player.get("id", "?")
        player_label = player.get("label", player.get("name", f"Player {player_id}"))
        n_sessions = len(player.get("session_ids", []))

        if is_new:
            print(f"    New player registered: {player_label} (ID: {player_id})")
        else:
            print(f"    Returning player: {player_label} (ID: {player_id}, {n_sessions} sessions)")

        # ── Longitudinal Feedback ──
        hist = player.get("historical_metrics", {})
        current_for_trends = session_metrics

        # Check if we have multiple sessions for history
        has_history = any(len(v) > 1 for v in hist.values())
        long_feedback = analyze_trends(current_for_trends, hist, session_prefix)

        # Session type flag
        session_flags = []
        if long_feedback.get("fatigue_flag"):
            session_flags.append("fatigue")
        elif long_feedback.get("session_type") == "improvement":
            session_flags.append("improvement")

        # Generate custom voiceover script
        voiceover_ctx = generate_voiceover_script(
            current_for_trends, hist, report_data, session_prefix
        )
        custom_script = voiceover_ctx["script"]
        has_history = voiceover_ctx["has_history"]

        print(f"    Session type: {voiceover_ctx['session_type']}")
        if long_feedback.get("fatigue_flag"):
            print(f"    ⚠ Fatigue/off-day detected: {long_feedback.get('fatigue_detail', '')[:80]}...")
        print(f"    Voiceover: {len(custom_script)} chars ({'with history' if has_history else 'baseline'})")

        # Build player history for PDF
        pdf_player_history = _build_player_history(
            player, current_session_id=session_prefix,
            current_flags=session_flags,
        )

        player_context = {
            "player_id": player_id,
            "label": player_label,
            "session_type": voiceover_ctx["session_type"],
            "is_new": is_new,
            "n_sessions": n_sessions,
        }
    else:
        print(f"    Stance confidence too low ({stance_sig.get('_confidence', 0):.2f}) — "
              f"skipping player recognition")
        custom_script = None
        player_context = None
        pdf_player_history = None
        has_history = False
        session_flags = []

    # ── Step 3: Voiceover + Video Compression ──
    print("\n  Step 3/4: Voiceover & video compression...")
    orig_analysis = analysis_video_out if os.path.exists(analysis_video_out) else result.get("output_video_path", "")

    if orig_analysis and os.path.exists(orig_analysis):
        orig_size = os.path.getsize(orig_analysis) / (1024 * 1024)
        coached = generate_full_video(
            video_path=orig_analysis,
            session_data=result,
            report_data=report_data,
            output_path=coached_video_out,
            video_bitrate=VIDEO_BITRATE,
            voice=VOICE,
            coaching_script=custom_script,  # longitudinal-aware script
            player_context=player_context,
        )
        if coached and os.path.exists(coached):
            final_mb = os.path.getsize(coached) / (1024 * 1024)
            print(f"    Coached video: {final_mb:.1f} MB (was {orig_size:.1f} MB)")
        else:
            print(f"    Voiceover failed — check logs above.")
        vid_for_pdf = orig_analysis
    else:
        print("    No analysis video found — skipping voiceover.")
        vid_for_pdf = video_path if os.path.exists(video_path) else None

    # ── Step 4: PDF Report ──
    print("\n  Step 4/4: PDF coaching report...")
    if vid_for_pdf and os.path.exists(vid_for_pdf):
        try:
            generate_report(
                session_data=result,
                report_data=report_data,
                analysis_video_path=vid_for_pdf,
                output_path=pdf_out,
                player_history=pdf_player_history,
            )
            pdf_size = os.path.getsize(pdf_out) / 1024
            print(f"    PDF: {os.path.basename(pdf_out)} ({pdf_size:.0f} KB)")
        except Exception as e:
            print(f"    PDF failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("    No video available for annotated frames — skipping PDF.")

    # ── Save coaching report JSON ──
    report_json_out = os.path.join(OUTPUT_DIR, f"{session_prefix}_report.json")
    with open(report_json_out, "w") as f:
        json.dump(report_data, f, indent=2, default=str)
    print(f"    Report JSON: {os.path.basename(report_json_out)}")

    elapsed = (time.time() - t_start) / 60
    print(f"\n  [{session_prefix}] Done in {elapsed:.1f} min")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Batch run CREASE analysis on multiple videos"
    )
    parser.add_argument(
        "--videos", type=str, default=None,
        help="Comma-separated video indices (e.g. 1,3,5). Default: all 6",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip MediaPipe analysis (reuse existing JSON)",
    )
    parser.add_argument(
        "--start-from", type=int, default=1,
        help="Start from video index N (1-based)",
    )
    args = parser.parse_args()

    # Select videos from the list
    if args.videos:
        indices = [int(i.strip()) for i in args.videos.split(",")]
        selected = [
            SESSION_VIDEOS[i - 1] for i in indices if 1 <= i <= len(SESSION_VIDEOS)
        ]
    else:
        selected = SESSION_VIDEOS[args.start_from - 1:]

    print(f"\nTHE CREASE — Batch Processor")
    print(f"  Sessions to process: {len(selected)}")
    for idx, tag, fname, dur in selected:
        print(f"    session_{idx}_{tag}  ({dur})  {fname}")
    print()

    successful = 0
    for idx, tag, fname, dur in selected:
        vpath = os.path.join(VIDEO_DIR, fname)
        prefix = f"session_{idx}_{tag}"

        if not os.path.exists(vpath):
            print(f"\n  WARNING: Video not found — {vpath}")
            continue
        try:
            ok = process_video(vpath, prefix, skip_analysis=args.skip_analysis)
            if ok:
                successful += 1
        except Exception as e:
            print(f"\n  FAILED: {prefix} — {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"  Batch complete: {successful}/{len(selected)} sessions processed successfully")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
