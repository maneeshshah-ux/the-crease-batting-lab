"""
Run the full analysis pipeline:
  1. Video analysis (pose, ball, bat, phase detection)
  2. Coaching report (benchmarks, priorities, session score)
  3. NATURAL VOICEOVER (edge-tts — British coaching voice)
  4. COMPRESSED VIDEO (H.264 at 500kbps, audio overlaid)
  5. PDF COACHING REPORT (annotated screenshots, charts, comparisons)

Set VOICEOVER=false or COMPRESS=false to skip those steps.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

from engine.report_generator import generate_report

from engine.analyser import BattingAnalyser
from engine.benchmarks import (
    get_bat_speed_benchmark,
    get_head_stability_assessment,
    get_knee_assessment,
    get_spine_assessment,
    BAT_SPEED_BENCHMARKS,
    PLAYER_BAT_SPEED,
    HEAD_STABILITY_PLAYERS,
)
from engine.voiceover import generate_full_video

video_path = "/Users/mac/Desktop/the CREASE/batting_analyser/Batting Videos/Dia.mp4"
output_dir = "/Users/mac/Desktop/the CREASE/batting_analyser/sessions"

REPORT = os.environ.get("REPORT", "true").lower() == "true"
VOICEOVER = os.environ.get("VOICEOVER", "true").lower() == "true"
COMPRESS = os.environ.get("COMPRESS", "true").lower() == "true"
VIDEO_BITRATE = os.environ.get("VIDEO_BITRATE", "500k")

print(f"=== THE CREASE BATTING LAB — ANALYSIS RUN ===")
print(f"Video: {video_path}")
print(f"Voiceover: {'ON' if VOICEOVER else 'OFF'} | "
      f"Compress: {'ON' if COMPRESS else 'OFF'} @ {VIDEO_BITRATE} | "
      f"Report: {'ON' if REPORT else 'OFF'}")
print()

def progress(f, t, s):
    if f % 300 == 0:
        pct = f / t * 100 if t else 0
        print(f"  [{pct:5.1f}%] frame {f}/{t}")

print("Creating analyser...")
a = BattingAnalyser(batting_hand="right", ball_color="red")

t0 = time.time()
result = a.analyse_video(video_path, output_dir=output_dir,
                         generate_video=True, progress_callback=progress)
t1 = time.time()

print(f"\nDone in {(t1-t0)/60:.1f} min")

# ====================== COACHING REPORT ======================
print("\n" + "=" * 60)
print("  COACHING REPORT")
print("=" * 60)

shots = result.get("shot_summary", [])
ss = result.get("session_summary", {})

# --- BAT SPEED ---
bat_speed = result.get("bat_speed", {})
if bat_speed and bat_speed.get("kmh_estimated"):
    avg_kmh = bat_speed.get("speed_kmh", 0)        # overall avg (includes shuffle)
    swing_avg_kmh = bat_speed.get("swing_avg_kmh", avg_kmh)  # avg during swings only
    peak_kmh = bat_speed.get("peak_kmh", 0)
    cal_method = bat_speed.get("calibration", {}).get("method", "unknown")
    print(f"\n  BAT SPEED: {swing_avg_kmh:.0f} km/h avg (swing) | PEAK: {peak_kmh:.0f} km/h")
    print(f"   Overall avg (incl. stance/shuffle): {avg_kmh:.0f} km/h")
    print(f"   Calibration: {cal_method}")

    bench = get_bat_speed_benchmark(swing_avg_kmh, peak_kmh)
    print(f"   Level: {bench['level_label']}")
    print(f"   {bench['comparison_text']}")

    if bench['nearest_player']:
        pname, pdata = bench['nearest_player']
        print(f"   Nearest: {pname} ({pdata['peak_kmh']} km/h peak)")
        print(f"   \"{pdata['style']}\"")

else:
    print(f"\n  BAT SPEED: {bat_speed.get('speed_px_per_sec', 0):.0f} px/sec (uncalibrated)")
    cal = bat_speed.get("calibration")
    if cal:
        print(f"   Calibration: {cal.get('method', 'unknown')}")

# --- HEAD STABILITY ---
head_score = ss.get("head_stability_score", 0)
head_assessment = get_head_stability_assessment(head_score)
print(f"\n  HEAD STABILITY: {head_score:.0f}/100 ({head_assessment['level'].replace('_', ' ').title()})")
print(f"   {head_assessment['level_note']}")
if head_assessment['nearest_player']:
    pname, pdata = head_assessment['nearest_player']
    diff = pdata['score'] - head_score
    print(f"   Reference: {pname} scores {pdata['score']}/100 ({'+' if diff > 0 else ''}{diff:.0f} from you)")

# --- SHOT COMPLETENESS ---
complete_shots_list = [s for s in shots if s.get("has_impact")]
partial = [s for s in shots if not s.get("has_impact")]
total = len(shots)
completion_pct = (len(complete_shots_list) / total * 100) if total > 0 else 0
print(f"\n  SHOT COMPLETION: {completion_pct:.0f}% ({len(complete_shots_list)} complete / {len(partial)} partial)")

# --- FRONT KNEE ---
avg_knee = ss.get("avg_front_knee_angle", 0)
min_knee = ss.get("min_front_knee_angle", 0)
knee_assessment = get_knee_assessment(avg_knee)
knee_deepest = get_knee_assessment(min_knee)
print(f"\n  FRONT KNEE: Avg {avg_knee:.0f}° ({knee_assessment['level'].replace('_', ' ').title()})")
print(f"   {knee_assessment['note']}")
print(f"   Deepest bend: {min_knee:.0f}° ({knee_deepest['note']})")

# --- SPINE ANGLE ---
avg_spine = ss.get("avg_spine_angle", 0)
min_spine = ss.get("min_spine_angle", 0)
spine_assessment = get_spine_assessment(avg_spine)
print(f"\n  SPINE LEAN: Avg {avg_spine:.0f}° ({spine_assessment['level'].title()})")
print(f"   {spine_assessment['note']}")
if min_spine < 155:
    worst_spine = get_spine_assessment(min_spine)
    print(f"   Worst lean: {min_spine:.0f}° — {worst_spine['note']}")

# --- TOP 3 PRIORITIES ---
print(f"\n" + "=" * 60)
print("  TOP 3 PRIORITIES")
print("=" * 60)

priorities = []
if head_score < 60:
    priorities.append({
        "rank": 1,
        "area": "HEAD STABILITY",
        "why": f"Your head moves {head_assessment['avg_movement_px']:.1f}px on average. "
               f"For reference, Kohli's head barely moves 1-2px through the shot.",
        "drill": "Head-Still Drill: Place a bottle cap on your head while shadow batting. "
                 "Play 50 forward defensive strokes without it falling off.",
        "target": f"Score 60+ (currently {head_score:.0f})",
    })

if completion_pct < 50:
    priorities.append({
        "rank": len(priorities) + 1,
        "area": "SHOT COMMITMENT",
        "why": f"Only {len(complete_shots_list)} of {total} shots were completed. "
               f"You're lifting the bat but not committing to the shot {len(partial)} times.",
        "drill": "Commitment Drill: Commit fully to every shot you start. "
                 "If you lift the bat, complete the swing - partial swings build bad habits.",
        "target": "Complete 70%+ of shots",
    })

if avg_knee > 155:
    priorities.append({
        "rank": len(priorities) + 1,
        "area": "KNEE FLEX",
        "why": f"Your front knee averages {avg_knee:.0f}° — quite straight. "
               f"Bending deeper ({avg_knee-15:.0f}°-{avg_knee-10:.0f}°) would help you reach the ball better.",
        "drill": "Knee-Tap Drill: Mark a spot 12 inches down the pitch. "
                 "Every shot, front foot to that spot with knee bent to 130°.",
        "target": f"Average knee bend of 130-145° (currently {avg_knee:.0f}°)",
    })

if avg_spine < 155:
    priorities.append({
        "rank": len(priorities) + 1,
        "area": "POSTURE",
        "why": f"You're leaning forward ({avg_spine:.0f}°). "
               f"Head falling past the front knee on {len([s for s in shots if s.get('has_impact')])} shots.",
        "drill": "Corridor Drill: Place a second set of stumps 4ft down. "
                 "Reach with your front foot only, not your head.",
        "target": f"Spine angle above 155° on all shots (currently avg {avg_spine:.0f}°)",
    })

for p in priorities:
    print(f"\n  #{p['rank']}: {p['area']}")
    print(f"    Why: {p['why']}")
    print(f"    Drill: {p['drill']}")
    print(f"    Target: {p['target']}")

# --- SESSION SCORE ---
peak_kmh = bat_speed.get("peak_kmh", 0) if bat_speed and bat_speed.get("kmh_estimated") else 0
session_score = min(100, max(0,
    (head_score * 0.35) +
    (completion_pct * 0.25) +
    (min(100, (avg_knee - 100) * 1.5) * 0.15) +
    (min(100, max(0, 180 - avg_spine) * 3) * 0.15) +
    (min(100, max(0, peak_kmh - 60)) * 0.10)
))

print(f"\n" + "=" * 60)
print(f"  SESSION SCORE: {session_score:.0f}/100")
print("=" * 60)

# Save report
report = {
    "bat_speed": {
        "avg_kmh": avg_kmh if bat_speed and bat_speed.get("kmh_estimated") else None,
        "peak_kmh": peak_kmh if bat_speed and bat_speed.get("kmh_estimated") else None,
        "calibration": bat_speed.get("calibration") if bat_speed else None,
        "benchmark": bench if bat_speed and bat_speed.get("kmh_estimated") else None,
    },
    "head_stability": {
        "score": head_score,
        "assessment": head_assessment,
    },
    "shot_completion": {
        "total": total,
        "complete": len(complete_shots_list),
        "partial": len(partial),
        "completion_pct": completion_pct,
    },
    "front_knee": {
        "avg": avg_knee,
        "min": min_knee,
        "assessment": knee_assessment,
    },
    "spine": {
        "avg": avg_spine,
        "min": min_spine,
        "assessment": spine_assessment,
    },
    "priorities": priorities,
    "session_score": session_score,
}

report_path = os.path.join(output_dir, f"coaching_report_{result.get('session_id', 'unknown')}.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nReport saved to: {report_path}")
print(f"Analysis JSON: {result.get('result_path', 'N/A')}")
print(f"Analysis video: {result.get('output_video_path', 'N/A')}")

# ====================== VOICEOVER + COMPRESSION ======================
orig_video = result.get("output_video_path")
if orig_video and os.path.exists(orig_video):
    orig_size_mb = os.path.getsize(orig_video) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"  VOICEOVER & COMPRESSION")
    print(f"{'=' * 60}")
    print(f"  Original video: {orig_size_mb:.1f} MB")

    if VOICEOVER and COMPRESS:
        # Full pipeline: voiceover + compression
        # Try Ryan (British) for a natural coaching voice; fallback to William
        VOICE = os.environ.get("VOICE", "en-GB-RyanNeural")
        coached_video = generate_full_video(
            video_path=orig_video,
            session_data=result,
            report_data=report,
            output_path=orig_video.replace(".mp4", "_coached.mp4"),
            video_bitrate=VIDEO_BITRATE,
            voice=VOICE,
        )
        if coached_video and os.path.exists(coached_video):
            final_size = os.path.getsize(coached_video) / (1024 * 1024)
            print(f"  Final video: {coached_video}")
            print(f"  Final size: {final_size:.1f} MB ({orig_size_mb/final_size:.1f}x smaller)")
    elif COMPRESS:
        # Compression only (no voiceover)
        from engine.voiceover import compress_video_only
        compressed = compress_video_only(
            orig_video,
            output_path=orig_video.replace(".mp4", "_compressed.mp4"),
            video_bitrate=VIDEO_BITRATE,
        )
        if compressed and os.path.exists(compressed):
            final_size = os.path.getsize(compressed) / (1024 * 1024)
            print(f"  Compressed: {final_size:.1f} MB ({orig_size_mb/final_size:.1f}x smaller)")
    else:
        print("  Skipped (VOICEOVER=false and COMPRESS=false)")
else:
    print("\n  No video output to process.")

# ====================== PDF REPORT ======================
if REPORT:
    print(f"\n{'=' * 60}")
    print(f"  PDF COACHING REPORT")
    print(f"{'=' * 60}")
    tagged_video = orig_video if orig_video and os.path.exists(orig_video) else None
    if tagged_video:
        pdf_path = orig_video.replace(".mp4", "_report.pdf")
        try:
            generate_report(
                session_data=result,
                report_data=report,
                analysis_video_path=tagged_video,
                output_path=pdf_path,
            )
        except Exception as e:
            print(f"  Report generation failed: {e}")
    else:
        print("  No analysis video available for screenshots.")
