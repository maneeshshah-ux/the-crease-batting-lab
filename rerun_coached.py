"""
Quick re-run: voiceover + compression + PDF using existing analysis data.
Skips the 4-minute MediaPipe video analysis.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

from engine.voiceover import generate_full_video
from engine.report_generator import generate_report
from engine.benchmarks import (
    get_bat_speed_benchmark, get_head_stability_assessment,
    get_knee_assessment, get_spine_assessment,
)

SESSION_DIR = "/Users/mac/Desktop/the CREASE/batting_analyser/sessions"
JSON_PATH = os.path.join(SESSION_DIR, "analysis_4a21d06f.json")
VIDEO_PATH = os.path.join(SESSION_DIR, "analysis_4a21d06f.mp4")

print("=== THE CREASE — VOICEOVER + PDF RE-RUN ===")
print(f"Loading analysis: {JSON_PATH}")

with open(JSON_PATH) as f:
    result = json.load(f)

shots = result.get("shot_summary", [])
ss = result.get("session_summary", {})
bat_speed = result.get("bat_speed", {})
head_score = ss.get("head_stability_score", 0)
avg_kmh = bat_speed.get("speed_kmh", 0)
swing_avg_kmh = bat_speed.get("swing_avg_kmh", avg_kmh)
peak_kmh = bat_speed.get("peak_kmh", 0)
total = len(shots)
complete = len([s for s in shots if s.get("has_impact")])
completion_pct = (complete / total * 100) if total > 0 else 0
avg_knee = ss.get("avg_front_knee_angle", 154)
avg_spine = ss.get("avg_spine_angle", 166)
head_assessment = get_head_stability_assessment(head_score)

bench = get_bat_speed_benchmark(swing_avg_kmh, peak_kmh)
knee_assessment = get_knee_assessment(avg_knee)
spine_assessment = get_spine_assessment(avg_spine)

# Build report data (same logic as run_full.py)
priorities = []
if head_score < 60:
    priorities.append({
        "rank": 1, "area": "HEAD STABILITY",
        "why": f"Your head moves {head_assessment['avg_movement_px']:.1f}px on average. For reference, Kohli barely moves 1-2px.",
        "drill": "Head-Still Drill: Place a bottle cap on your head while shadow batting. Play 50 forward defensive strokes without it falling off.",
        "target": f"Score 60+ (currently {head_score:.0f})",
    })
if completion_pct < 50:
    priorities.append({
        "rank": len(priorities) + 1, "area": "SHOT COMMITMENT",
        "why": f"Only {complete} of {total} shots completed. You lift the bat without swinging {total-complete} times.",
        "drill": "Commitment Drill: Commit fully to every shot you start. If you lift the bat, complete the swing - partial swings build bad habits.",
        "target": "Complete 70%+ of shots",
    })
if avg_knee > 155:
    priorities.append({
        "rank": len(priorities) + 1, "area": "KNEE FLEX",
        "why": f"Front knee avg {avg_knee:.0f} deg - quite straight. Bending deeper would help reach the ball.",
        "drill": "Knee-Tap Drill: Mark a spot 12 inches down the pitch. Every shot, front foot to that spot with knee bent.",
        "target": f"Average knee bend 130-145 deg (currently {avg_knee:.0f} deg)",
    })
if avg_spine < 155:
    priorities.append({
        "rank": len(priorities) + 1, "area": "POSTURE",
        "why": f"Leaning forward ({avg_spine:.0f} deg). Head falling past front knee.",
        "drill": "Corridor Drill: Place a second set of stumps 4ft down. Reach with front foot only, not your head.",
        "target": f"Spine angle above 155 deg (currently avg {avg_spine:.0f} deg)",
    })

session_score = min(100, max(0,
    (head_score * 0.35) + (completion_pct * 0.25) +
    (min(100, (avg_knee - 100) * 1.5) * 0.15) +
    (min(100, max(0, 180 - avg_spine) * 3) * 0.15) +
    (min(100, max(0, peak_kmh - 60)) * 0.10)
))

report = {
    "bat_speed": {"avg_kmh": avg_kmh, "peak_kmh": peak_kmh, "benchmark": bench},
    "head_stability": {"score": head_score, "assessment": head_assessment},
    "shot_completion": {"total": total, "complete": complete, "completion_pct": completion_pct},
    "front_knee": {"avg": avg_knee, "assessment": knee_assessment},
    "spine": {"avg": avg_spine, "assessment": spine_assessment},
    "priorities": priorities,
    "session_score": session_score,
}

VOICE = os.environ.get("VOICE", "en-GB-RyanNeural")
VIDEO_BITRATE = os.environ.get("VIDEO_BITRATE", "500k")

# Step 1: Voiceover + compression
print(f"\n--- VOICEOVER & COMPRESSION ---")
coached = generate_full_video(
    video_path=VIDEO_PATH,
    session_data=result,
    report_data=report,
    output_path=VIDEO_PATH.replace(".mp4", "_coached.mp4"),
    video_bitrate=VIDEO_BITRATE,
    voice=VOICE,
)
if coached and os.path.exists(coached):
    mb = os.path.getsize(coached) / (1024 * 1024)
    print(f"Coached: {coached} ({mb:.1f} MB)")

# Step 2: PDF report
print(f"\n--- PDF COACHING REPORT ---")
pdf_path = VIDEO_PATH.replace(".mp4", "_report.pdf")
try:
    generate_report(
        session_data=result,
        report_data=report,
        analysis_video_path=VIDEO_PATH,
        output_path=pdf_path,
    )
    print(f"PDF: {pdf_path}")
except Exception as e:
    print(f"PDF failed: {e}")

print(f"\nDone.")
