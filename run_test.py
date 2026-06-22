"""Quick test script to run analysis and print shot summary."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

from engine.analyser import BattingAnalyser

video_path = "/Users/mac/Desktop/the CREASE/batting_analyser/Batting Videos/Dia.mp4"
output_dir = "/Users/mac/Desktop/the CREASE/batting_analyser/sessions"

print(f"Analysing: {video_path}")
print(f"Output dir: {output_dir}")
print(f"File size: {os.path.getsize(video_path) / 1024 / 1024:.1f} MB")
print()

def progress(f, t, s):
    if f % 300 == 0:
        pct = f / t * 100 if t else 0
        print(f"  [{pct:5.1f}%] frame {f}/{t} — {s}")

print("Creating analyser...")
a = BattingAnalyser(batting_hand="right", ball_color="red")

t0 = time.time()
result = a.analyse_video(video_path, output_dir=output_dir, generate_video=False, progress_callback=progress)
t1 = time.time()

elapsed = t1 - t0
print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"Result JSON: {result.get('result_path', 'N/A')}")

print(f"\n=== SHOT SUMMARY ===")
print(f"Total shots: {result.get('num_shots_detected', 0)}")
print(f"Video: {result.get('duration_sec', 0)}s, {result.get('total_frames', 0)} frames")
print()

shots = result.get("shot_summary", [])
for s in shots:
    phases = s.get("phases", [])
    print(f"  Shot #{s['shot_number']:2d}: frames {s['start_frame']:4d}-{s['end_frame']:4d}  "
          f"({s['duration_frames']:3d} frames, {s['duration_sec']:5.2f}s)  "
          f"phases: {','.join(str(p) for p in phases)}"
          f"{'  *** HAS IMPACT ***' if s.get('has_impact') else ''}")

print(f"\n=== SESSION SUMMARY ===")
ss = result.get("session_summary", {})
for k, v in sorted(ss.items()):
    if isinstance(v, float):
        print(f"  {k}: {v:.2f}")
    else:
        print(f"  {k}: {v}")

print(f"\n=== COACHING TIPS ({len(result.get('coaching_tips', []))}) ===")
for tip in result.get("coaching_tips", []):
    print(f"  - {tip}")

print(f"\n=== BALL TRACKING ===")
bs = result.get("ball_speed", {})
print(f"  Trajectory length: {result.get('ball_trajectory_length', 0)}")
for k, v in bs.items():
    print(f"  {k}: {v}")

print(f"\n=== PHASES (first 20 / last 10) ===")
phases = result.get("phases", [])
if phases:
    for f, p in phases[:20]:
        print(f"  frame {f:4d}: {p}")
    if len(phases) > 30:
        print(f"  ... ({len(phases) - 30} more)")
        for f, p in phases[-10:]:
            print(f"  frame {f:4d}: {p}")
