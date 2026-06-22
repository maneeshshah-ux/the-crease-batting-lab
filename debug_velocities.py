"""Debug script: print hand/foot velocities for all frames to calibrate thresholds."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from engine.pose_estimator import PoseEstimator
from engine.phase_detector import PhaseDetector, BattingPhase

video_path = "/Users/mac/Desktop/the CREASE/batting_analyser/Batting Videos/Dia.mp4"

import cv2
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {total_frames} frames, {fps} fps, {w}x{h}")

# Process with pose estimator
pe = PoseEstimator(static_mode=False, model_complexity=1, smooth=True)
pd = PhaseDetector(batting_hand="right", fps=fps)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    pose = pe.process_frame(frame)
    if pose["success"]:
        pd.add_frame(pose["landmarks"], frame_idx)
    if frame_idx % 500 == 0:
        print(f"  processed frame {frame_idx}/{total_frames}")
    frame_idx += 1

cap.release()
pe.close()

print(f"\nExtracted {len(pd.frame_data)} frames with pose data")

# Compute velocities
hand_vel = pd._compute_hand_velocity()
foot_vel = pd._compute_foot_velocity()

# Save to JSON for analysis
data = []
for i in range(len(hand_vel)):
    data.append({
        "frame": i,
        "hand_vel": hand_vel[i],
        "foot_vel": foot_vel[i],
    })

out_path = "/Users/mac/Desktop/the CREASE/batting_analyser/sessions/velocity_data.json"
with open(out_path, "w") as f:
    json.dump(data, f, indent=1)

print(f"\nSaved velocity data to {out_path}")

# Stats
hv = np.array(hand_vel)
fv = np.array(foot_vel)
print(f"\n=== HAND VELOCITY STATS ===")
print(f"  Mean: {hv.mean():.2f}")
print(f"  Std:  {hv.std():.2f}")
print(f"  Min:  {hv.min():.2f}")
print(f"  Max:  {hv.max():.2f}")
print(f"  P1:   {np.percentile(hv, 1):.2f}")
print(f"  P5:   {np.percentile(hv, 5):.2f}")
print(f"  P10:  {np.percentile(hv, 10):.2f}")
print(f"  P25:  {np.percentile(hv, 25):.2f}")
print(f"  P75:  {np.percentile(hv, 75):.2f}")
print(f"  P90:  {np.percentile(hv, 90):.2f}")
print(f"  P95:  {np.percentile(hv, 95):.2f}")
print(f"  P99:  {np.percentile(hv, 99):.2f}")
print(f"  % < -3: {(hv < -3).mean() * 100:.1f}%")
print(f"  % < -5: {(hv < -5).mean() * 100:.1f}%")
print(f"  % < -8: {(hv < -8).mean() * 100:.1f}%")
print(f"  % > 3:  {(hv > 3).mean() * 100:.1f}%")
print(f"  % > 5:  {(hv > 5).mean() * 100:.1f}%")
print(f"  % > 8:  {(hv > 8).mean() * 100:.1f}%")

print(f"\n=== FOOT VELOCITY STATS ===")
print(f"  Mean: {fv.mean():.2f}")
print(f"  Std:  {fv.std():.2f}")
print(f"  Min:  {fv.min():.2f}")
print(f"  Max:  {fv.max():.2f}")
print(f"  P1:   {np.percentile(fv, 1):.2f}")
print(f"  P5:   {np.percentile(fv, 5):.2f}")
print(f"  P25:  {np.percentile(fv, 25):.2f}")
print(f"  P75:  {np.percentile(fv, 75):.2f}")
print(f"  P90:  {np.percentile(fv, 90):.2f}")
print(f"  P95:  {np.percentile(fv, 95):.2f}")
print(f"  P99:  {np.percentile(fv, 99):.2f}")
print(f"  % > 3:  {(fv > 3).mean() * 100:.1f}%")
print(f"  % > 5:  {(fv > 5).mean() * 100:.1f}%")
print(f"  % > 8:  {(fv > 8).mean() * 100:.1f}%")

# Find the top 20 largest negative hand velocities (potential backlift moments)
print(f"\n=== TOP 20 BACKLIFT CANDIDATES (most negative hand_vel) ===")
indices = np.argsort(hv)[:20]
for idx in indices:
    print(f"  frame {idx:4d}: hand_vel={hv[idx]:6.2f}, foot_vel={fv[idx]:6.2f}")

# Find the top 20 largest positive foot velocities (potential stride moments)
print(f"\n=== TOP 20 STRIDE CANDIDATES (most positive foot_vel) ===")
indices = np.argsort(fv)[-20:][::-1]
for idx in indices:
    print(f"  frame {idx:4d}: foot_vel={fv[idx]:6.2f}, hand_vel={hv[idx]:6.2f}")

# Find the top 20 largest positive hand velocities (potential downswing moments)
print(f"\n=== TOP 20 DOWNSWING CANDIDATES (most positive hand_vel) ===")
indices = np.argsort(hv)[-20:][::-1]
for idx in indices:
    print(f"  frame {idx:4d}: hand_vel={hv[idx]:6.2f}, foot_vel={fv[idx]:6.2f}")

# Look at the detected "shot" regions (frames 142-157, 1096-1111, 1842-1858)
print(f"\n=== VELOCITIES AT DETECTED SHOT REGIONS ===")
for start, end in [(142, 157), (1096, 1111), (1842, 1858)]:
    print(f"\n  Frames {start}-{end}:")
    for i in range(start, end+1):
        print(f"    frame {i:4d}: hand_vel={hv[i]:6.2f}, foot_vel={fv[i]:6.2f}")
