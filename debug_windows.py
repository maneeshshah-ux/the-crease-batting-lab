"""Analyze velocity context around candidate shot regions."""
import json, numpy as np

with open("/Users/mac/Desktop/the CREASE/batting_analyser/sessions/velocity_data.json") as f:
    data = json.load(f)

# Extract
frames = [d["frame"] for d in data]
hv = np.array([d["hand_vel"] for d in data])
fv = np.array([d["foot_vel"] for d in data])

def show_window(center, width=30, label=""):
    """Show velocities around a center frame."""
    start = max(0, center - width)
    end = min(len(hv), center + width)
    print(f"\n--- {label} (center={center}, frames {start}-{end}) ---")
    print(f"{'Frame':>6} {'HandVel':>8} {'FootVel':>8} {'H<-3':>5} {'H>5':>5} {'F>5':>5}")
    for i in range(start, end):
        h = hv[i]
        fv_val = fv[i]
        h_neg = "BACK" if h < -3 else ""
        h_pos = "DOWN" if h > 5 else ""
        f_pos = "STRD" if fv_val > 5 else ""
        print(f"{i:6d} {h:8.1f} {fv_val:8.1f} {h_neg:>5} {h_pos:>5} {f_pos:>5}")

# Key regions to examine
show_window(147, 40, "Shot candidate near frame 147")
show_window(1096, 50, "Shot candidate near frame 1096 (has backlift+stride+downswing)")
show_window(1843, 50, "Shot candidate near frame 1843")
show_window(1048, 30, "Extreme backlift+stride at frame 1048")
show_window(216, 30, "Stride+downswing at frame 216")
show_window(1296, 30, "Extreme downswing at frame 1296")
show_window(880, 30, "Downswing at frame 880")
show_window(125, 30, "Backlift at frame 125")

# Also show a long quiet period
show_window(2000, 20, "Quiet period around 2000")

# Now check: how many frames would get labeled BACKLIFT with threshold -3 vs -5
backlift_counts = {"lt3": 0, "lt5": 0, "lt8": 0, "lt3_cons3": 0, "lt5_cons3": 0}

# Simulate the backlift detection algorithm
def simulate_backlift(hv, threshold, min_consecutive):
    """Simulate backlift labeling."""
    labels = [False] * len(hv)
    consecutive = 0
    for i in range(len(hv)):
        if hv[i] < threshold:
            consecutive += 1
            if consecutive >= min_consecutive:
                labels[i] = True
        else:
            consecutive = 0
    return labels

for thr, label in [(-3, "lt3"), (-5, "lt5"), (-8, "lt8")]:
    labels_cons3 = simulate_backlift(hv, thr, 3)
    labels_cons1 = simulate_backlift(hv, thr, 1)
    print(f"\n=== BACKLIFT SIMULATION (threshold={thr}) ===")
    print(f"  min_consecutive=1: {sum(labels_cons1)} frames labeled ({sum(labels_cons1)/len(hv)*100:.1f}%)")
    print(f"  min_consecutive=3: {sum(labels_cons3)} frames labeled ({sum(labels_cons3)/len(hv)*100:.1f}%)")

# Similar for downswing
def simulate_downswing(hv, threshold, min_consecutive):
    """Simulate downswing labeling."""
    labels = [False] * len(hv)
    consecutive = 0
    for i in range(len(hv)):
        if hv[i] > threshold:
            consecutive += 1
            if consecutive >= min_consecutive:
                labels[i] = True
        else:
            consecutive = 0
    return labels

for thr, label in [(5, "gt5"), (8, "gt8")]:
    labels_cons3 = simulate_downswing(hv, thr, 3)
    labels_cons1 = simulate_downswing(hv, thr, 1)
    print(f"\n=== DOWNSWING SIMULATION (threshold={thr}) ===")
    print(f"  min_consecutive=1: {sum(labels_cons3)} frames labeled ({sum(labels_cons3)/len(hv)*100:.1f}%)")
    print(f"  min_consecutive=3: {sum(labels_cons1)} frames labeled ({sum(labels_cons1)/len(hv)*100:.1f}%)")
