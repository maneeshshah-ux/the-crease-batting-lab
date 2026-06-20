# ☁️ the CREASE Batting Lab — Cloud Deployment Guide

> Deploy the full analysis engine (including **MediaPipe pose estimation**)
> to a free cloud server — access it from anywhere on any device.

---

## 📋 Why Cloud?

| On‑Phone (Termux) | Cloud (Render/Railway) |
|-------------------|----------------------|
| ❌ No MediaPipe pose skeleton | ✅ Full pose estimation + skeleton overlay |
| ❌ Drains phone battery | ✅ Server handles all processing |
| ❌ Phone must stay on | ✅ Runs 24/7 (or wakes on demand) |
| ❌ Local network only | ✅ Accessible from anywhere |
| ❌ Limited RAM/CPU | ✅ 512MB RAM, shared CPU (free tier) |

---

## 🎯 Quick Overview

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Your Phone  │────►│  Cloud Server    │────►│  Any Browser  │
│  (Record     │     │  (Render/Railwy) │     │  (View       │
│   video)     │     │  - MediaPipe     │     │   results)   │
│             │     │  - Ball tracking  │     │              │
│             │     │  - Bat analysis   │     │              │
│             │     │  - Charts         │     │              │
└─────────────┘     └──────────────────┘     └──────────────┘
```

---

## 🚀 Option 1: Deploy to Render (Recommended — Easiest)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/YOUR_USER/batting_analyser)

### Step-by-step:

#### 1. Push the project to GitHub

```bash
# On your Mac, in the project directory:
cd "/Users/mac/Desktop/the CREASE/batting_analyser"

# Initialize git (if not already)
git init
git add -A

# Create a .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
uploads/
sessions/
reports/
frames/
*.apk
.DS_Store
EOF

git commit -m "Initial commit — the CREASE Batting Lab"

# Create a GitHub repo, then:
git remote add origin https://github.com/YOUR_USER/batting_analyser.git
git branch -M main
git push -u origin main
```

#### 2. Deploy on Render

1. Go to [render.com](https://render.com) and sign up (free, no credit card).
2. In the dashboard, click **New +** → **Blueprint**.
3. Connect your GitHub account and select the `batting_analyser` repo.
4. Render will detect `render.yaml` and auto-configure everything.
5. Click **Apply** — deployment starts immediately.

**⏱ First build takes 5–10 minutes** (installing OpenCV + MediaPipe).

#### 3. Get your URL

Once deployed, Render gives you a URL like:
```
https://the-crease-batting-lab.onrender.com
```

Open this URL in any browser — you'll see the CREASE dashboard.

---

## 🚀 Option 2: Deploy to Railway

### Step-by-step:

1. Push the project to GitHub (same as Step 1 above).
2. Go to [railway.app](https://railway.app) and sign up (free tier: $5 credit, no credit card needed).
3. Click **New Project** → **Deploy from GitHub repo**.
4. Select your `batting_analyser` repo.
5. Railway auto-detects Python + Dockerfile and deploys.

The `nixpacks.toml` in the repo ensures all system libraries are installed.

**⏱ First build takes 5–10 minutes.**

Railway gives you a URL like:
```
https://batting-analyser.up.railway.app
```

---

## 🌐 Using the Cloud Deployment

### From Your Phone:

1. Open the URL in **Chrome** (or any browser).
2. The full CREASE UI loads — tap **"Add to Home Screen"** for a PWA.
3. Record a batting video, upload it, and get analysis.
4. MediaPipe pose estimation works because the server has full CPU/RAM.

### From Your Mac/PC:

Same URL — works in any browser.

### Setting a Custom Domain (Optional):

Both Render and Railway let you set a custom domain in the free tier.

---

## ⚙️ Configuration

### Environment Variables

These can be set in the Render/Railway dashboard:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5005` | Server port (Render/Railway set this automatically) |
| `SECRET_KEY` | auto-generated | Flask session encryption key |
| `DEBUG` | `false` | Set to `true` for verbose logging |

### File Upload Limits

The free tiers have:
- **Render free**: 512MB RAM, files up to ~100MB recommended
- **Railway free**: 512MB RAM, similar limits

For best results, keep videos under **100MB** or 2–3 overs long.

---

## 🧠 How MediaPipe Works in the Cloud

On Termux (Android), MediaPipe doesn't have an ARM64 pip package.
On the cloud Linux server, MediaPipe runs perfectly:

```
[Video Upload] ──► [Cloud Server] ──► [MediaPipe Pose] ──► [Analysis]
                        │                      │
                   opencv-python         33-point landmarks
                   -headless              + batting skeleton
```

The server uses `opencv-python-headless` (no GUI needed) and MediaPipe
processes every frame to extract the 33-point pose skeleton.

---

## 💾 File Storage Caveat

Render and Railway both have **ephemeral filesystems** — uploaded videos and
analysis results disappear if the server restarts or goes to sleep.

**This is fine for typical use**: you upload → analyse → view/download results
in the same session. The JSON session data is also in memory.

If you want permanent storage, we can add:
- **Cloudinary** (free tier: 25GB storage) — for video uploads
- **Supabase** (free tier: 500MB database) — for session persistence

Let me know if you'd like me to add cloud storage integration.

---

## 🌙 Sleep/Wake Behavior (Free Tier)

Both platforms **sleep** the server after ~15 minutes of inactivity:

- **Render**: First request after sleep takes **30–60 seconds** to wake up.
- **Railway**: Similar behavior.

This means:
- The first analysis of the day may have a 30s delay.
- Once the server is hot, it processes at full speed.
- You can "ping" the URL periodically to keep it warm (tools like UptimeRobot).

---

## 🧪 Testing Locally with Docker

Want to test the exact Docker image before deploying?

```bash
cd "/Users/mac/Desktop/the CREASE/batting_analyser"

# Build the Docker image
docker build -t crease-batting-lab .

# Run it locally
docker run -p 5005:5005 crease-batting-lab

# Open in browser
open http://127.0.0.1:5005
```

---

## 🔄 Updating After Deployment

Any time you push changes to GitHub, Render/Railway auto-redeploys:

```bash
git add -A
git commit -m "Updated analysis engine"
git push
```

The cloud server rebuilds and redeploys automatically.

---

## 📊 Which Path Should You Choose?

| Your situation | Best option |
|---------------|-------------|
| Want full pose estimation + any device access | **☁️ Cloud (Render)** |
| Want everything 100% free, no accounts | **📱 Termux on phone** |
| Have a Mac + phone on same Wi-Fi | **📲 PWA (Mac server)** |
| Want no sleep/wake delays, instant access | **📱 Termux (always on)** |
| Want best experience everywhere | **☁️ Cloud + PWA on phone** |

---

**the CREASE Batting Lab** — Built for cricketers, by cricketers. Zero cost. Zero limits.
