# 📱 the CREASE Batting Lab — Android Deployment Guide

> **Goal**: Run the full analysis engine **on your Android phone** using Termux,
> then use the CREASE APK as a native-looking frontend — **no laptop required**.

---

## 📋 What You'll Need

| Item | Where to get it | Required? |
|------|----------------|-----------|
| **F-Droid** app | [f-droid.org](https://f-droid.org) (download & sideload the `.apk`) | ✅ Yes |
| **Termux** (main app) | F-Droid → search "Termux" | ✅ **Essential** |
| **Termux:API** | F-Droid → search "Termux:API" | ⭐ Strongly recommended |
| **Termux:Widget** | F-Droid → search "Termux:Widget" | 🆗 Nice to have |
| **the CREASE APK** | `batting_analyser/the_CREASE_v1.0.0.apk` | 🆗 Alternative to browser |
| A file transfer method | USB cable, cloud drive, or git | ✅ Yes |

---

## 🚀 Step-by-Step Setup

### Step 1: Install Termux from F‑Droid

1. Open the F‑Droid app on your phone.
2. Tap the **search icon** 🔍 and type **Termux**.
3. You'll see a list — these are all Termux-related apps.
4. **Tap "Termux"** (the main one, labelled just "Termux — Terminal emulator with packages").
   > Skip "Termux Hub" — that's just a tool indexer, not the terminal itself.
5. Tap **Install**.
6. While you're there, also tap **Install** on:
   - **Termux:API** — gives wake lock (keeps phone awake during analysis), plus camera/battery/sensor access.
   - **Termux:Widget** — optional, lets you start/stop the server from your home screen.

---

### Step 2: Grant Storage Permission

Termux needs access to your phone's storage so it can read video files and save analysis results.

```bash
# Inside Termux, run:
termux-setup-storage
```

A permission dialog will pop up — tap **Allow**. This creates a `~/storage/` folder that links to your phone's shared storage (Downloads, DCIM, etc.).

---

### Step 3: Update Packages & Install Dependencies

```bash
# Update package lists and upgrade existing packages
pkg update && pkg upgrade -y

# Install core dependencies
pkg install -y python git opencv which build-essential

# Install additional system libraries needed by MediaPipe
pkg install -y libjpeg-turbo libpng libwebp freetype

# Install Termux:API helper (lets Python scripts use Termux:API)
pip install termux-api
```

**⏱ This takes 5–15 minutes** depending on your internet speed. OpenCV on Termux is a native Android build, so it's optimised for your phone.

---

### Step 4: Install Python Packages

```bash
# Install the Python packages for the CREASE engine
pip install flask numpy matplotlib scipy pandas
```

**Note about MediaPipe**: Unfortunately, `mediapipe` doesn't currently have an official Android/ARM64 pip package. Your phone will still run all the analysis **except** the pose estimation. We have two options:

| Option | What works | What's limited |
|--------|-----------|----------------|
| **A — On‑phone (this guide)** | Ball tracking, bat analysis (via contour detection), phase detection, metrics, charts | No MediaPipe pose skeleton overlay |
| **B — Mac server + phone client (PWA)** | Everything including pose skeleton | Requires laptop on same network |

If you want full pose estimation, use **Option B** in the PWA section (your Mac handles MediaPipe, phone just views results).

---

### Step 5: Get the Code Onto Your Phone

Choose **one** of these methods:

<details>
<summary><b>Method A: Git clone (easiest if you push to GitHub)</b></summary>

```bash
# If you've pushed the project to GitHub:
git clone https://github.com/YOUR_USERNAME/batting_analyser.git
```
</details>

<details>
<summary><b>Method B: Transfer via USB cable</b></summary>

1. Connect your phone to your Mac via USB.
2. On your phone, swipe down and tap "Charging via USB" → select **File Transfer**.
3. On your Mac, copy the project folder:
   ```bash
   cp -r "/Users/mac/Desktop/the CREASE/batting_analyser" /Volumes/PHONE_NAME/Download/
   ```
4. On your phone (in Termux):
   ```bash
   cp -r /storage/emulated/0/Download/batting_analyser ~/
   ```
</details>

<details>
<summary><b>Method C: Transfer via cloud (Google Drive / Dropbox)</b></summary>

1. Zip the project on your Mac:
   ```bash
   cd "/Users/mac/Desktop/the CREASE"
   zip -r batting_analyser.zip batting_analyser
   ```
2. Upload to Google Drive / Dropbox / etc.
3. On your phone, download the zip and extract it with a file manager.
4. In Termux:
   ```bash
   cp -r /storage/emulated/0/Download/batting_analyser ~/
   ```
</details>

<details>
<summary><b>Method D: Serve over Wi-Fi (no cables, no cloud)</b></summary>

On your **Mac**, start a temporary HTTP server:
```bash
cd "/Users/mac/Desktop/the CREASE"
python3 -m http.server 9999
```

On your **phone**, in Termux:
```bash
# Find your Mac's IP address (look for 192.168.x.x)
# Then download the project (you'll need to zip it first on the Mac)
# Or use a tool like 'wget' to fetch individual files
pkg install wget
wget -r http://192.168.1.100:9999/batting_analyser/
```
</details>

---

### Step 6: Install the CREASE APK

The APK is a WebView wrapper that opens `http://127.0.0.1:5005` — it makes the analysis UI feel like a native app.

> **Note on APK installation**: On modern Android (Android 14+), you may need to enable "Install from unknown apps" for your file manager.

1. Transfer `the_CREASE_v1.0.0.apk` to your phone (same Methods A–D above).
2. On your phone, tap the APK file and tap **Install**.
3. If prompted, allow "Install from this source" for your file manager app.

---

### Step 7: Run the Server 🎯

```bash
# Go to the project folder
cd ~/batting_analyser

# Start the CREASE analysis server
python3 app.py
```

You should see:
```
 * Running on http://127.0.0.1:5005
 * Serving Flask app 'app'
```

**⚠️ Keep Termux running in the background.** Don't swipe it away — use the **Termux:API wake lock** to prevent Android from killing it:

```bash
# In a second Termux session (swipe from left edge → New session):
pkg install termux-api
termux-wake-lock
```

> Or set `WAKE_LOCK = True` in a config file — the engine can request it automatically.

---

### Step 8: Open the CREASE App

**Option A — Using the APK (recommended):**
1. Tap the **CREASE** app icon on your home screen / app drawer.
2. It will connect to `http://127.0.0.1:5005` automatically.

**Option B — Using Chrome:**

Open Chrome and go to:
```
http://127.0.0.1:5005
```

Both work. The APK gives a full-screen, native-like experience.

---

## 🧩 Optional Enhancements

### 📌 Home Screen Shortcut (Termux:Widget)

Create a one-tap shortcut to start/stop the server:

```bash
# Create the shortcuts directory
mkdir -p ~/.shortcuts

# Create a start script
cat > ~/.shortcuts/Start\ CREASE << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
cd ~/batting_analyser
termux-wake-lock
python3 app.py
EOF

# Create a stop script
cat > ~/.shortcuts/Stop\ CREASE << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
pkill -f "python3 app.py"
termux-wake-unlock
echo "CREASE server stopped."
EOF

# Make them executable
chmod +x ~/.shortcuts/*
```

Then:
1. Long-press your home screen → **Widgets**.
2. Find **Termux:Widget** and add it.
3. Tap the widget → select a shortcut → choose **Start CREASE**.

### 🔄 Auto-Start on Boot (Termux:Boot)

If you want the server to start automatically when the phone turns on:

```bash
# Install Termux:Boot from F-Droid (already installed in Step 1)

# Create the boot script
mkdir -p ~/.termux/boot
cat > ~/.termux/boot/crease << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
termux-wake-lock
cd ~/batting_analyser
python3 app.py
EOF

chmod +x ~/.termux/boot/crease
```

Now reboot your phone — the server will start automatically!

---

## 📹 Recording & Analysis Workflow

Once the server is running on your phone:

```
1. Record batting with your phone camera (side-on, off side, landscape)
2. Open the CREASE app (APK or Chrome → http://127.0.0.1:5005)
3. Tap "Upload Video" → choose from gallery
4. The analysis engine processes it on-device
5. View results: swing path, ball trajectory, joint angles, coaching tips
```

### Camera Setup (for best results)

| Setting | Value |
|---------|-------|
| **Position** | Side-on, **off side** of the batter |
| **Distance** | 8–12 m (25–40 ft) from popping crease |
| **Height** | Batter's chest height |
| **Recording** | 1080p, 30–60 fps, **landscape** |
| **Stability** | Use a tripod — no handheld |
| **Lighting** | Good front light, no backlight/sun behind batter |

---

## 🚨 Troubleshooting

### "OpenCV not found" / ImportError
```bash
# Install the native Termux OpenCV package:
pkg install opencv
# Then uninstall the pip version if installed:
pip uninstall opencv-python -y
```

### "Address already in use" when starting server
```bash
# Kill any existing Flask process:
pkill -f "python3 app.py"
# Then start again:
python3 app.py
```

### Server crashes / out of memory
- Close other apps on the phone.
- Use shorter video clips (1–2 overs / 12–24 balls per clip).
- If your phone has 4GB RAM or less, reduce frame processing:
  ```bash
  # Edit app.py and add to the config section:
  # process_every_n_frames = 3   (processes every 3rd frame)
  ```

### "Permission denied" when running scripts
```bash
chmod +x ~/.shortcuts/*
chmod +x ~/.termux/boot/*
```

### Can't install the APK
- On Android 14+: go to **Settings → Security → Install unknown apps** → allow your file manager.
- On Samsung: **Settings → Biometrics and security → Install unknown apps**.
- On Xiaomi: **Settings → Privacy → Special permissions → Install unknown apps**.

### Termux keeps getting killed in background
- Go to **Phone Settings → Apps → Termux → Battery** and select **Unrestricted** (or "Don't optimise").
- On Xiaomi: also lock Termux in the recent apps list (long-press the Termux card → lock icon).
- On Samsung: also go to Settings → Device care → Battery → "Unmonitored apps" → add Termux.

### I want full pose estimation (MediaPipe)
The phone option doesn't support MediaPipe's ARM64 binary. Two workarounds:
1. **Use the Mac as the server** (PWA option) — the Mac runs MediaPipe, your phone views results.
2. **Use a cloud VPS** — deploy the server to a free-tier server (Railway, Render, etc.) — let me know if you want this.

---

## 📊 Architecture Summary

```
┌─────────────────────────────────────────┐
│           Android Phone                 │
│                                         │
│  ┌────────────┐    ┌────────────────┐   │
│  │ CREASE APK │◄──►│ Termux Server  │   │
│  │ (WebView)  │    │ (Flask :5005)  │   │
│  │            │    │                │   │
│  │  UI that   │    │  • Ball tracker│   │
│  │  feels     │    │  • Bat swing   │   │
│  │  native    │    │  • Phase detect│   │
│  │            │    │  • Metrics     │   │
│  │            │    │  • Charts      │   │
│  └────────────┘    └────────────────┘   │
└─────────────────────────────────────────┘
```

---

## ✅ Checklist

- [ ] **F-Droid** installed
- [ ] **Termux** installed from F-Droid
- [ ] **Termux:API** installed
- [ ] Storage permission granted (`termux-setup-storage`)
- [ ] Packages updated (`pkg update && pkg upgrade`)
- [ ] Core deps installed (`python git opencv`)
- [ ] Python packages installed (`flask numpy matplotlib scipy pandas`)
- [ ] Project code on phone (`~/batting_analyser/`)
- [ ] **the CREASE APK** installed
- [ ] Server starts successfully (`python3 app.py`)
- [ ] App opens and shows the CREASE UI

---

**the CREASE Batting Lab** — Built for cricketers, by cricketers. Zero cost. Zero limits.
