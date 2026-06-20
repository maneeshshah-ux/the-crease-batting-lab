# Phone Setup Guide — the CREASE Apps

## Two APKs, One Shared Signing

| App | File | Size |
|-----|------|------|
| 🏏 Batting Lab | `the_CREASE_v1.1.0.apk` | 49 KB |
| 📊 Cricket Scoring | `the_CREASE_Scoring_v1.0.0.apk` | 45 KB |

Both are now signed with the **same certificate** — future updates will install seamlessly.

---

## Step 1: Enable "Install from Unknown Sources"

On your Android phone:
1. Open **Settings** → **Security** (or **Biometrics and Security**)
2. Enable **"Install unknown apps"** or **"Install from unknown sources"**
3. Allow your file manager or browser to install apps

If using **Samsung**: Settings → Biometrics and security → Install unknown apps → (choose your file manager) → Allow

---

## Step 2: Transfer APKs to Phone

**Option A — Google Drive (easiest):**
1. On your Mac: upload both APKs to Google Drive
2. On your phone: open Google Drive, download both APKs, tap each to install

**Option B — USB cable:**
1. Connect phone to Mac via USB
2. Copy files:
   ```
   cp the_CREASE_v1.1.0.apk /Volumes/Phone/Download/
   cp the_CREASE_Scoring_v1.0.0.apk /Volumes/Phone/Download/
   ```
3. On phone: open Files/My Files app → Download folder → tap each APK

**Option C — ADB (if developer mode enabled):**
```
adb install the_CREASE_v1.1.0.apk
adb install the_CREASE_Scoring_v1.0.0.apk
```

---

## Step 3: Install (⚠ Important)

> **If you have a PREVIOUS version installed (v1.0.0):**
> You must **uninstall the old version first**, then install the new one.
> Settings → Apps → "the CREASE" → Uninstall
> 
> This is because the old APK was signed with a different key. All future updates will work without uninstalling.

1. Tap the APK file on your phone
2. Tap **Install**
3. Tap **Open**

---

## Step 4: Configure Batting Lab App

When you first open the Batting Lab app, it may show a connection error. That's expected.

1. Tap the **Menu** button (three dots ⋮ in top-right corner)
2. Tap **Server Settings**
3. Enter: `https://the-crease-batting-lab.onrender.com`
4. Tap **Connect**
5. The app will load the Batting Lab dashboard

> 💡 **Cloud server note:** Render's free tier spins down after 15 minutes of inactivity. The first request may take 30–60 seconds to wake up. After that it's fast.

---

## Step 5: Use the Scoring App

The Scoring APK opens directly to: `https://the-crease-batting-lab.onrender.com/scoring/`

No configuration needed — it connects to the cloud server and loads the full scoring interface.

You can also:
- Open `https://the-crease-batting-lab.onrender.com/scoring/` in Chrome
- Tap **Add to Home Screen** from Chrome menu to install as a PWA (works offline)

---

## APK Files Location (on Mac)

```
/Users/mac/Desktop/the CREASE/batting_analyser/
├── the_CREASE_v1.1.0.apk          ← Batting Lab
├── the_CREASE_Scoring_v1.0.0.apk  ← Cricket Scoring
├── android_source/                 ← Batting Lab source
├── android_scoring_source/         ← Scoring app source
└── shared_debug.keystore           ← Shared signing key
```

---

## How to Rebuild (if needed)

```bash
# Batting Lab
cd "/Users/mac/Desktop/the CREASE/batting_analyser"
JAVA_HOME=/tmp/jdk-17.0.12+7/Contents/Home android_source/build_apk.sh

# Scoring App
JAVA_HOME=/tmp/jdk-17.0.12+7/Contents/Home android_scoring_source/build_apk.sh
```

---

## Quick Reference

| Action | What to do |
|--------|-----------|
| Install APK | Uninstall old version first, then tap APK file |
| Change server URL | Menu → Server Settings → enter URL |
| Scoring app offline | Open in Chrome → Add to Home Screen (PWA caches everything) |
| Both apps need internet | Only for initial load or connecting to cloud server |
