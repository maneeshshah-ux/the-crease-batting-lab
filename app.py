#!/usr/bin/env python3
"""
the CREASE Batting Lab — Zero-cost AI cricket batting analysis.
Record, upload, analyse. No subscriptions. No limits.

Run:     python3 app.py
Open:    http://127.0.0.1:5005
Android: Install PWA from Chrome 'Add to Home Screen'
         Or run locally in Termux (see README)
"""

import os
import json
import uuid
import threading
import functools
import cv2
from datetime import datetime
from pathlib import Path

from flask import (Flask, render_template, request, jsonify,
                   send_from_directory, url_for, redirect, session,
                   send_file)
from werkzeug.utils import secure_filename

from engine.analyser import BattingAnalyser
from engine.highlight_reel import HighlightReel
from engine.scorecard_image import ScorecardImage
from engine.multi_cam_sync import MultiCameraSync, FfmpegNotFoundError
from engine.pro_comparison import ProComparison, ZonalComparison
from engine.report_generator import generate_report

# ── Commercial SaaS imports ────────────────────────────────────────────────
from auth import auth_bp, login_required
from stripe_payments import (
    is_stripe_configured, create_checkout_session,
    handle_webhook, get_usage_remaining, increment_usage
)
from supabase_client import get_supabase, is_configured as supabase_configured


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
SESSION_DIR = BASE_DIR / "sessions"
REPORT_DIR = BASE_DIR / "reports"
FRAME_DIR = BASE_DIR / "frames"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}

for d in [UPLOAD_DIR, SESSION_DIR, REPORT_DIR, FRAME_DIR]:
    d.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "the-crease-batting-lab-2026")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB
app.config["PREFERRED_URL_SCHEME"] = os.environ.get("URL_SCHEME", "http")
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Register the SaaS auth blueprint
app.register_blueprint(auth_bp)

# Trust X-Forwarded-* headers from Render/Railway proxies
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Prevent WebView caching — always serve fresh pages
@app.after_request
def no_cache(response):
    if response.content_type and response.content_type.startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["Clear-Site-Data"] = "\"cache\", \"storage\""
    return response

# In-memory job tracking
analysis_jobs = {}


def allowed_file(filename):
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def _find_session_file(session_id):
    """
    Locate a session JSON on disk, trying both filename patterns:
        {session_id}.json  (legacy)
        analysis_{session_id}.json  (current)
    Returns Path or None.
    """
    p = SESSION_DIR / f"{session_id}.json"
    if p.exists():
        return p
    p = SESSION_DIR / f"analysis_{session_id}.json"
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Dashboard home."""
    sessions = _list_sessions()
    stats = _compute_stats(sessions)
    return render_template("index.html", sessions=sessions, stats=stats)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Upload a batting video for analysis. OPEN TO EVERYONE — no login needed."""
    if request.method == "GET":
        return render_template("upload.html")

    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported format. Use: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # Save upload
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_name = f"{timestamp}_{uuid.uuid4().hex[:6]}_{filename}"
    filepath = str(UPLOAD_DIR / unique_name)
    file.save(filepath)

    # Get parameters
    batting_hand = request.form.get("batting_hand", "right")
    ball_color = request.form.get("ball_color", "red")
    camera_view = request.form.get("camera_view", "side_off")

    # Generate share token for viral distribution
    share_token = uuid.uuid4().hex[:10]

    # Multi-camera session code (auto-generated; user can optionally enter existing)
    session_code = request.form.get("session_code", "").strip().upper()
    if not session_code or not MultiCameraSync.validate_session_code(session_code):
        session_code = MultiCameraSync.generate_session_code()

    # Get current user_id (optional — for logged-in users only)
    current_user_id = session.get("user_id", "")

    # Start background analysis
    job_id = uuid.uuid4().hex[:12]
    analysis_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Waiting to start...",
        "video_path": filepath,
        "video_name": filename,
        "batting_hand": batting_hand,
        "ball_color": ball_color,
        "camera_view": camera_view,
        "session_label": request.form.get("session_label", ""),
        "session_code": session_code,
        "user_id": current_user_id,
        "share_token": share_token,
        "result": None,
        "error": None,
    }

    thread = threading.Thread(
        target=_run_analysis,
        args=(job_id, filepath, batting_hand, ball_color, camera_view),
        daemon=True,
    )
    thread.start()

    return render_template("processing.html", job_id=job_id, video_name=filename,
                           session_code=session_code)


@app.route("/job/<job_id>")
def job_status(job_id):
    """Job status page."""
    job = analysis_jobs.get(job_id)
    if not job:
        return render_template("error.html", message="Analysis job not found"), 404
    if job["status"] == "completed" and job["result"]:
        return redirect(url_for("session_view", session_id=job["result"]["session_id"]))
    return render_template("processing.html", job_id=job_id,
                           video_name=job.get("video_name", ""),
                           session_code=job.get("session_code", ""))


@app.route("/api/job/<job_id>")
def api_job_status(job_id):
    """API: job status for polling."""
    job = analysis_jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    response = {
        "status": job["status"],
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
    }
    if job["status"] == "completed":
        response["session_id"] = job["result"]["session_id"]
        response["redirect"] = url_for("session_view",
                                       session_id=job["result"]["session_id"])
    if job["error"]:
        response["error"] = job["error"]
    return jsonify(response)


@app.route("/session/<session_id>")
def session_view(session_id):
    """View analysis results — OPEN TO EVERYONE."""
    session_path = _find_session_file(session_id)
    if not session_path:
        return render_template("error.html",
                               message=f"Session {session_id} not found"), 404
    with open(session_path) as f:
        data = json.load(f)

    # Convert flat phase list [[frame, name], ...] to segments [{phase, start_frame, end_frame}, ...]
    raw_phases = data.get("phases", [])
    if raw_phases and isinstance(raw_phases[0], list):
        segments = []
        if raw_phases:
            current_phase = raw_phases[0][1]
            current_start = raw_phases[0][0]
            for i in range(1, len(raw_phases)):
                frame, name = raw_phases[i]
                if name != current_phase:
                    segments.append({
                        "phase": current_phase,
                        "start_frame": current_start,
                        "end_frame": raw_phases[i - 1][0]
                    })
                    current_phase = name
                    current_start = frame
            # Last segment
            segments.append({
                "phase": current_phase,
                "start_frame": current_start,
                "end_frame": raw_phases[-1][0]
            })
        data["phases"] = segments

    share_token = data.get("share_token", session_id[:10])
    share_url = url_for("shared_session", share_token=share_token, _external=True)

    # Check if we can generate highlights (video must exist)
    has_video = bool(data.get("output_video_path"))
    num_shots = len(data.get("shot_summary", []))

    # Bragging rights for the hero area
    bragging = data.get("bragging_rights", {})

    bowling = data.get("bowling_analysis", {})

    return render_template(
        "session_detail.html",
        session=data,
        share_url=share_url,
        has_video=has_video,
        num_shots=num_shots,
        bragging=bragging,
        bowling=bowling,
    )


@app.route("/s/<share_token>")
def shared_session(share_token):
    """Public shareable session link — no login required, great for social sharing."""
    # Search all session files for matching share_token
    for f in SESSION_DIR.glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            # Match by: share_token, session_id, or filename pattern
            session_id_from_file = data.get("session_id", "")
            share_token_from_data = data.get("share_token", "")
            file_basename = f.stem  # e.g. "analysis_abc12345" or "abc12345"

            if (share_token_from_data == share_token
                    or session_id_from_file == share_token
                    or session_id_from_file == share_token[:8]
                    or file_basename.endswith(share_token)
                    or file_basename.endswith(share_token[:8])):
                # Convert flat phase list [[frame, name], ...] to segments
                raw_phases = data.get("phases", [])
                if raw_phases and isinstance(raw_phases[0], list):
                    segments = []
                    if raw_phases:
                        current_phase = raw_phases[0][1]
                        current_start = raw_phases[0][0]
                        for i in range(1, len(raw_phases)):
                            frame, name = raw_phases[i]
                            if name != current_phase:
                                segments.append({
                                    "phase": current_phase,
                                    "start_frame": current_start,
                                    "end_frame": raw_phases[i - 1][0]
                                })
                                current_phase = name
                                current_start = frame
                        segments.append({
                            "phase": current_phase,
                            "start_frame": current_start,
                            "end_frame": raw_phases[-1][0]
                        })
                    data["phases"] = segments

                share_url = url_for("shared_session", share_token=share_token, _external=True)
                has_video = bool(data.get("output_video_path"))
                bragging = data.get("bragging_rights", {})
                bowling = data.get("bowling_analysis", {})
                return render_template("session_detail.html", session=data, share_url=share_url,
                                       is_shared=True, has_video=has_video,
                                       num_shots=len(data.get("shot_summary", [])),
                                       bragging=bragging, bowling=bowling)
        except (json.JSONDecodeError, IOError):
            continue
    return render_template("error.html", message="Session not found. The link may have expired."), 404


# ---------------------------------------------------------------------------
# Phase 4 — Social Sharing, Highlights, Scorecards
# ---------------------------------------------------------------------------

# Track share counts in-memory (simple key-value, persists per session)
_share_counts: dict = {}


@app.route("/session/<session_id>/api/share", methods=["POST"])
def api_track_share(session_id):
    """Track a social share event (called from the share button JS)."""
    platform = request.json.get("platform", "unknown") if request.is_json else "unknown"
    key = f"{session_id}:{platform}"
    _share_counts[key] = _share_counts.get(key, 0) + 1
    return jsonify({"ok": True, "total_shares": sum(v for k, v in _share_counts.items()
                                                    if k.startswith(session_id))})


@app.route("/session/<session_id>/qr")
def session_qr(session_id):
    """Generate and serve a QR code for the session's multi-camera code."""
    path = _find_session_file(session_id)
    if not path:
        return render_template("error.html", message="Session not found"), 404
    with open(path) as f:
        data = json.load(f)

    code = data.get("session_code", session_id[:6].upper())
    from io import BytesIO
    try:
        qr_path = MultiCameraSync.generate_qr_code(code)
        with open(qr_path, "rb") as fh:
            buf = BytesIO(fh.read())
        os.unlink(qr_path)
    except ImportError:
        # Fallback: text-based code representation
        buf = BytesIO()
        buf.write(code.encode("ascii"))
    buf.seek(0)
    return send_file(buf, mimetype="image/png",
                     download_name=f"crease_qr_{code}.png")


# ---------------------------------------------------------------------------
# Phase 7 — Multi-Camera Sync (PRO-gated)
# ---------------------------------------------------------------------------

@app.route("/multi-cam")
def multi_cam_page():
    """Multi-camera sync page. PRO feature."""
    sessions = _list_sessions()
    return render_template("multi_cam.html", sessions=sessions,
                           pro_feature=MultiCameraSync.PRO_FEATURE)


@app.route("/api/sync-multi-cam", methods=["POST"])
def api_sync_multi_cam():
    """API: sync multiple videos by session codes. PRO-gated."""
    if MultiCameraSync.PRO_FEATURE:
        # Check if user is PRO
        from auth import login_required  # already imported
        user_id = session.get("user_id", "")
        if not user_id:
            return jsonify({"error": "Login required. Multi-camera sync is a PRO feature."}), 401
        # TODO: check subscription_tier == 'pro' or 'enterprise' in supabase profiles
        # For now, allow any logged-in user to test

    data = request.get_json() or {}
    session_ids = data.get("session_ids", [])
    session_codes = data.get("session_codes", [])

    if len(session_ids) + len(session_codes) < 2:
        return jsonify({"error": "Provide at least 2 session IDs or codes to sync"}), 400

    # Collect video paths by session id
    video_paths = []
    for sid in session_ids:
        sp = SESSION_DIR / f"{sid}.json"
        if sp.exists():
            with open(sp) as f:
                sdata = json.load(f)
            vp = sdata.get("video_path") or sdata.get("output_video_path")
            if vp and os.path.exists(vp):
                video_paths.append(vp)
            else:
                return jsonify({"error": f"Video not found for session {sid}"}), 404

    # Also find videos by session code
    for code in session_codes:
        code = code.strip().upper()
        for f in SESSION_DIR.glob("*.json"):
            try:
                with open(f) as fh:
                    sdata = json.load(fh)
                if sdata.get("session_code", "").upper() == code:
                    vp = sdata.get("video_path") or sdata.get("output_video_path")
                    if vp and os.path.exists(vp) and vp not in video_paths:
                        video_paths.append(vp)
            except (json.JSONDecodeError, IOError):
                continue

    if len(video_paths) < 2:
        return jsonify({"error": "Could not find at least 2 valid videos to sync"}), 400

    try:
        sync = MultiCameraSync()
        result = sync.sync_videos(video_paths)
        return jsonify(result)
    except FfmpegNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sync-multi-cam/composite", methods=["POST"])
def api_sync_multi_cam_composite():
    """API: create composite video from synced cameras. PRO-gated."""
    if MultiCameraSync.PRO_FEATURE:
        user_id = session.get("user_id", "")
        if not user_id:
            return jsonify({"error": "Login required. Multi-camera sync is a PRO feature."}), 401

    data = request.get_json() or {}
    session_ids = data.get("session_ids", [])
    session_codes = data.get("session_codes", [])
    layout = data.get("layout", "side_by_side")

    if len(session_ids) + len(session_codes) < 2:
        return jsonify({"error": "Provide at least 2 session IDs or codes to sync"}), 400

    # Collect video paths
    video_paths = []
    for sid in session_ids:
        sp = SESSION_DIR / f"{sid}.json"
        if sp.exists():
            with open(sp) as f:
                sdata = json.load(f)
            vp = sdata.get("video_path") or sdata.get("output_video_path")
            if vp and os.path.exists(vp):
                video_paths.append(vp)

    for code in session_codes:
        code = code.strip().upper()
        for f in SESSION_DIR.glob("*.json"):
            try:
                with open(f) as fh:
                    sdata = json.load(fh)
                if sdata.get("session_code", "").upper() == code:
                    vp = sdata.get("video_path") or sdata.get("output_video_path")
                    if vp and os.path.exists(vp) and vp not in video_paths:
                        video_paths.append(vp)
            except (json.JSONDecodeError, IOError):
                continue

    if len(video_paths) < 2:
        return jsonify({"error": "Need at least 2 videos for compositing"}), 400

    try:
        sync = MultiCameraSync()
        sync_result = sync.sync_videos(video_paths)
        output_name = f"multi_cam_composite_{uuid.uuid4().hex[:8]}.mp4"
        output_path = str(REPORT_DIR / output_name)
        result = sync.create_multi_cam_video(
            video_paths, sync_result["offsets"], output_path, layout=layout
        )
        return jsonify({
            "success": True,
            "output_path": result,
            "download_url": url_for("download_file", filename=output_name, _external=True),
        })
    except FfmpegNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def download_file(filename):
    """Generic file download (for composites etc)."""
    return send_from_directory(str(REPORT_DIR), filename, as_attachment=True)


@app.route("/session/<session_id>/scorecard")
def session_scorecard(session_id):
    """Generate and serve a shareable scorecard image for this session."""
    path = _find_session_file(session_id)
    if not path:
        return render_template("error.html", message="Session not found"), 404
    with open(path) as f:
        data = json.load(f)

    from io import BytesIO
    card = ScorecardImage()
    img = card.create(data, fmt="square")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png",
                     download_name=f"crease_scorecard_{session_id}.png",
                     as_attachment=True)


@app.route("/session/<session_id>/report")
def session_report(session_id):
    """Generate and download a professional PDF coaching report for this session."""
    path = _find_session_file(session_id)
    if not path:
        return render_template("error.html",
                               message=f"Session {session_id} not found"), 404

    with open(path) as f:
        session_data = json.load(f)

    # Build report_data (mirrors logic in report_generator.generate_report_from_json)
    ss = session_data.get("session_summary", {})
    shots = session_data.get("shot_summary", [])
    bs = session_data.get("bat_speed", {})
    head_score = ss.get("head_stability_score", 0)
    total_shots = len(shots)
    complete_shots = len([s for s in shots if s.get("has_impact")])
    completion_pct = (complete_shots / total_shots * 100) if total_shots > 0 else 0
    avg_knee = ss.get("avg_front_knee_angle", 154)
    avg_spine = ss.get("avg_spine_angle", 166)
    peak_kmh = bs.get("peak_kmh", 0) if bs.get("kmh_estimated") else 0

    session_score = min(100, max(0,
        (head_score * 0.35) +
        (completion_pct * 0.25) +
        (min(100, (avg_knee - 100) * 1.5) * 0.15) +
        (min(100, max(0, 180 - avg_spine) * 3) * 0.15) +
        (min(100, max(0, peak_kmh - 60)) * 0.10)
    ))

    priorities = []
    if head_score < 60:
        priorities.append({"rank": 1, "area": "HEAD STABILITY",
                           "drill": "Head-Still Drill: Place a bottle cap on your head while shadow batting. Play 50 forward defensive strokes without it falling off."})
    if completion_pct < 50:
        priorities.append({"rank": len(priorities) + 1, "area": "SHOT COMMITMENT",
                           "drill": "Commitment Drill: Commit to every shot you start. A full swing builds consistency."})
    if avg_knee > 155:
        priorities.append({"rank": len(priorities) + 1, "area": "KNEE FLEX",
                           "drill": "Knee-Tap Drill: Mark a spot 12 inches down the pitch. Every shot, front foot to that spot with knee bent to 130 deg."})
    if avg_spine < 155:
        priorities.append({"rank": len(priorities) + 1, "area": "POSTURE",
                           "drill": "Corridor Drill: Place a second set of stumps 4ft down. Reach with your front foot only, not your head."})

    report_data = {
        "priorities": priorities,
        "session_score": session_score,
    }

    # Find analysis video for frame annotation
    video_path = session_data.get("output_video_path", "")
    if not video_path or not os.path.exists(video_path):
        # Try the original upload video
        video_path = session_data.get("video_path", "")
    if not video_path or not os.path.exists(video_path):
        video_path = None

    # Generate PDF to a temp file
    pdf_path = str(REPORT_DIR / f"report_{session_id}.pdf")
    try:
        generate_report(
            session_data=session_data,
            report_data=report_data,
            analysis_video_path=video_path,
            output_path=pdf_path,
            skip_annotated_frames=(video_path is None),
        )
        return send_file(pdf_path, mimetype="application/pdf",
                         download_name=f"crease_report_{session_id}.pdf",
                         as_attachment=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Report generation failed: {str(e)}"}), 500


@app.route("/session/<session_id>/highlight/<int:clip_idx>")
def session_highlight(session_id, clip_idx):
    """Generate and serve a highlight clip. clip_idx=1 means best clip."""
    path = _find_session_file(session_id)
    if not path:
        return render_template("error.html", message="Session not found"), 404
    with open(path) as f:
        data = json.load(f)

    # Generate clips
    reel = HighlightReel(max_clips=5, clip_duration_sec=6.0)
    clips = reel.generate(data, str(SESSION_DIR), video_path=data.get("video_path"))

    if not clips:
        return jsonify({"error": "No highlight clips available"}), 404

    # clip_idx is 1-based (1 = best), convert to 0-based index
    idx = max(0, min(clip_idx - 1, len(clips) - 1))
    clip = clips[idx]

    if not os.path.isfile(clip["path"]):
        return jsonify({"error": "Clip file not found"}), 404

    return send_file(clip["path"], mimetype="video/mp4",
                     download_name=clip["filename"],
                     as_attachment=True)


@app.route("/highlights")
def highlights_gallery():
    """Public gallery of all session highlights — zero login needed."""
    from engine.bragging_rights import compute_bragging_rights

    entries = []
    for f in sorted(SESSION_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if not data.get("share_token") or not data.get("output_video_path"):
                continue
            # Shot counts for the card
            ss = data.get("shot_summary", [])
            bragging = compute_bragging_rights(ss, data.get("session_summary", {}),
                                               data)
            entries.append({
                "session_id": data.get("session_id", ""),
                "share_token": data.get("share_token", ""),
                "date": data.get("analysis_timestamp", "")[:10],
                "shots": data.get("num_shots_detected", 0),
                "duration": data.get("duration_sec", 0),
                "label": data.get("session_label", f"Session {data.get('session_id', '')[:8]}"),
                "batting_hand": data.get("batting_hand", "right").title(),
                "has_video": bool(data.get("output_video_path")),
                "best_shot": bragging.get("best_shot_label", ""),
                "top_speed": (
                    data.get("bat_speed", {}).get("swing_avg_kmh", 0)
                    if isinstance(data.get("bat_speed"), dict) else 0
                ),
                "one_liners": bragging.get("one_liners", []),
                "bowling_type": (data.get("bowling_analysis", {}) or {}).get("bowl_type_label", ""),
                "bowling_icon": (data.get("bowling_analysis", {}) or {}).get("bowl_type_icon", ""),
            })
        except (json.JSONDecodeError, KeyError, IOError):
            continue

    return render_template("highlights.html", entries=entries, count=len(entries))


@app.route("/sessions")
def sessions_list():
    """List all sessions."""
    sessions = _list_sessions()
    return render_template("sessions.html", sessions=sessions)


@app.route("/delete/<session_id>", methods=["POST"])
def delete_session(session_id):
    """Delete a single session."""
    path = _find_session_file(session_id)
    if path:
        path.unlink()
    return redirect(url_for("sessions_list"))


@app.route("/api/delete-all", methods=["POST"])
def api_delete_all():
    """Delete all sessions."""
    for f in SESSION_DIR.glob("*.json"):
        f.unlink()
    return jsonify({"success": True})


@app.route("/compare")
def compare_view():
    """Compare multiple sessions."""
    sessions = _list_sessions()
    compare_ids = request.args.getlist("ids")
    selected = [s for s in sessions if s["id"] in compare_ids]
    return render_template("compare.html", sessions=sessions, selected=selected)


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """API: compare sessions."""
    data = request.get_json()
    ids = data.get("session_ids", [])
    results = []
    for sid in ids:
        path = SESSION_DIR / f"{sid}.json"
        if path.exists():
            with open(path) as f:
                results.append(json.load(f))
    if len(results) < 2:
        return jsonify({"error": "Need at least 2 sessions"}), 400
    analyser = BattingAnalyser()
    comparison = analyser.compare_sessions(results)
    return jsonify(comparison)


# ---------------------------------------------------------------------------
# Phase 8 — Pro Comparison
# ---------------------------------------------------------------------------

@app.route("/pro-compare/<session_id>")
def pro_compare_view(session_id):
    """Pro Comparison page — compare your session against professional players."""
    session_path = _find_session_file(session_id)
    if not session_path:
        return render_template("error.html",
                               message=f"Session {session_id} not found"), 404
    with open(session_path) as f:
        data = json.load(f)

    # Run comparison
    try:
        comparator = ProComparison(camera_view=data.get("camera_view", "front_on"))
        result = comparator.compare(data)
    except Exception as e:
        return render_template("error.html",
                               message=f"Comparison failed: {str(e)}"), 500

    return render_template(
        "pro_comparison.html",
        session_id=session_id,
        session=data,
        comparison_result=result,
    )


@app.route("/api/pro-compare/<session_id>")
def api_pro_compare(session_id):
    """API: Get pro comparison data for a session."""
    session_path = _find_session_file(session_id)
    if not session_path:
        return jsonify({"error": "Session not found"}), 404
    with open(session_path) as f:
        data = json.load(f)

    try:
        comparator = ProComparison(camera_view=data.get("camera_view", "front_on"))
        result = comparator.compare(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pro-compare/players")
def api_pro_players():
    """API: List available pro players for comparison."""
    level = request.args.get("level")
    players = ProComparison.list_pro_players(level)
    return jsonify({"players": players, "disclaimer": ProComparison.get_legal_disclaimer()})


@app.route("/api/pro-compare/disclaimer")
def api_pro_disclaimer():
    """API: Get the legal disclaimer text."""
    return jsonify({"disclaimer": ProComparison.get_legal_disclaimer()})


# ---------------------------------------------------------------------------
# Zonal Comparison
# ---------------------------------------------------------------------------

@app.route("/zonal-compare/<session_id>")
def zonal_compare_view(session_id):
    """Zonal Comparison page — zone-level + player-level matching."""
    session_path = _find_session_file(session_id)
    if not session_path:
        return render_template("error.html",
                               message=f"Session {session_id} not found"), 404
    with open(session_path) as f:
        data = json.load(f)

    try:
        comparator = ZonalComparison(camera_view=data.get("camera_view", "front_on"))
        result = comparator.compare(data)
    except Exception as e:
        return render_template("error.html",
                               message=f"Zonal comparison failed: {str(e)}"), 500

    return render_template(
        "zonal_compare.html",
        session_id=session_id,
        session=data,
        comparison_result=result,
    )


@app.route("/api/zonal-compare/<session_id>")
def api_zonal_compare(session_id):
    """API: Get zonal comparison data for a session."""
    session_path = _find_session_file(session_id)
    if not session_path:
        return jsonify({"error": "Session not found"}), 404
    with open(session_path) as f:
        data = json.load(f)

    try:
        comparator = ZonalComparison(camera_view=data.get("camera_view", "front_on"))
        result = comparator.compare(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/zonal-compare/zones")
def api_zonal_zones():
    """API: List available zones."""
    zones = ZonalComparison.list_zones()
    return jsonify({"zones": zones})


@app.route("/api/zonal-compare/players")
def api_zonal_players():
    """API: List players, optionally filtered by zone/gender/level."""
    level = request.args.get("level")
    gender = request.args.get("gender")
    zone_key = request.args.get("zone")
    players = ZonalComparison.list_pro_players(level=level, gender=gender, zone_key=zone_key)
    return jsonify({"players": players, "disclaimer": ProComparison.get_legal_disclaimer()})


@app.route("/download/<session_id>/<filetype>")
def download_session(session_id, filetype):
    """Download analysis results."""
    if filetype == "json":
        path = _find_session_file(session_id)
        if path:
            return send_from_directory(str(SESSION_DIR), path.name,
                                       as_attachment=True)
    elif filetype in ("video", "original"):
        path = _find_session_file(session_id)
        if path:
            with open(path) as f:
                data = json.load(f)
            if filetype == "video":
                vpath = data.get("output_video_path")
            else:
                vpath = data.get("video_path")
            if vpath and os.path.exists(vpath):
                return send_from_directory(str(Path(vpath).parent),
                                           Path(vpath).name,
                                           mimetype=None,
                                           as_attachment=False)
    return render_template("error.html", message="File not found"), 404


@app.route("/camera-guide")
def camera_guide():
    """Camera positioning guide page."""
    return render_template("camera_guide.html")


# ---------------------------------------------------------------------------
# Background Analysis
# ---------------------------------------------------------------------------

def _run_analysis(job_id, video_path, batting_hand, ball_color, camera_view):
    """Run analysis in background."""
    job = analysis_jobs.get(job_id)
    if not job:
        return

    def progress_cb(current, total, status):
        if job_id in analysis_jobs:
            pct = int((current / max(total, 1)) * 100)
            analysis_jobs[job_id]["progress"] = pct
            analysis_jobs[job_id]["message"] = f"{status}: frame {current}/{total}"

    try:
        job["status"] = "processing"
        job["message"] = "Initializing analysis..."

        # Dynamic frame_step: longer videos can skip more frames to keep
        # analysis time reasonable on Render's free tier (512 MB, shared CPU).
        temp_cap = cv2.VideoCapture(video_path)
        _tf = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _fps = temp_cap.get(cv2.CAP_PROP_FPS)
        temp_cap.release()
        _dur = _tf / max(_fps, 1)
        if _dur > 120:
            frame_step = 4
        elif _dur > 60:
            frame_step = 3
        else:
            frame_step = 2

        analyser = BattingAnalyser(
            batting_hand=batting_hand,
            ball_color=ball_color,
            camera_view=camera_view,
            frame_step=frame_step,
        )
        # Disable server-side video generation: mp4v codec isn't playable
        # on mobile browsers, and H.264/avc1 requires libx264 which isn't
        # available on the Render slim image.  The original video + pose
        # overlay on the frontend is a better experience anyway.
        result = analyser.analyse_video(
            video_path=video_path,
            output_dir=str(SESSION_DIR),
            generate_video=False,
            progress_callback=progress_cb,
            share_token=job.get("share_token", ""),
        )
        analyser.close()

        result["camera_view"] = camera_view
        result["batting_hand"] = batting_hand
        result["ball_color"] = ball_color
        result["session_label"] = job.get("session_label", "")
        result["session_code"] = job.get("session_code", "")

        if result["success"]:
            job["status"] = "completed"
            job["progress"] = 100
            job["message"] = "Analysis complete!"
            job["result"] = result

            # Update the saved JSON with extra fields added after analysis
            result_path = result.get("result_path")
            if result_path and os.path.exists(result_path):
                try:
                    with open(result_path) as f:
                        saved = json.load(f)
                    saved["session_label"] = result["session_label"]
                    saved["session_code"] = result["session_code"]
                    with open(result_path, "w") as f:
                        json.dump(saved, f, indent=2, default=str)
                except Exception as exc:
                    print(f"[save_session] Warning: could not update JSON: {exc}")
        else:
            job["status"] = "failed"
            job["error"] = result.get("error", "Unknown error")
            job["message"] = "Analysis failed"
    except Exception as e:
        job["status"] = "failed"
        # Capture full traceback in error message for debugging
        import io, traceback as tb_mod
        tb_buf = io.StringIO()
        tb_mod.print_exc(file=tb_buf)
        job["error"] = f"[{type(e).__name__}] {str(e)}\n{tb_buf.getvalue()}"
        job["message"] = f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_sessions():
    """List saved sessions."""
    sessions = []
    for f in sorted(SESSION_DIR.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            with open(f) as fh:
                data = json.load(fh)
            sessions.append({
                "id": data.get("session_id", f.stem),
                "video_name": Path(data.get("video_path", "")).name or f.stem,
                "date": data.get("analysis_timestamp", ""),
                "duration": data.get("duration_sec", 0),
                "shots": data.get("num_shots_detected", 0),
                "total_frames": data.get("total_frames", 0),
                "has_video": bool(data.get("output_video_path")),
                "tip_count": len(data.get("coaching_tips", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return sessions


def _compute_stats(sessions):
    if not sessions:
        return {"total_sessions": 0, "total_shots": 0, "total_duration": 0}
    return {
        "total_sessions": len(sessions),
        "total_shots": sum(s.get("shots", 0) for s in sessions),
        "total_duration": round(sum(s.get("duration", 0) for s in sessions), 1),
    }


# ---------------------------------------------------------------------------
# SaaS Routes — Payments, webhooks, and account management
# ---------------------------------------------------------------------------

@app.route("/stripe/create-checkout")
@login_required
def stripe_checkout():
    """Redirect user to Stripe Checkout for a Pro subscription."""
    from flask import session as flask_session
    user_id = flask_session.get("user_id")
    email = flask_session.get("user_email")
    tier = request.args.get("tier", "pro")

    if not is_stripe_configured():
        flash("Payments are not configured yet.", "warning")
        return redirect(url_for("auth.profile"))

    url = create_checkout_session(user_id, email, tier)
    if url:
        return redirect(url)
    else:
        flash("Could not create checkout session. Please try again.", "error")
        return redirect(url_for("auth.profile"))


@app.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    """Handle Stripe webhook events (subscription lifecycle)."""
    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")

    result = handle_webhook(payload, sig_header)
    if result["success"]:
        return jsonify({"received": True}), 200
    else:
        return jsonify({"error": result["message"]}), 400


@app.route("/stripe/portal")
@login_required
def stripe_portal():
    """Redirect user to Stripe Customer Portal."""
    from flask import session as flask_session
    from supabase_client import get_supabase

    user_id = flask_session.get("user_id")
    supabase = get_supabase()
    if supabase:
        try:
            resp = supabase.table("profiles")\
                .select("stripe_customer_id")\
                .eq("id", user_id)\
                .execute()
            if resp.data and resp.data[0].get("stripe_customer_id"):
                from stripe_payments import create_portal_session
                url = create_portal_session(resp.data[0]["stripe_customer_id"])
                if url:
                    return redirect(url)
        except Exception:
            pass

    flash("Could not open billing portal.", "error")
    return redirect(url_for("auth.profile"))


# ---------------------------------------------------------------------------
# Usage limit check decorator for analysis routes
# ---------------------------------------------------------------------------

def check_analysis_limit(view):
    """Decorator: check if user has remaining analyses before uploading."""
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        from flask import session as flask_session
        # Skip check in dev mode
        if os.environ.get("DEBUG_AUTH") == "1":
            return view(**kwargs)
        # Check supabase usage
        if supabase_configured():
            user_id = flask_session.get("user_id")
            if user_id:
                remaining = get_usage_remaining(user_id)
                if remaining <= 0:
                    flash(
                        "You've used all your analyses this month. "
                        "Upgrade to Pro for unlimited analyses.",
                        "warning"
                    )
                    return redirect(url_for("auth.profile"))
        return view(**kwargs)
    return wrapped_view


# ---------------------------------------------------------------------------
# Scoring App (PWA) — served at /scoring/
# ---------------------------------------------------------------------------

@app.route("/scoring/")
@app.route("/scoring/<path:filename>")
def scoring_app(filename=None):
    """Serve the Scoring App PWA under /scoring/ path."""
    scoring_dir = BASE_DIR / "static" / "scoring"
    if filename is None:
        filename = "index.html"
    file_path = scoring_dir / filename
    if not file_path.exists() or not file_path.is_file():
        filename = "index.html"
        file_path = scoring_dir / "index.html"
    # Determine mimetype for service worker and manifest
    mimetype = None
    if filename == "sw.js":
        mimetype = "application/javascript"
    elif filename == "manifest.json":
        mimetype = "application/manifest+json"
    return send_from_directory(str(scoring_dir),
                               filename,
                               mimetype=mimetype)


# ---------------------------------------------------------------------------
# PWA / Manifest (for the main Batting Lab app)
# ---------------------------------------------------------------------------

@app.route("/manifest.json")
def manifest():
    return send_from_directory(str(BASE_DIR / "static" / "manifest"),
                               "manifest.json",
                               mimetype="application/manifest+json")


@app.route("/sw.js")
def service_worker():
    return send_from_directory(str(BASE_DIR / "static"),
                               "sw.js",
                               mimetype="application/javascript")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    HOST = os.environ.get("HOST", "127.0.0.1")
    PORT = int(os.environ.get("PORT", 5005))
    DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")

    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║        🏏  the CREASE Batting Lab        ║")
    print("  ║   Zero-cost AI Cricket Batting Analysis  ║")
    print("  ╚══════════════════════════════════════════╝")
    print()
    print(f"  🌐  Server running:  http://{HOST}:{PORT}")
    print("  📱  Android (PWA):    Chrome → Add to Home Screen")
    print("  📱  Android (local):  Install via Termux (see docs)")
    print()
    print("  Upload a batting video and get:")
    print("  •  Pose estimation & skeleton tracking")
    print("  •  Ball trajectory & speed")
    print("  •  Bat swing path & speed analysis")
    print("  •  Shot phase detection (7 phases)")
    print("  •  Joint angle biomechanics")
    print("  •  Automated coaching tips")
    print("  •  Session comparison & progress tracking")
    print()
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
