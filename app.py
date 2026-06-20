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
from datetime import datetime
from pathlib import Path

from flask import (Flask, render_template, request, jsonify,
                   send_from_directory, url_for, redirect, session,
                   send_file)
from werkzeug.utils import secure_filename

from engine.analyser import BattingAnalyser

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

# Trust X-Forwarded-* headers from Render/Railway proxies
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# In-memory job tracking
analysis_jobs = {}


def allowed_file(filename):
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


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
    """Upload a batting video for analysis."""
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
        "result": None,
        "error": None,
    }

    thread = threading.Thread(
        target=_run_analysis,
        args=(job_id, filepath, batting_hand, ball_color, camera_view),
        daemon=True,
    )
    thread.start()

    return render_template("processing.html", job_id=job_id, video_name=filename)


@app.route("/job/<job_id>")
def job_status(job_id):
    """Job status page."""
    job = analysis_jobs.get(job_id)
    if not job:
        return render_template("error.html", message="Analysis job not found"), 404
    if job["status"] == "completed" and job["result"]:
        return redirect(url_for("session_view", session_id=job["result"]["session_id"]))
    return render_template("processing.html", job_id=job_id,
                           video_name=job.get("video_name", ""))


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
    """View analysis results."""
    session_path = SESSION_DIR / f"{session_id}.json"
    if not session_path.exists():
        return render_template("error.html",
                               message=f"Session {session_id} not found"), 404
    with open(session_path) as f:
        data = json.load(f)
    return render_template("session_detail.html", session=data)


@app.route("/sessions")
def sessions_list():
    """List all sessions."""
    sessions = _list_sessions()
    return render_template("sessions.html", sessions=sessions)


@app.route("/delete/<session_id>", methods=["POST"])
def delete_session(session_id):
    """Delete a single session."""
    path = SESSION_DIR / f"{session_id}.json"
    if path.exists():
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


@app.route("/download/<session_id>/<filetype>")
def download_session(session_id, filetype):
    """Download analysis results."""
    if filetype == "json":
        path = SESSION_DIR / f"{session_id}.json"
        if path.exists():
            return send_from_directory(str(SESSION_DIR), f"{session_id}.json",
                                       as_attachment=True)
    elif filetype == "video":
        path = SESSION_DIR / f"{session_id}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            vpath = data.get("output_video_path")
            if vpath and os.path.exists(vpath):
                return send_from_directory(str(Path(vpath).parent),
                                           Path(vpath).name,
                                           as_attachment=True)
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

        analyser = BattingAnalyser(
            batting_hand=batting_hand,
            ball_color=ball_color,
        )
        result = analyser.analyse_video(
            video_path=video_path,
            output_dir=str(SESSION_DIR),
            generate_video=True,
            progress_callback=progress_cb,
        )
        analyser.close()

        result["camera_view"] = camera_view
        result["batting_hand"] = batting_hand
        result["ball_color"] = ball_color

        if result["success"]:
            job["status"] = "completed"
            job["progress"] = 100
            job["message"] = "Analysis complete!"
            job["result"] = result
        else:
            job["status"] = "failed"
            job["error"] = result.get("error", "Unknown error")
            job["message"] = "Analysis failed"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["message"] = f"Error: {str(e)}"
        import traceback
        traceback.print_exc()


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
