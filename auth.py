"""
Authentication Blueprint — Signup, login, logout, profile, and account management.

Uses Supabase Auth for credential management and the profiles table for
user metadata and subscription tier tracking.

Works in two modes:
  1. Supabase configured: full auth flow with email/password or magic link
  2. No Supabase: single-user "developer mode" with a bypass (DEBUG_AUTH=1)
"""

import os
import functools
from datetime import datetime
from typing import Optional

from flask import (Blueprint, render_template, request, redirect,
                   url_for, flash, session as flask_session, jsonify, g)

from supabase_client import get_supabase, is_configured

# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------
auth_bp = Blueprint("auth", __name__, url_prefix="/auth",
                    template_folder="templates/auth")


def login_required(view):
    """Decorator: require a logged-in user. Redirects to login if not authenticated."""
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if not _is_authenticated():
            return redirect(url_for("auth.login", next=request.path))
        return view(**kwargs)
    return wrapped_view


def subscription_required(min_tier="pro"):
    """Decorator: require at least a given subscription tier."""
    def decorator(view):
        @functools.wraps(view)
        @login_required
        def wrapped_view(**kwargs):
            tier = flask_session.get("subscription_tier", "free")
            tier_rank = {"free": 0, "pro": 1, "enterprise": 2}
            if tier_rank.get(tier, 0) < tier_rank.get(min_tier, 1):
                flash(f"This feature requires a {min_tier} subscription.", "warning")
                return redirect(url_for("auth.profile"))
            return view(**kwargs)
        return wrapped_view
    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_authenticated() -> bool:
    """Check if current request has a valid user session."""
    # Developer bypass
    if os.environ.get("DEBUG_AUTH") == "1":
        return True
    # Supabase session check
    return bool(flask_session.get("user_id"))


def _load_user_profile(user_id: str) -> Optional[dict]:
    """Fetch user profile from Supabase profiles table."""
    supabase = get_supabase()
    if not supabase:
        return None
    try:
        resp = supabase.table("profiles").select("*").eq("id", user_id).execute()
        if resp.data and len(resp.data) > 0:
            return resp.data[0]
    except Exception:
        pass
    return None


def _sync_session_from_supabase(access_token: str, refresh_token: str = "") -> bool:
    """Given valid Supabase tokens, load the user profile into Flask session."""
    supabase = get_supabase()
    if not supabase:
        return False

    try:
        # Set the session so subsequent calls are authenticated
        supabase.auth.set_session(access_token, refresh_token)
        user = supabase.auth.get_user()
        if not user or not user.user:
            return False

        user_id = user.user.id
        email = user.user.email or ""

        # Ensure profile exists
        profile = _load_user_profile(user_id)
        if not profile:
            # Create profile on first login
            try:
                supabase.table("profiles").insert({
                    "id": user_id,
                    "email": email,
                    "role": "player",
                    "subscription_tier": "free",
                    "subscription_status": "inactive",
                    "analyses_used": 0,
                    "analyses_limit": 3,
                }).execute()
                profile = _load_user_profile(user_id)
            except Exception:
                pass

        # Populate Flask session
        flask_session["user_id"] = user_id
        flask_session["user_email"] = email
        flask_session["access_token"] = access_token
        flask_session["refresh_token"] = refresh_token

        if profile:
            flask_session["subscription_tier"] = profile.get("subscription_tier", "free")
            flask_session["subscription_status"] = profile.get("subscription_status", "inactive")
            flask_session["user_role"] = profile.get("role", "player")
            flask_session["analyses_used"] = profile.get("analyses_used", 0)
            flask_session["analyses_limit"] = profile.get("analyses_limit", 3)

        flask_session.permanent = True
        return True

    except Exception as e:
        print(f"[Auth] Session sync failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Before request — inject user into g
# ---------------------------------------------------------------------------

@auth_bp.before_app_request
def load_logged_in_user():
    """Make user data available to all templates via g."""
    g.user = None
    if _is_authenticated():
        g.user = {
            "id": flask_session.get("user_id"),
            "email": flask_session.get("user_email"),
            "role": flask_session.get("user_role", "player"),
            "subscription_tier": flask_session.get("subscription_tier", "free"),
            "subscription_status": flask_session.get("subscription_status", "inactive"),
            "analyses_used": flask_session.get("analyses_used", 0),
            "analyses_limit": flask_session.get("analyses_limit", 3),
        }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Login page — email/password or magic link via Supabase."""
    if _is_authenticated():
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            error = "Email and password are required."
        else:
            supabase = get_supabase()
            if not supabase:
                error = "Authentication is not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY."
            else:
                try:
                    resp = supabase.auth.sign_in_with_password({
                        "email": email,
                        "password": password,
                    })
                    if resp.user:
                        access_token = resp.session.access_token
                        refresh_token = resp.session.refresh_token
                        if _sync_session_from_supabase(access_token, refresh_token):
                            flash("Welcome back!", "success")
                            next_page = request.args.get("next") or url_for("index")
                            return redirect(next_page)
                        else:
                            error = "Could not load your profile."
                    else:
                        error = "Invalid email or password."
                except Exception as e:
                    error = str(e)

    return render_template("auth/login.html", error=error,
                           supabase_configured=is_configured())


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    """Signup page — creates Supabase user + profile."""
    if _is_authenticated():
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        full_name = request.form.get("full_name", "").strip()
        role = request.form.get("role", "player")

        if not email or not password:
            error = "Email and password are required."
        elif password != confirm:
            error = "Passwords do not match."
        elif len(password) < 8:
            error = "Password must be at least 8 characters."
        elif role not in ("coach", "player"):
            error = "Invalid role selected."
        else:
            supabase = get_supabase()
            if not supabase:
                error = "Authentication is not configured."
            else:
                try:
                    resp = supabase.auth.sign_up({
                        "email": email,
                        "password": password,
                        "options": {
                            "data": {
                                "full_name": full_name,
                                "role": role,
                            }
                        }
                    })
                    if resp.user:
                        # Create profile
                        try:
                            supabase.table("profiles").insert({
                                "id": resp.user.id,
                                "email": email,
                                "full_name": full_name,
                                "role": role,
                                "subscription_tier": "free",
                                "subscription_status": "inactive",
                                "analyses_used": 0,
                                "analyses_limit": 3,
                            }).execute()
                        except Exception:
                            pass  # profile may be created by DB trigger

                        flash("Account created! Check your email for verification.", "success")
                        return redirect(url_for("auth.login"))
                    else:
                        error = "Could not create account."
                except Exception as e:
                    error = str(e)

    return render_template("auth/signup.html", error=error,
                           supabase_configured=is_configured())


@auth_bp.route("/logout")
def logout():
    """Log out — clear Flask session and invalidate Supabase session."""
    supabase = get_supabase()
    if supabase:
        try:
            supabase.auth.sign_out()
        except Exception:
            pass

    flask_session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))


@auth_bp.route("/profile")
@login_required
def profile():
    """User profile and subscription management page."""
    user_id = flask_session.get("user_id")
    profile = _load_user_profile(user_id)

    # Get recent sessions
    sessions = []
    supabase = get_supabase()
    if supabase and user_id:
        try:
            resp = supabase.table("sessions").select("*")\
                .eq("user_id", user_id)\
                .order("created_at", desc=True)\
                .limit(10)\
                .execute()
            sessions = resp.data or []
        except Exception:
            pass

    return render_template("auth/profile.html",
                           profile=profile,
                           sessions=sessions,
                           user=g.get("user", {}))


@auth_bp.route("/magic-link", methods=["POST"])
def magic_link():
    """Send a magic link (passwordless login) email."""
    email = request.form.get("email", "").strip()
    if not email:
        flash("Email is required.", "error")
        return redirect(url_for("auth.login"))

    supabase = get_supabase()
    if supabase:
        try:
            supabase.auth.sign_in_with_otp({
                "email": email,
                "options": {"redirect_to": url_for("auth.login", _external=True)}
            })
            flash(f"Magic link sent to {email}. Check your inbox!", "success")
        except Exception as e:
            flash(str(e), "error")
    else:
        flash("Authentication is not configured.", "error")

    return redirect(url_for("auth.login"))


@auth_bp.route("/callback")
def callback():
    """Handle OAuth/callback redirects from Supabase."""
    access_token = request.args.get("access_token")
    refresh_token = request.args.get("refresh_token")

    if access_token:
        if _sync_session_from_supabase(access_token, refresh_token or ""):
            flash("Logged in successfully!", "success")
            return redirect(url_for("index"))
        else:
            flash("Authentication failed.", "error")

    return redirect(url_for("auth.login"))
