"""
Supabase Client — Database, Auth, and Storage wrapper.

Provides a single point of initialisation for all Supabase interactions.
Set SUPABASE_URL and SUPABASE_SERVICE_KEY in environment variables.

Usage:
    from supabase_client import get_supabase
    supabase = get_supabase()
    # supabase.table("users").select("*").execute()
"""

import os
import functools
from typing import Optional

# Import supabase conditionally so the app can still run without it
try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

_SUPABASE_CLIENT = None


def get_supabase() -> Optional[Client]:
    """Get or create the Supabase client singleton."""
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    anon_key = os.environ.get("SUPABASE_ANON_KEY", "")

    if not url or (not key and not anon_key):
        return None  # Supabase not configured — run in offline/dev mode

    if create_client is None:
        raise ImportError(
            "supabase-py is not installed. Run: pip install supabase"
        )

    use_key = key or anon_key
    _SUPABASE_CLIENT = create_client(url, use_key)
    return _SUPABASE_CLIENT


def is_configured() -> bool:
    """Check if Supabase is configured (returns True even if client init fails)."""
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "") or os.environ.get("SUPABASE_ANON_KEY", "")
    return bool(url and key)


# ---------------------------------------------------------------------------
# Schema helpers — return SQL CREATE statements for reference
# ---------------------------------------------------------------------------

USERS_TABLE_SQL = """
-- Extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Profiles table (extends Supabase auth.users)
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT,
    full_name TEXT,
    avatar_url TEXT,
    role TEXT NOT NULL DEFAULT 'player' CHECK (role IN ('coach', 'player', 'admin')),
    subscription_tier TEXT NOT NULL DEFAULT 'free' CHECK (subscription_tier IN ('free', 'pro', 'enterprise')),
    subscription_status TEXT NOT NULL DEFAULT 'inactive' CHECK (subscription_status IN ('active', 'inactive', 'past_due', 'cancelled')),
    stripe_customer_id TEXT,
    analyses_used INTEGER DEFAULT 0,
    analyses_limit INTEGER DEFAULT 3,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Users can read their own profile; admins can read all
CREATE POLICY "Users view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users update own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id)
    WITH CHECK (auth.uid() = id);
"""

SESSIONS_TABLE_SQL = """
-- Analysis sessions table
CREATE TABLE IF NOT EXISTS public.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    session_label TEXT,
    video_path TEXT,
    video_name TEXT,
    duration_sec REAL,
    num_shots INTEGER,
    session_score REAL,
    batting_hand TEXT,
    camera_view TEXT,
    ball_color TEXT,
    stance_signature JSONB,
    summary_metrics JSONB,
    coaching_tips JSONB,
    status TEXT NOT NULL DEFAULT 'processing' CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Index for fast per-user lookups
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON public.sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON public.sessions(created_at DESC);

ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;

-- Users can CRUD their own sessions only
CREATE POLICY "Users manage own sessions"
    ON public.sessions FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);
"""

PLAYERS_TABLE_SQL = """
-- Player registry (managed by coaches)
CREATE TABLE IF NOT EXISTS public.players (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    coach_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    name TEXT,
    label TEXT,
    age_group TEXT,
    batting_hand TEXT,
    stance_signature JSONB,
    historical_metrics JSONB,
    session_ids UUID[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_players_coach_id ON public.players(coach_id);

ALTER TABLE public.players ENABLE ROW LEVEL SECURITY;

-- Coaches manage their own players
CREATE POLICY "Coaches manage own players"
    ON public.players FOR ALL
    USING (auth.uid() = coach_id)
    WITH CHECK (auth.uid() = coach_id);
"""

SUBSCRIPTIONS_TABLE_SQL = """
-- Subscription tracking (mirrors Stripe for fast access)
CREATE TABLE IF NOT EXISTS public.subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    stripe_subscription_id TEXT UNIQUE,
    stripe_price_id TEXT,
    tier TEXT NOT NULL,
    status TEXT NOT NULL,
    current_period_start TIMESTAMPTZ,
    current_period_end TIMESTAMPTZ,
    cancel_at_period_end BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id ON public.subscriptions(user_id);

ALTER TABLE public.subscriptions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users view own subscription"
    ON public.subscriptions FOR SELECT
    USING (auth.uid() = user_id);
"""

FULL_SCHEMA_SQL = (
    USERS_TABLE_SQL + "\n\n" + SESSIONS_TABLE_SQL + "\n\n"
    + PLAYERS_TABLE_SQL + "\n\n" + SUBSCRIPTIONS_TABLE_SQL
)
