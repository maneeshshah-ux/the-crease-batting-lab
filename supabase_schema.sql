-- =============================================================================
-- the CREASE Batting Lab — Supabase Database Schema
-- =============================================================================
-- Run this in the Supabase SQL Editor to set up all tables, indexes,
-- triggers, and Row-Level Security policies.
--
-- Usage:
--   1. Go to your Supabase project → SQL Editor
--   2. Paste this entire file
--   3. Click "Run"
-- =============================================================================

-- 1. Extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- PROFILES — extends Supabase auth.users
-- =============================================================================
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

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Auto-create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name, role)
    VALUES (
        NEW.id,
        NEW.email,
        NEW.raw_user_meta_data->>'full_name',
        COALESCE(NEW.raw_user_meta_data->>'role', 'player')
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- RLS: users read/update own profile, admins read all
CREATE POLICY "Users view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users update own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id)
    WITH CHECK (auth.uid() = id);

-- Function to increment analyses_used
CREATE OR REPLACE FUNCTION public.increment_analyses_used(p_user_id UUID)
RETURNS void AS $$
BEGIN
    UPDATE public.profiles
    SET analyses_used = analyses_used + 1
    WHERE id = p_user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================================================
-- SESSIONS — per-user analysis records
-- =============================================================================
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
    shot_summary JSONB,
    bowling_analysis JSONB,       -- BowlingAnalyzer output (bowl type, arm speed, release height, deliveries)
    session_code TEXT,            -- Multi-camera sync code (6-char, free tier structural)
    status TEXT NOT NULL DEFAULT 'processing' CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON public.sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON public.sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_session_code ON public.sessions(session_code);

ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users manage own sessions"
    ON public.sessions FOR ALL
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- =============================================================================
-- PLAYERS — coach-managed player registry
-- =============================================================================
CREATE TABLE IF NOT EXISTS public.players (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    coach_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    name TEXT,
    label TEXT,
    age_group TEXT,
    batting_hand TEXT DEFAULT 'right',
    stance_signature JSONB,
    historical_metrics JSONB,
    session_ids UUID[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_players_coach_id ON public.players(coach_id);

ALTER TABLE public.players ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Coaches manage own players"
    ON public.players FOR ALL
    USING (auth.uid() = coach_id)
    WITH CHECK (auth.uid() = coach_id);

-- =============================================================================
-- SUBSCRIPTIONS — mirrors Stripe data for fast access
-- =============================================================================
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

-- =============================================================================
-- STORAGE BUCKETS
-- =============================================================================
-- Create a bucket for analysis videos (run in Storage section or via API)
-- INSERT INTO storage.buckets (id, name, public) VALUES ('videos', 'videos', false);
-- INSERT INTO storage.buckets (id, name, public) VALUES ('reports', 'reports', false);

-- =============================================================================
-- USAGE RESET FUNCTION (run via cron/scheduler monthly)
-- =============================================================================
CREATE OR REPLACE FUNCTION public.reset_monthly_usage()
RETURNS void AS $$
BEGIN
    -- Only reset free-tier users; pro/enterprise keep their higher limits
    UPDATE public.profiles
    SET analyses_used = 0
    WHERE subscription_tier = 'free'
      AND subscription_status = 'inactive';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
