"""
Stripe Payments — Subscription management for the CREASE Batting Lab.

Provides:
  - create_checkout_session(user_id, tier) → redirect URL
  - handle_webhook(payload, sig_header) → updates subscription
  - get_portal_session(stripe_customer_id) → customer portal URL

Tiers:
  - free:    3 analyses/month, no video storage
  - pro:     unlimited analyses, full storage + reports, multi-player
  - enterprise: everything + priority support + custom branding
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional

# Import stripe conditionally — app works without it in dev mode
try:
    import stripe
    _stripe_available = True
except ImportError:
    stripe = None
    _stripe_available = False

from supabase_client import get_supabase

# ---------------------------------------------------------------------------
# Stripe config
# ---------------------------------------------------------------------------
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_FREE = os.environ.get("STRIPE_PRICE_FREE", "price_free")
STRIPE_PRICE_PRO = os.environ.get("STRIPE_PRICE_PRO", "price_pro")
STRIPE_PRICE_ENTERPRISE = os.environ.get("STRIPE_PRICE_ENTERPRISE", "price_enterprise")

if stripe:
    stripe.api_key = STRIPE_SECRET_KEY

# Tier → price ID mapping
TIER_PRICES = {
    "pro": STRIPE_PRICE_PRO,
    "enterprise": STRIPE_PRICE_ENTERPRISE,
}

TIER_ANALYSIS_LIMITS = {
    "free": 3,
    "pro": 999999,
    "enterprise": 999999,
}

DOMAIN = os.environ.get("SITE_DOMAIN", "http://localhost:5005")


def is_stripe_configured() -> bool:
    """Check if Stripe is configured for payments."""
    return bool(STRIPE_SECRET_KEY) and STRIPE_SECRET_KEY != ""


def create_checkout_session(user_id: str, email: str, tier: str = "pro") -> Optional[str]:
    """
    Create a Stripe Checkout Session for a subscription.

    Args:
        user_id: Supabase user ID
        email: User's email address
        tier: Subscription tier ('pro' or 'enterprise')

    Returns:
        URL to redirect the user to Stripe Checkout, or None on failure.
    """
    if not is_stripe_configured():
        return None

    price_id = TIER_PRICES.get(tier)
    if not price_id:
        return None

    try:
        # Find or create Stripe customer
        supabase = get_supabase()
        customer_id = None
        if supabase:
            try:
                resp = supabase.table("profiles")\
                    .select("stripe_customer_id")\
                    .eq("id", user_id)\
                    .execute()
                if resp.data and resp.data[0].get("stripe_customer_id"):
                    customer_id = resp.data[0]["stripe_customer_id"]
            except Exception:
                pass

        if not customer_id:
            customer = stripe.Customer.create(
                email=email,
                metadata={"supabase_user_id": user_id},
            )
            customer_id = customer.id
            # Store customer ID
            if supabase:
                try:
                    supabase.table("profiles")\
                        .update({"stripe_customer_id": customer_id})\
                        .eq("id", user_id)\
                        .execute()
                except Exception:
                    pass

        session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{DOMAIN}/auth/profile?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{DOMAIN}/auth/profile",
            metadata={
                "user_id": user_id,
                "tier": tier,
            },
            subscription_data={
                "metadata": {
                    "user_id": user_id,
                    "tier": tier,
                },
            },
        )

        return session.url

    except Exception as e:
        print(f"[Stripe] Checkout session error: {e}")
        return None


def create_portal_session(stripe_customer_id: str) -> Optional[str]:
    """
    Create a Stripe Customer Portal session so users can manage their subscription.

    Args:
        stripe_customer_id: The Stripe customer ID

    Returns:
        URL to redirect the user to the Stripe Customer Portal, or None.
    """
    if not is_stripe_configured() or not stripe_customer_id:
        return None

    try:
        session = stripe.billing_portal.Session.create(
            customer=stripe_customer_id,
            return_url=f"{DOMAIN}/auth/profile",
        )
        return session.url
    except Exception as e:
        print(f"[Stripe] Portal session error: {e}")
        return None


def handle_webhook(payload: bytes, sig_header: str) -> dict:
    """
    Handle a Stripe webhook event.

    Verifies the signature, processes the event, and updates Supabase.

    Returns:
        dict with {"success": bool, "message": str}
    """
    if not is_stripe_configured() or not STRIPE_WEBHOOK_SECRET:
        return {"success": False, "message": "Stripe not configured"}

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        return {"success": False, "message": "Invalid payload"}
    except stripe.error.SignatureVerificationError:
        return {"success": False, "message": "Invalid signature"}

    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    supabase = get_supabase()
    if not supabase:
        return {"success": False, "message": "Supabase not configured"}

    try:
        if event_type == "checkout.session.completed":
            metadata = data.get("metadata", {})
            user_id = metadata.get("user_id")
            tier = metadata.get("tier", "pro")
            customer_id = data.get("customer")
            subscription_id = data.get("subscription")

            if user_id:
                # Update profile
                supabase.table("profiles").update({
                    "subscription_tier": tier,
                    "subscription_status": "active",
                    "stripe_customer_id": customer_id,
                    "analyses_limit": TIER_ANALYSIS_LIMITS.get(tier, 3),
                }).eq("id", user_id).execute()

                # Create subscription record
                line_items = data.get("line_items", {}).get("data", [])
                price_id = line_items[0]["price"]["id"] if line_items else ""

                supabase.table("subscriptions").upsert({
                    "user_id": user_id,
                    "stripe_subscription_id": subscription_id,
                    "stripe_price_id": price_id,
                    "tier": tier,
                    "status": "active",
                    "current_period_start": datetime.utcnow().isoformat(),
                    "current_period_end": (datetime.utcnow() + timedelta(days=30)).isoformat(),
                }).execute()

        elif event_type == "invoice.paid":
            subscription_id = data.get("subscription")
            # Extend the subscription period
            if subscription_id and supabase:
                supabase.table("subscriptions").update({
                    "status": "active",
                    "current_period_end": (datetime.utcnow() + timedelta(days=30)).isoformat(),
                }).eq("stripe_subscription_id", subscription_id).execute()

        elif event_type == "customer.subscription.updated":
            subscription_id = data.get("id")
            status = data.get("status")
            cancel_at_period_end = data.get("cancel_at_period_end", False)

            if subscription_id and supabase:
                supabase.table("subscriptions").update({
                    "status": status,
                    "cancel_at_period_end": cancel_at_period_end,
                }).eq("stripe_subscription_id", subscription_id).execute()

                # Also update profile status
                if status == "past_due":
                    sub_data = supabase.table("subscriptions")\
                        .select("user_id").eq("stripe_subscription_id", subscription_id).execute()
                    if sub_data.data:
                        supabase.table("profiles").update({
                            "subscription_status": "past_due",
                        }).eq("id", sub_data.data[0]["user_id"]).execute()

        elif event_type == "customer.subscription.deleted":
            subscription_id = data.get("id")
            if subscription_id and supabase:
                sub_data = supabase.table("subscriptions")\
                    .select("user_id").eq("stripe_subscription_id", subscription_id).execute()
                supabase.table("subscriptions").update({
                    "status": "cancelled",
                }).eq("stripe_subscription_id", subscription_id).execute()
                if sub_data.data:
                    supabase.table("profiles").update({
                        "subscription_tier": "free",
                        "subscription_status": "inactive",
                        "analyses_limit": 3,
                    }).eq("id", sub_data.data[0]["user_id"]).execute()

    except Exception as e:
        print(f"[Stripe] Webhook processing error: {e}")
        return {"success": False, "message": str(e)}

    return {"success": True, "message": f"Processed {event_type}"}


def get_usage_remaining(user_id: str) -> int:
    """Check how many analyses a user has remaining this billing period."""
    supabase = get_supabase()
    if not supabase:
        return 0

    try:
        resp = supabase.table("profiles")\
            .select("analyses_used, analyses_limit")\
            .eq("id", user_id)\
            .execute()
        if resp.data:
            p = resp.data[0]
            used = p.get("analyses_used", 0)
            limit = p.get("analyses_limit", 3)
            return max(0, limit - used)
    except Exception:
        pass
    return 0


def increment_usage(user_id: str) -> bool:
    """Increment the user's analysis counter."""
    supabase = get_supabase()
    if not supabase:
        return False

    try:
        supabase.rpc("increment_analyses_used", {"user_id": user_id}).execute()
        return True
    except Exception:
        # Fallback: direct update
        try:
            resp = supabase.table("profiles")\
                .select("analyses_used")\
                .eq("id", user_id)\
                .execute()
            if resp.data:
                current = resp.data[0].get("analyses_used", 0)
                supabase.table("profiles")\
                    .update({"analyses_used": current + 1})\
                    .eq("id", user_id)\
                    .execute()
                return True
        except Exception:
            pass
    return False
