# Deploy the CREASE Batting Lab (Commercial SaaS)

Choose your deployment path:

## 🚀 Option A: Coolify (Self-Hosted, Recommended)

1. Provision a **$10–$20/mo VPS** (Hetzner, DigitalOcean, or AWS Lightsail).
2. Install **Coolify** on it.
3. In Coolify dashboard:
   - Add your GitHub repo
   - Set `Docker Compose` as the build pack
   - Point to `docker-compose.yml`
4. Set the environment variables from `.env.example`.
5. Deploy!

**Estimated cost**: $10–$20/mo VPS + $0 for self-hosted services.

## 🚀 Option B: Render (Simpler, Free to Start)

1. Register at [render.com](https://render.com) (free tier).
2. Click **New +** → **Blueprint**.
3. Connect your GitHub repo containing this project.
4. Render auto-detects `render.yaml`.
5. Add environment variables:
   - `SUPABASE_URL`, `SUPABASE_ANON_KEY`
   - `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`
   - `SECRET_KEY` (generate a random one)
6. Deploy.

**Limitations**: Ephemeral storage (videos lost on restart), 512MB RAM.
**Upgrade**: $7/mo for 1GB RAM + persistent disk.

## 🚀 Option C: Railway (Alternative to Render)

1. Register at [railway.app](https://railway.app) (free tier with $5 credit).
2. Click **New Project** → **Deploy from GitHub repo**.
3. Same env vars as Render above.

## 🚀 Option D: Fly.io (Global Edge)

1. Install `flyctl`.
2. Run `fly launch` in this directory.
3. Set env vars with `fly secrets set`.

---

# Post-Deployment Checklist

- [ ] Run `supabase_schema.sql` in Supabase SQL Editor
- [ ] Create Stripe products/prices (Pro: $29/mo, Enterprise: $99/mo)
- [ ] Set Stripe webhook to `https://your-domain.com/stripe/webhook`
- [ ] Add a custom domain
- [ ] Configure Plausible analytics
- [ ] Set up n8n workflows (email notifications, report delivery)
- [ ] Test the full flow: signup → login → upload → analysis → payment
