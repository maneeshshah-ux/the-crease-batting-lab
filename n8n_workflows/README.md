# n8n Workflows for the CREASE Batting Lab

These workflows automate key processes. Import them into your n8n instance.

## Setup

1. Deploy n8n (included in `docker-compose.yml` on port 5678).
2. Create credentials in n8n:
   - **Supabase API** (use your service key)
   - **Stripe API** (use your secret key)
   - **SMTP** (for emails — use Amazon SES for low cost)
3. Import each workflow JSON file.
4. Update webhook URLs to point to your CREASE API domain.

## Workflows

### 1. `analysis_complete.json`
**Trigger**: Webhook from CREASE API when analysis finishes.
**Actions**:
- Look up user email from Supabase
- Send "Your analysis is ready!" email with link to results
- If Pro user, generate and attach PDF report
- Log to analytics

### 2. `new_signup.json`
**Trigger**: Webhook from Supabase Auth (new user signup).
**Actions**:
- Send welcome email with getting-started guide
- Add user to mailing list (listmonk)
- Create onboarding checklist in DB

### 3. `subscription_event.json`
**Trigger**: Stripe webhook events.
**Actions**:
- On `checkout.session.completed`: activate Pro features
- On `invoice.past_due`: send payment reminder
- On `customer.subscription.deleted`: downgrade to Free

### 4. `monthly_report.json`
**Trigger**: Cron (every 30 days).
**Actions**:
- For each active user, aggregate their sessions
- Send monthly progress digest email with charts
- Reset free-tier usage counter
