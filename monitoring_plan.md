# Monitoring & Responsible-Use Plan — Churn Scoring API (Part 4)

This API puts a 60-day churn score in front of the retention team, and that score ends up steering real money
toward real customers. So once it's live it can't just be left alone — it needs watching, and it needs a few
rules about how people are allowed to use it. Here's how I'd keep an eye on it and where I'd draw the lines.

## 1. Are the inputs still what the model expects? (data drift)
- **What I'd watch:** how each input feature's distribution compares to the 2025-09-30 training snapshot.
- **How:** a weekly Population Stability Index on the features that matter most — `recency_days`,
  `last_visit_days_ago`, `frequency_180d`, `monetary_180d`, `return_rate_180d`, `negative_ticket_rate_90d` —
  plus a check on null rates and the category mix in the profile fields.
- **When to worry:** PSI above 0.2 on any top driver, a categorical value the model never saw in training, or
  a feature's null rate more than doubling.

## 2. Are the predictions themselves shifting? (output drift)
- **What I'd watch:** the day-to-day spread of `churn_probability` and how many customers land in
  `high` / `medium` / `low`.
- **Why it's useful:** the scores usually move *before* we have any churn labels to check against, so a sudden
  jump is an early warning that something upstream broke.
- **When to worry:** the average predicted probability drifts by more than ±0.10 against the trailing four
  weeks, or the `high` bucket's share swings by more than half.

## 3. Is it actually any good in the wild? (business outcomes)
- **What I'd watch:** once each 60-day window closes, recompute precision/recall/F1/ROC-AUC of the old scores
  against what really happened, and compare retention for customers we contacted vs didn't within each risk band.
- **Why it matters:** offline test numbers go stale fast — the only honest measure is how it does on live data.
- **When to worry:** realised recall on churners slips below ~0.80, precision drops far enough that outreach
  stops paying for itself, or an A/B holdout shows we aren't actually saving anyone.

## 4. Is the service healthy? (operations)
- **What I'd watch:** request volume, p50/p95 latency, the 4xx rate (especially 422 validation failures), the
  5xx rate, and the `model_loaded` flag on `/health`.
- **Why:** a surge in 422s usually means a caller changed their payload format; a 503 with
  `model_loaded=false` means the model artifact didn't load.
- **When to worry:** 5xx above 1%, a sustained 422 rate over 5%, p95 latency past target, or `model_loaded`
  ever coming back false.

## 5. When to retrain
I'd retrain — and re-check the threshold — whenever any of these is true:
- It's been a quarter (regular refresh).
- A drift alert from Section 1 fires, or live performance from Section 3 slips.
- Something material changes on the business side: pricing, the catalogue, the loyalty programme, or the
  acquisition mix.
- The input schema changes (a feature added or removed).

Retraining just reruns the Part 3 pipeline on a fresh snapshot, compares the new model against the current one
on a held-out split, and only promotes it if it matches or beats what's already in production.

---

## How the retention team should (and shouldn't) use this

**Good uses:**
- **Prioritising** who to contact first when there's only so much time and budget.
- **Combining** the score with the Part 2 segments and plain human judgement — it's one input, not a verdict.
- **Triggering helpful things** — a check-in, fixing an open complaint, a relevant offer, an onboarding nudge.

**Please don't:**
- **Make service worse for "likely churners"** (slower support, less help). That actively causes the churn the
  model predicts, and it isn't fair to the customer.
- **Wire it up to automatic or customer-facing decisions** with nobody in the loop.
- **Target by demographics.** Age group and city tier feed the model, so keep actions tied to behaviour and
  keep checking that outreach isn't quietly disadvantaging any group.
- **Read too much into the number.** It's a *relative* risk score, not a calibrated probability — a 0.72 means
  "high risk", not "72% certain to leave".

And whatever the score says, **every customer still deserves a baseline level of care** — the model should
never become an excuse to write someone off.
