# Monitoring & Responsible-Use Plan — Churn Scoring API (Part 4)

This service scores a customer's 60-day churn risk. Because it informs retention spend on real customers, it
must be monitored after deployment and used within clear guardrails.

## 1. Data drift (inputs)
- **What:** distribution of each input feature vs the 2025-09-30 training snapshot.
- **How:** weekly Population Stability Index (PSI) on the top drivers (`recency_days`, `last_visit_days_ago`,
  `frequency_180d`, `monetary_180d`, `return_rate_180d`, `negative_ticket_rate_90d`); track null rates and
  category-level frequencies for the profile fields.
- **Alert:** PSI > 0.2 on any top driver, or a new/!unseen categorical level, or a >2× change in a feature's
  null rate.

## 2. Prediction distribution (outputs)
- **What:** the daily/weekly distribution of `churn_probability` and the share scored `high`/`medium`/`low`.
- **Why:** a sudden shift (e.g. average score jumps) usually signals upstream data problems before labels exist.
- **Alert:** mean predicted probability moves more than ±0.10 vs the trailing 4-week baseline, or the `high`
  bucket share changes by more than 50% relative.

## 3. Business outcomes (model quality, once labels mature)
- **What:** after each 60-day window closes, compute realised precision/recall/F1/ROC-AUC of past scores
  against actual churn; track the lift of contacted vs not-contacted within risk bands (ideally via holdout).
- **Why:** offline test metrics decay; live performance is the real signal of value.
- **Alert:** realised recall on churners drops below ~0.80, or precision falls enough to make outreach
  uneconomic, or a retention A/B holdout shows no incremental retention.

## 4. API errors & operational health
- **What:** request rate, p50/p95 latency, HTTP 4xx (esp. 422 validation failures) and 5xx rates, and the
  `/health` `model_loaded` flag.
- **Why:** a spike in 422s means a caller's schema drifted; 503/`model_loaded=false` means the artifact failed
  to load.
- **Alert:** 5xx rate > 1%, sustained 422 rate > 5%, p95 latency above target, or any `model_loaded=false`.

## 5. Retraining triggers
Retrain (and re-validate the threshold) when **any** holds:
- Scheduled cadence: at least **quarterly**.
- Drift alert fires (Section 1) or live performance degrades (Section 3).
- A material business change: pricing, catalogue, loyalty programme, or acquisition mix.
- The input schema changes (new/removed features).
Retraining repeats Part 3's pipeline on a fresh snapshot, compares against the current model on a held-out
split, and only promotes the new model if it matches or beats the incumbent.

---

## Responsible-use note (for the retention team)

**Use the score to:**
- **Prioritise** outreach — rank who to contact first when budget/time is limited.
- **Combine** with the Part 2 segments and human judgement; the score is one input, not a verdict.
- **Trigger supportive actions** — check-ins, service recovery, relevant offers, or onboarding nudges.

**Do NOT use the score to:**
- **Deny or degrade service** (e.g. slower support, withholding help) for "likely-to-churn" customers — that
  causes the churn it predicts and is unethical.
- **Make automated, irreversible, or customer-facing decisions** with no human in the loop.
- **Target by demographics.** Age group and city tier are model inputs; base actions on behaviour, and monitor
  that outreach does not systematically disadvantage any group.
- **Over-read the number.** It is a *relative risk score*, not a calibrated probability or a certainty — treat
  a 0.72 as "high risk", not "72% guaranteed to leave".

**Always keep** a low-cost baseline experience for every customer, including low scorers, so the model never
becomes a reason to abandon recoverable customers.
