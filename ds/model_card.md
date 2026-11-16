# Model card: card portfolio risk

Two models, one section each. Both are reproduced from `etl/build_data.py` and evaluated in
[`notebooks/02_pd_scorecard`](notebooks/02_pd_scorecard.ipynb) and [`notebooks/03_fraud_model`](notebooks/03_fraud_model.ipynb).
Owner: Mohamed Ayman (portfolio project). Status: demonstration models on public data, not approved for lending or
fraud decisions.

---

## 1. Behavioural PD score

**Intended use.** Rank existing cardholders by the probability of missing next month's payment, to decide credit-line
reviews (decrease or freeze) and early collections contact. Not for new applicants, pricing or regulatory capital.

**Data and period.** UCI "Default of Credit Card Clients" (Yeh & Lien, 2009; CC BY 4.0): 30,000 Taiwanese cardholders,
six monthly statements April-September 2005, NT$. **Target:** default on the October 2005 payment (22.1% of accounts).

**Features (22, behavioural only).** Credit limit; repayment status for each of six months; utilisation (balance / limit,
capped at 2) for each month; five payment ratios (payment / previous bill, capped at 5); worst status and months past
due over six months; utilisation trend; average payment ratio. Sex, age, marital status and education are excluded.

**Training and validation.** The ETL's 70/30 split stratified on default (21,000 / 9,000), random seed 42. The data is a
single snapshot (one scoring date, one outcome month), so no out-of-time hold-out exists. Candidates are compared on
the same five stratified folds of the training set, then once on the hold-out with bootstrap intervals and paired tests.

**Models.**

| Model | Role | Hold-out AUC (95% CI) | Gini | KS |
|---|---|---|---|---|
| HistGradientBoosting (ETL parameters, 72 iterations after early stopping) | Champion (published) | 0.776 (0.764-0.788) | 0.552 | 0.421 |
| LightGBM with monotone constraints, Optuna-tuned (20 trials) | Challenger | 0.778 (0.765-0.789) | 0.556 | 0.424 |
| WoE scorecard, 6 characteristics, points at 600 / 50:1 / PDO 20 | Explainable alternative | 0.746 (0.732-0.758) | 0.492 | 0.398 |
| Logistic regression (ETL) | Baseline | 0.740 (0.726-0.753) | 0.480 | 0.391 |
| September status only | Naive rule | 0.700 (0.687-0.715) | 0.401 | 0.360 |

Paired bootstrap: challenger - champion +0.002 (CI 0.000 to +0.004); unconstrained - monotone LightGBM +0.0016 (CI
+0.0003 to +0.0028), the cost of monotonicity; champion - scorecard +0.030 (CI 0.023 to 0.037). Published numbers
reproduced: test AUC 0.7762, train AUC 0.8095, logistic 0.7402 (published 0.7404), KS 0.4214, Gini 0.5524.

**Calibration.** Mean predicted PD 21.95% vs 22.12% observed; worst decile off by 1.7 percentage points. Isotonic
recalibration changes the Brier score by 0.0002, so none is applied.

**Decision.** Score = 600 + 20/ln 2 x ln(odds / 50), clipped to 300-850; hold-out scores run 445-580. Accounts below 525
get a line review: 30.5% of accounts, 63.0% of October defaults, 54.4% of expected loss (PD x exposure x LGD), a 45.7%
bad rate against 11.8% for accounts kept. Bands: below 500 decrease and call (14% of accounts, 62% bad rate); 500-524
analyst review (16%, 31%); 525-574 maintain (67%, 12%); 575+ eligible for an increase (2.7%, 2.9%).

**Segment performance and fairness (cutoff 525, hold-out).**

| Group | Accounts | Default rate | Kept rate | Adverse-impact ratio | AUC | Predicted / observed PD |
|---|---|---|---|---|---|---|
| Female | 5,437 | 21.0% | 70.4% | 1.00 | 0.783 | 1.03 |
| Male | 3,563 | 23.9% | 68.1% | 0.97 | 0.766 | 0.94 |
| Under 25 | 802 | 26.7% | 58.1% | 0.81 | 0.775 | 1.04 |
| 25-34 | 3,856 | 19.7% | 70.8% | 0.99 | 0.775 | 1.06 |
| 35-44 | 2,707 | 22.1% | 71.5% | 1.00 | 0.775 | 0.96 |
| 45-54 | 1,302 | 25.0% | 69.0% | 0.96 | 0.775 | 0.91 |
| 55+ | 333 | 28.5% | 67.0% | 0.94 | 0.786 | 0.84 |

No group falls below the four-fifths rule. Under-25s are closest (0.81), mostly because their default rate is higher;
PD runs 4% high for them. PD under-states risk for the over-55s (0.84; small group). The behavioural inputs predict sex
with AUC 0.61 and age with R² 0.04, so proxy effects are weak but not zero.

**Explainability.** SHAP (TreeExplainer, 3,000 hold-out accounts): worst status in six months 30% of mean |SHAP|,
September status 15%, credit limit 12%, September utilisation 8%. Permutation importance agrees (Spearman rho 0.89).
Risk jumps at two months late and is flat after that; the limit matters below about NT$150K. Reason codes from SHAP and
from scorecard points lost agree on the median declined account (two months late now, two months late as worst status,
NT$30,000 limit).

**Stability.** Score PSI training vs hold-out 0.002. Month-on-month input PSI below 0.1 for utilisation and payment
ratio; status 0.63 from August to September, a coding artefact ("one month late" appears only in September).

**Limitations.** 2005 Taiwan, during a card-debt crisis (22% default), so levels do not transfer. One outcome month, not a
12-month default definition. No out-of-time validation possible. Exposure at default (balance or half the line) and LGD
(one assumption) are simplifications. Credit limit partly encodes the issuer's earlier risk view.

**Monitoring and retraining.** Monthly: predicted vs observed default rate by score band (recalibrate if the ratio leaves
0.8-1.25), AUC/Gini on the latest closed month (review if Gini drops by more than 0.05), PSI per input and for the score
(investigate above 0.10, act above 0.25 after ruling out coding changes like the September status code), and the
adverse-impact ratio by age band (review if any falls below 0.85). Refit at least annually, or on any trigger, on a
multi-cohort window. Validate the line-decrease policy with a random 10% control group below the cutoff before rollout.

---

## 2. Transaction fraud score

**Intended use.** Rank card transactions at authorisation so that a fixed analyst team reviews the riskiest 0.1-0.2%.
Not for automatic declines without review, and not for customer-facing explanations.

**Data and period.** ULB / Worldline "Credit Card Fraud Detection" (OpenML 1597, DbCL v1.0): 284,807 European card
transactions over two days in September 2013, 492 frauds (0.17%). V1-V28 are principal components of confidential
features; Time and Amount are original. **Target:** the fraud label on the transaction.

**Features (31).** V1-V28, amount, log amount, hour of day (assumes the extract starts near midnight; the overnight volume
trough supports it).

**Training and validation.** Forward in time, as in the ETL: train on the first 31.4 hours (159,847 transactions, 356
frauds), test on the next 16.6 hours (124,960 transactions, 136 frauds). Tuning (Optuna, 12 trials, average precision)
and model comparison use two expanding time-ordered folds inside the training window (78 and 75 validation frauds).

**Models and metrics (test window).**

| Model | Role | PR AUC (95% CI) | ROC AUC | Recall / precision, top 0.1% |
|---|---|---|---|---|
| HistGradientBoosting, fraud weighted 20x (ETL parameters) | Champion (published) | 0.742 (0.652-0.825) | 0.979 | 76.5% / 83.9% |
| LightGBM, unweighted, tuned | Challenger | 0.790 (0.716-0.854) | 0.985 | 76.5% / 83.9% |
| Logistic regression | Baseline | 0.638 (0.550-0.723) | 0.972 | 60.3% / 66.1% |
| Best single component (V14) | Naive rule | 0.582 (0.505-0.681) | 0.936 | 55.9% / 61.3% |

Paired bootstrap: challenger - champion PR AUC +0.048 (CI +0.009 to +0.090); recall at 0.1% +0.000 (CI -0.022 to +0.059).
Published numbers reproduced exactly: PR AUC 0.7416, ROC AUC 0.9785, recall / precision 76.5% / 83.9% at 0.1%, 80.9% / 44.2%
at 0.2%, 84.6% / 18.4% at 0.5%, 86.0% / 9.4% at 1%. Class weighting does not improve ranking (unweighted HistGB 0.763 vs
weighted 0.742, difference not significant).

**Calibration.** The weighted champion expects 436 frauds in a window with 136. Dividing the odds by 20 overshoots (87
expected). The unweighted challenger expects 146 with no correction. For a top-k queue this does not matter; for loss
forecasts or a queue ranked by p x amount, use the unweighted model or recalibrate on a recent labelled window.

**Decision.** Review the top 0.1-0.2% of transactions: 7-15 alerts an hour at this volume. Under the stated assumptions
(EUR 5 per alert; loss avoided = amount + EUR 20 per fraud caught) net value peaks at 0.15% reviewed (EUR 14,644 over
16.6 hours vs EUR 14,028 at 0.1%), the optimum stays within 0.12-0.30% for alert costs of EUR 2-20 and handling costs of
EUR 0-50, and value turns negative beyond 3% reviewed.

**Segment performance.** No customer, merchant or channel fields exist, so no group analysis is possible. Fraud is up to
10x more frequent at night; with 136 test frauds, recall by hour cannot be estimated reliably.

**Explainability.** SHAP on all 136 test frauds plus 2,864 genuine transactions: V14 23% of mean |SHAP|, V4 16%, the top
four components 51%; amount and hour about 2% each. Permutation importance agrees only on V14 (rank correlation 0.44,
noisy with 136 frauds). The components have no business meaning, so SHAP serves monitoring and alert clustering, not
reason codes.

**Drift.** Component V3 flips sign at midnight (PSI 1.4 from day 1 to day 2) and the score's hourly PSI sits at 0.2-0.4
in the test window, yet a model trained on day 1 only scores PR AUC 0.762 (vs 0.742). The shift sits in components with
little fraud signal.

**Limitations.** Two days of 2013 data. The provider fitted the PCA on both days, a mild look-ahead that makes the test
slightly optimistic. No card, merchant or device identifiers, so no velocity features. Labels are taken as final. Cost
assumptions are illustrative.

**Monitoring and retraining.** Daily: PSI per input and for the score against the last training window, like-for-like
clock hours (investigate above 0.10; act above 0.25 on a top-10 SHAP feature or the score only if queue precision also
falls); precision of the review queue at the budget (retrain if below 50% for three days). Weekly: recall on confirmed
fraud including later chargebacks (retrain if below 70%); expected vs observed fraud count (re-estimate the correction
if the ratio leaves 0.5-2). Retrain daily on a rolling window. Shadow-score the challenger for two weeks and promote it
if its PR AUC lead holds with a paired interval above zero. Send a random 5% of alerts to a no-review holdout to measure
the loss actually prevented.
