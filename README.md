# Fraud Triage Service

## Overview

The Fraud Triage Service identifies potentially fraudulent payments and helps determine which transactions require human review. It cleans payment data, processes relevant features, and scores fraud risk using a Random Forest classifier at a validated decision threshold. It retrieves the matching fraud policy using TF-IDF-based retrieval and asks a scripted mock language model to generate a case note in validated JSON format. The service is built with FastAPI and includes input and output guardrails, logging, retry handling, safe defaults, and request metrics. Its purpose is to demonstrate an end-to-end fraud triage pipeline, from payment data preparation to risk scoring and policy-based case handling.

## Reproduce

Python 3.11+, run from the repository root (Windows PowerShell shown).

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py          # prints: rows written: 6090
python prepare.py                # Part A audit and checkpoints
python train.py                  # Part B tables (1-3 minutes)
python retrieval.py              # Part C retrieval table
python casenote.py               # Part C scenario table
Remove-Item service.log -ErrorAction SilentlyContinue
uvicorn app:app --port 8000      # terminal 1, wait for "Uvicorn running"
python smoke_test.py             # terminal 2, run once per server start
```

## Part A results

### Audit

| Item | Value |
|---|---:|
| Rows, columns as read | 6,090 x 14 |
| Exact duplicate rows | 90 |
| Rows after de-duplication | 6,000 |
| Missing amount_inr | 76 |
| Missing distance_from_home_km | 193 |
| amount_inr below 1 or above 1,000,000 | 5 |
| Distinct raw spellings, merchant_category / city | 24 / 24 |

### Cleaned table

5,919 rows. Categories: Grocery 1,406; Food Delivery 1,326; Fashion 976; Electronics 845; Utilities 685; Travel 681.

Payment modes: UPI 3,221; Card 1,450; NetBanking 731; Wallet 517.

amount_inr minimum 141.37, maximum 105,883.95. Missing distance_from_home_km: 191.

is_night sums to 365, high_value to 17, maximum amount_vs_avg_ratio 13.11.

The missing `distance_from_home_km` values are not filled in Part A because the median must be calculated using only the training rows. Filling missing values before splitting the dataset could allow information from validation and test rows to influence the training process, causing data leakage. The training median of 12.6 is therefore learned later and applied consistently to all splits.

### Leakage audit

| chargeback_filed | Rows | Fraud rows |
|---:|---:|---:|
| 1 | 373 | 304 |
| 0 | 5,546 | 62 |

The `chargeback_filed` column must not be used as a model input because a chargeback is filed after a payment settles, whereas fraud risk must be assessed before settlement. Since 304 of the 373 rows with a filed chargeback are fraud, this feature almost reveals the target label and causes data leakage. Similarly, `txn_id` and `customer_id` are identifiers rather than behavioural characteristics, so they are not reliable features for predicting fraud in new transactions.

**Excluded columns:** `chargeback_filed`, `txn_id`, `customer_id`, `is_fraud`.

**Model-input columns (before one-hot encoding):**

`amount_inr`, `hour_of_day`, `txn_count_last_24h`, `avg_amount_30d_inr`, `distance_from_home_km`, `is_new_device`, `account_age_days`, `amount_vs_avg_ratio`, `is_night`, `high_value`, `merchant_category`, `city`, `payment_mode`.

### Why accuracy misleads

Fraud rows: 366 of 5,919 (6.18%). A rule that predicts "not fraud" for everything has accuracy 93.82% and recall 0.

Accuracy is misleading here because the dataset is highly imbalanced, with legitimate payments making up most of the records. A model can achieve 93.82% accuracy by predicting every payment as legitimate while failing to detect any fraud. Precision, recall, and F1-score provide a more useful assessment of how effectively the model identifies fraudulent transactions.

## Part B results

### Input columns (26, pinned order)

```python
[
    'amount_inr',
    'hour_of_day',
    'txn_count_last_24h',
    'avg_amount_30d_inr',
    'distance_from_home_km',
    'is_new_device',
    'account_age_days',
    'amount_vs_avg_ratio',
    'is_night',
    'high_value',
    'merchant_category_Electronics',
    'merchant_category_Fashion',
    'merchant_category_Food Delivery',
    'merchant_category_Grocery',
    'merchant_category_Travel',
    'merchant_category_Utilities',
    'city_Bengaluru',
    'city_Chennai',
    'city_Delhi',
    'city_Kolkata',
    'city_Mumbai',
    'city_Pune',
    'payment_mode_Card',
    'payment_mode_NetBanking',
    'payment_mode_UPI',
    'payment_mode_Wallet'
]
```

### Split

| Split | Rows | Fraud rows |
|---|---:|---:|
| Train | 3,551 | 220 |
| Validation | 1,184 | 73 |
| Test | 1,184 | 73 |

Training median of `distance_from_home_km`: 12.6.

### Imbalance treatments (logistic regression, validation, threshold 0.5)

| Variant | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain | 0.625 | 0.342 | 0.442 | 1096 | 15 | 48 | 25 |
| class_weight="balanced" | 0.252 | 0.781 | 0.381 | 942 | 169 | 16 | 57 |
| Over-sampled | 0.257 | 0.781 | 0.386 | 946 | 165 | 16 | 57 |

Over-sampled training table: 6,662 rows (3,331 non-fraud + 3,331 fraud).

A fraud triage desk that must avoid missing fraudulent payments would prefer the class-weighted or over-sampled model because both achieve a recall of 0.781, compared with 0.342 for the plain model. This improvement comes at the cost of more false positives: the balanced model produces 169 false positives and the over-sampled model produces 165, compared with only 15 for the plain model. Consequently, more legitimate payments are flagged for review, and precision falls from 0.625 to approximately 0.25. The choice reflects a trade-off between catching more fraud and increasing the workload for human reviewers.

### Cross-validation and tuning (training split, 5-fold stratified, F1)

| Model | Mean F1 | Std |
|---|---:|---:|
| Logistic regression, balanced | 0.402 | 0.015 |
| Random forest, untuned | 0.453 | 0.052 |

Grid search: 12 combinations x 5 folds.

**Best configuration:** `max_depth=None`, `min_samples_leaf=5`, `n_estimators=200`, `class_weight="balanced"`, `random_state=42`.

**Best mean CV F1:** 0.465.

The tuned Random Forest was selected because its best mean cross-validation F1-score of 0.465 is higher than the balanced logistic regression score of 0.402, indicating a better balance between precision and recall on the training folds. It can capture nonlinear relationships and interactions between payment features that may be difficult for a linear model to represent. However, the forest has a higher fold-to-fold standard deviation of 0.052, compared with 0.015 for logistic regression, indicating less consistent performance across folds. Logistic regression is also easier to interpret because its coefficients show how features influence predictions, whereas a forest's decisions are distributed across multiple trees. The selected forest therefore prioritises predictive performance while accepting reduced simplicity and stability.

### Validation threshold table (tuned forest)

| t | Precision | Recall | F1 | Flagged |
|---:|---:|---:|---:|---:|
| 0.1 | 0.111 | 0.973 | 0.200 | 638 |
| 0.2 | 0.158 | 0.945 | 0.271 | 437 |
| 0.3 | 0.209 | 0.863 | 0.336 | 302 |
| 0.4 | 0.272 | 0.712 | 0.394 | 191 |
| 0.5 | 0.352 | 0.589 | 0.441 | 122 |
| 0.6 | 0.507 | 0.507 | 0.507 | 73 |
| 0.7 | 0.528 | 0.260 | 0.349 | 36 |
| 0.8 | 0.778 | 0.192 | 0.308 | 18 |
| 0.9 | 0.800 | 0.055 | 0.103 | 5 |

**Threshold selection rule:** `t_star` is the highest threshold with validation recall at least 0.65.

**Selected threshold:** `t_star = 0.4`.

### Final test result at t_star = 0.4 (used once)

| Precision | Recall | F1 | Flagged | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.281 | 0.781 | 0.413 | 203 | 965 | 146 | 16 | 57 |

The risk committee selected a threshold of 0.4 using the rule that chooses the highest validation threshold achieving a recall of at least 0.65. On the held-out test set, 203 of 1,184 payments were flagged for further attention, including 57 fraudulent payments and 146 legitimate payments. This corresponds to approximately 2.56 legitimate payments flagged for every fraud detected (146 / 57). The model missed 16 fraudulent payments, demonstrating that the system reduces but does not eliminate fraud risk. A stricter threshold would reduce the number of payments flagged, but it would also miss more fraud: at 0.4, validation recall was 0.712 with 191 flagged payments, while at 0.6, recall fell to 0.507 with only 73 flagged payments. The selected threshold therefore favours fraud detection over minimising manual review, although the false-positive rate and missed fraud cases remain important operational concerns.

### Leakage experiment (validation F1 at 0.5)

| Model | F1 |
|---|---:|
| Clean tuned forest | 0.441 |
| With chargeback_filed (27th column) | 0.797 |

The F1-score increase from 0.441 to 0.797 is caused by giving the model access to `chargeback_filed`, which contains information that becomes available only after settlement. This leaky score is not a genuine improvement because the feature would not be available when a new payment needs to be assessed, so the result cannot be trusted as evidence of real-world performance.

### Feature importance (top five)

| Feature | Importance |
|---|---:|
| amount_vs_avg_ratio | 0.2700 |
| txn_count_last_24h | 0.1779 |
| amount_inr | 0.1339 |
| account_age_days | 0.0835 |
| distance_from_home_km | 0.0709 |

The two most important features are `amount_vs_avg_ratio` and `txn_count_last_24h`. A payment far above a customer's usual spending pattern may indicate unusual activity, while a burst of transactions in a short period may indicate an attempt to drain an account or test stolen payment credentials before they are blocked.

## Part C results

### Retrieval (TF-IDF, default settings)

All 8 in-scope queries retrieved their expected policies.

| Query | Expected | Retrieved | Score |
|---|---|---|---:|
| card payment above INR 50000 from a new device | P1 | P1 | 0.5501 |
| how many UPI payments in 24 hours trigger a velocity alert | P2 | P2 | 0.5622 |
| payment at 3 AM far away from the home city | P3 | P3 | 0.4948 |
| customer reports unauthorised payment and chargeback case | P4 | P4 | 0.5689 |
| electronics merchant payment on an account younger than 30 days | P5 | P5 | 0.5641 |
| which payments are approved automatically without manual review | P6 | P6 | 0.6250 |
| temporary hold on repeated UPI payments within a day | P2 | P2 | 0.4862 |
| what happens after a customer reports an unauthorised payment | P4 | P4 | 0.4867 |
| what is the cafeteria menu today | none | P3 | 0.4251 |
| how do I reset my email password | none | P1 | 0.0000 |

**Minimum retrieval score:** `MIN_SCORE = 0.45`.

This threshold is above the highest out-of-scope score of 0.4251 and below the lowest in-scope score of 0.4862, separating the tested in-scope queries from the tested out-of-scope queries.

### Scenario run

| Scenario | Retrieved policy | Attempts | Failures | Status | Safe default | Final action | Override |
|---|---|---:|---|---|---|---|---|
| S1 (T900001) | P1 | 1 | [] | ok | no | hold_for_review | no |
| S2 (T900002) | P2 | 2 | [1] | ok | no | hold_for_review | no |
| S3 (T900003) | P3 | 2 | [2] | ok | no | hold_for_review | no |
| S4 (T900004) | P4 | 2 | [3] | ok | no | block_and_notify | no |
| S5 (T900005) | P5 | 3 | [1, 1, 1] | ok | yes | hold_for_review | no |
| S6 (T900006) | P2 | 1 | [] | ok | no | hold_for_review | yes |
| S7 (T900007) | P1 | 2 | [3] | ok | no | hold_for_review | no |
| S8 (T900008) | P4 | 1 | [] | ok | no | block_and_notify | no |
| S9 (T900009) | P3 | 3 | [1, 2] | ok | no | hold_for_review | no |
| S10 (T900010) | P2 | 2 | [3] | ok | no | hold_for_review | no |

**Totals:** 19 model calls; 3 valid on the first call (S1, S6, S8); 6 recovered by retry (S2, S3, S4, S7, S9, S10); 1 safe default (S5); 1 guardrail override (S6).

Hosted language models are probabilistic, so even when given the same prompt, they can produce different outputs or repeat the same formatting or validation error. Outputs may contain invalid JSON, missing required keys, or values outside the allowed set, and repeated attempts can still fail, as demonstrated by scenario S5. Validation checks are therefore necessary to prevent malformed outputs from reaching downstream systems. Retries allow recoverable errors to be corrected, while safe defaults and guardrail overrides ensure that unresolved failures or prohibited actions do not bypass the service's safety rules.

## Part D results

### Smoke test

Smoke test (fresh server, one run): **11 of 11 PASS**.

| Id | Category | Expected | Observed | Result |
|---|---|---:|---:|---|
| Q1 | happy | 200 | 200 | PASS |
| Q2 | happy | 200 | 200 | PASS |
| Q3 | escalation | 200 | 200 | PASS |
| Q4 | escalation | 200 | 200 | PASS |
| Q5 | edge | 400 | 400 | PASS |
| Q6 | edge | 422 | 422 | PASS |
| Q7 | escalation | 200 | 200 | PASS |
| Q8 | escalation | 200 | 200 | PASS |
| Q9 | edge | 503 | 503 | PASS |
| Q10 | happy | 200 | 200 | PASS |
| Q11 | edge | 200 | 200 | PASS |

**Per category:** happy 3/3, escalation 4/4, edge 4/4.

### GET /metrics

```json
{
  "requests_total": 10,
  "success_total": 7,
  "error_total": 3,
  "success_rate": 0.7,
  "error_rate": 0.3,
  "avg_latency_ms": 38.5316,
  "human_review_total": 5,
  "human_review_rate": 0.5
}
```

These metrics summarise the requests handled during the observed server session. The service recorded 10 requests, with 7 successes and 3 errors, an average latency of 38.5316 ms, and 5 requests requiring human review.

### Example service.log entries

```text
2026-10-22 16:58:09,390 INFO endpoint=/score status=200 latency_ms=63.1 txn_id=T800001 fraud_probability=0.0430
2026-10-22 16:58:09,490 WARNING endpoint=/score status=400 latency_ms=0.9
2026-10-22 16:58:09,642 ERROR endpoint=/case status=503 latency_ms=45.7 txn_id=T900099 fraud_probability=0.9597
```

The log entries demonstrate how the service records successful requests, input-related errors, and server-side failures. They include useful diagnostic information such as endpoint, status code, latency, transaction identifier where available, and predicted fraud probability where applicable.

## Limitations

- The dataset is synthetic, and its fraud labels are generated using a hand-built rule, so the model may learn patterns specific to that rule rather than the full complexity of real-world fraud.
- The retrieval component uses keyword-based TF-IDF similarity rather than semantic understanding, which means that unrelated queries can still retrieve a policy; for example, the out-of-scope cafeteria query scored 0.4251.
- The language model is a scripted mock, not a real hosted language model, so the scenario results do not establish how an actual model would behave in production.
- The threshold was selected using a validation set containing only 73 fraud cases, making the recall estimate sensitive to a relatively small number of cases.
- The service metrics are held in memory and reset when the server restarts, limiting long-term monitoring and historical analysis.
- The model is retrained at every application start-up, which adds unnecessary overhead and means the service does not yet load a versioned, pre-trained model artifact.
- The test results are based on a synthetic dataset and a single reported smoke-test run; performance should be validated on representative real-world data before the service is used for operational fraud decisions.