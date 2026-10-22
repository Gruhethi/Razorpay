import time, requests, json
B = "http://127.0.0.1:8000"
BEN = dict(merchant_category="Grocery", city="Pune", payment_mode="UPI", amount_inr=1500,
           hour_of_day=13, txn_count_last_24h=0, avg_amount_30d_inr=1600,
           distance_from_home_km=3.0, is_new_device=0, account_age_days=900)
RISKY = dict(merchant_category="Electronics", city="Mumbai", payment_mode="Card", amount_inr=45000,
             hour_of_day=2, txn_count_last_24h=9, avg_amount_30d_inr=3000,
             distance_from_home_km=60.0, is_new_device=1, account_age_days=40)
BIG = {**BEN, "merchant_category": "Utilities", "payment_mode": "NetBanking", "amount_inr": 150000,
       "avg_amount_30d_inr": 250000, "distance_from_home_km": 0.0, "is_new_device": 1,
       "account_age_days": 2200}
def P(base, **kw): return {**base, **kw}
nohour = {k: v for k, v in BEN.items() if k != "hour_of_day"}
Q10 = "card payment above INR 50000 from a new device"

# id, category, method, path, body, expected status, checker(json)
TESTS = [
 ("Q1", "happy", "GET", "/health", None, 200, lambda j: j == {"status": "ok"}),
 ("Q2", "happy", "POST", "/score", P(BEN, txn_id="T800001"), 200,
  lambda j: j["flagged"] is False and j["decision"] == "approve" and j["human_review"] is False),
 ("Q3", "escalation", "POST", "/score", P(RISKY, txn_id="T800002"), 200,
  lambda j: j["flagged"] and j["decision"] == "hold_for_review" and j["human_review"]),
 ("Q4", "escalation", "POST", "/score", P(BIG, txn_id="T800003"), 200,
  lambda j: j["flagged"] is False and j["decision"] == "hold_for_review" and j["human_review"]),
 ("Q5", "edge", "POST", "/score", P(BEN, txn_id="T800004", amount_inr=-5), 400,
  lambda j: "error" in j and j["field"] == "amount_inr"),
 ("Q6", "edge", "POST", "/score", P(nohour, txn_id="T800005"), 422, lambda j: True),
 ("Q7", "escalation", "POST", "/case",
  P(RISKY, txn_id="T900002", question="more than 6 UPI payments in 24 hours velocity alert"), 200,
  lambda j: j["flagged"] and j["attempts"] == 2 and not j["safe_default"]
  and j["case_note"]["policy_id"] == "P2" and j["human_review"]),
 ("Q8", "escalation", "POST", "/case",
  P(RISKY, txn_id="T900005", question="electronics merchant payment on an account younger than 30 days"), 200,
  lambda j: j["attempts"] == 3 and j["safe_default"] and j["decision"] == "hold_for_review" and j["human_review"]),
 ("Q9", "edge", "POST", "/case", P(RISKY, txn_id="T900099", question=Q10), 503, lambda j: "error" in j),
 ("Q10", "happy", "POST", "/case", P(BEN, txn_id="T800001", question=Q10), 200,
  lambda j: j["case_note"] is None and j["attempts"] == 0 and j["decision"] == "approve" and not j["human_review"]),
 ("Q11", "edge", "POST", "/case",
  P(RISKY, txn_id="T800006", question="what is the cafeteria menu today"), 200,
  lambda j: j["case_note"] is None and j["attempts"] == 0 and j["decision"] == "hold_for_review" and j["human_review"]),
]

for _ in range(120):
    try:
        if requests.get(B + "/health", timeout=2).status_code == 200: break
    except requests.RequestException: time.sleep(1)

tally = {}
for qid, cat, m, path, body, exp, chk in TESTS:
    r = requests.request(m, B + path, json=body)
    try: ok = r.status_code == exp and chk(r.json())
    except Exception: ok = False
    tally.setdefault(cat, [0, 0]); tally[cat][1] += 1; tally[cat][0] += bool(ok)
    print(f"{qid:4} {cat:11} expected={exp} observed={r.status_code} {'PASS' if ok else 'FAIL'}")
print({k: f"{a}/{b}" for k, (a, b) in tally.items()})
print(json.dumps(requests.get(B + "/metrics").json(), indent=2))