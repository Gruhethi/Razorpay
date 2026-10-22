import logging, time
import pandas as pd
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prepare import add_features
from train import build_final_model
from casenote import run_case

logging.basicConfig(filename="service.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("triage")

MODEL = build_final_model()
COLS, MED, THR = MODEL["columns"], MODEL["median"], float(MODEL["threshold"])
CATS = {"Grocery", "Electronics", "Travel", "Food Delivery", "Fashion", "Utilities"}
M = dict(total=0, ok=0, err=0, latency=0.0, review=0)

app = FastAPI()

class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: Optional[float] = None
    is_new_device: int
    account_age_days: int

class CaseIn(Txn):
    question: str

@app.middleware("http")
async def observe(request: Request, call_next):
    t0 = time.perf_counter()
    request.state.info = {}
    response = await call_next(request)
    if request.method == "POST":
        ms = (time.perf_counter() - t0) * 1000
        info, st = request.state.info, response.status_code
        M["total"] += 1; M["latency"] += ms
        M["ok" if st < 400 else "err"] += 1
        M["review"] += 1 if info.get("human_review") else 0
        if st >= 500: lvl = logging.ERROR
        elif st >= 400 or info.get("safe_default") or info.get("override"): lvl = logging.WARNING
        else: lvl = logging.INFO
        msg = f"endpoint={request.url.path} status={st} latency_ms={ms:.1f}"
        if "txn_id" in info:
            msg += f" txn_id={info['txn_id']} fraud_probability={info['prob']:.4f}"
        log.log(lvl, msg)
    return response

def reject(msg, field):
    return JSONResponse(status_code=400, content={"error": msg, "field": field})

def check(t):
    if not 1 <= t.amount_inr <= 1_000_000: return reject("amount_inr must be 1 to 1,000,000", "amount_inr")
    if not 0 <= t.hour_of_day <= 23: return reject("hour_of_day must be 0 to 23", "hour_of_day")
    if t.txn_count_last_24h < 0: return reject("txn_count_last_24h must be >= 0", "txn_count_last_24h")
    if t.avg_amount_30d_inr <= 0: return reject("avg_amount_30d_inr must be > 0", "avg_amount_30d_inr")
    if t.merchant_category.strip().title() not in CATS: return reject("unknown merchant_category", "merchant_category")
    return None

def probability(t):
    d = t.model_dump()
    d["merchant_category"] = d["merchant_category"].strip().title()
    d["city"] = d["city"].strip().title()
    df = add_features(pd.DataFrame([d]).drop(columns=["txn_id"]))
    X = pd.get_dummies(df, columns=["merchant_category", "city", "payment_mode"], dtype=int)
    X = X.reindex(columns=COLS, fill_value=0)
    X["distance_from_home_km"] = pd.to_numeric(X["distance_from_home_km"]).fillna(MED)
    return float(MODEL["model"].predict_proba(X.astype(float))[0, 1])

def base_decision(t, p, info):
    flagged = p >= THR
    decision = "hold_for_review" if flagged else "approve"
    if not flagged and t.amount_inr >= 100_000 and t.is_new_device == 1:
        decision, info["override"] = "hold_for_review", True
    return flagged, decision

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(t: Txn, request: Request):
    info = request.state.info
    if (err := check(t)): return err
    p = probability(t)
    info.update(txn_id=t.txn_id, prob=p)
    flagged, decision = base_decision(t, p, info)
    info["human_review"] = decision == "hold_for_review"
    return dict(txn_id=t.txn_id, fraud_probability=round(p, 4), flagged=flagged,
                decision=decision, human_review=info["human_review"])

@app.post("/case")
def case(c: CaseIn, request: Request):
    info = request.state.info
    if (err := check(c)): return err
    p = probability(c)
    info.update(txn_id=c.txn_id, prob=p)
    flagged, decision = base_decision(c, p, info)
    note, attempts, safe, override = None, 0, False, False
    if flagged:
        try:
            r = run_case(c.txn_id, p, c.amount_inr, c.question)
        except ConnectionError:
            return JSONResponse(status_code=503, content={
                "error": "case-note service temporarily unavailable, try again later"})
        if r["status"] == "no_policy_match":
            decision = "hold_for_review"
        else:
            note, attempts, safe, override = r["case_note"], r["attempts"], r["safe_default"], r["guardrail_override"]
            if note["recommended_action"] == "approve":      # flagged is never auto-approved
                note["recommended_action"], override = "hold_for_review", True
            decision = note["recommended_action"]
    human = decision != "approve" or safe
    info.update(human_review=human, safe_default=safe, override=info.get("override") or override)
    return dict(txn_id=c.txn_id, fraud_probability=round(p, 4), flagged=flagged, decision=decision,
                human_review=human, case_note=note, attempts=attempts,
                safe_default=safe, guardrail_override=override)

@app.get("/metrics")
def metrics():
    n = M["total"]
    r = lambda x: round(x, 4)
    return dict(requests_total=n, success_total=M["ok"], error_total=M["err"],
                success_rate=r(M["ok"] / n) if n else 0.0, error_rate=r(M["err"] / n) if n else 0.0,
                avg_latency_ms=r(M["latency"] / n) if n else 0.0,
                human_review_total=M["review"], human_review_rate=r(M["review"] / n) if n else 0.0)