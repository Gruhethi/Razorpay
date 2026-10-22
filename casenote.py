import json
import mock_llm
from retrieval import retrieve, MIN_SCORE

SCHEMA = {
    "type": "object",
    "required": ["summary", "priority", "recommended_team", "recommended_action", "policy_id"],
    "properties": {
        "summary": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_team": {"type": "string",
                             "enum": ["card_risk", "upi_risk", "merchant_ops", "customer_care"]},
        "recommended_action": {"type": "string",
                               "enum": ["approve", "hold_for_review", "block_and_notify"]},
        "policy_id": {"type": "string", "enum": [f"P{i}" for i in range(1, 7)]},
    },
}

def build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text):
    return f"""ROLE: You are a fraud-operations assistant at a payments risk desk.

TASK: Write a short case note for the held payment below, citing the policy provided.

CONSTRAINTS:
- Use only the policy text given below; do not invent rules.
- Reply with one JSON object and nothing else (no prose, no markdown).

OUTPUT FORMAT: a JSON object with exactly these keys:
summary (string), priority (low|medium|high),
recommended_team (card_risk|upi_risk|merchant_ops|customer_care),
recommended_action (approve|hold_for_review|block_and_notify),
policy_id (P1|P2|P3|P4|P5|P6)

POLICY TEXT: {policy_text}

TXN_ID: {txn_id}
FRAUD_PROBABILITY: {fraud_probability}
AMOUNT_INR: {amount_inr}
POLICY_ID: {policy_id}
"""

def validate(text, retrieved_policy_id):
    try:
        note = json.loads(text)
    except ValueError:
        return None, 1
    if not isinstance(note, dict):
        return None, 1
    for k in SCHEMA["required"]:
        if k not in note or not isinstance(note[k], str):
            return None, 2
    for k, spec in SCHEMA["properties"].items():
        if "enum" in spec and note[k] not in spec["enum"]:
            return None, 3
    if note["policy_id"] != retrieved_policy_id:
        return None, 3
    return note, 0

def run_case(txn_id, fraud_probability, amount_inr, question, max_attempts=3):
    pid, ptext, score = retrieve(question)
    out = dict(status="ok", attempts=0, failures=[], case_note=None,
               safe_default=False, guardrail_override=False)
    if score < MIN_SCORE:
        out["status"] = "no_policy_match"
        return out
    prompt = build_prompt(txn_id, fraud_probability, amount_inr, pid, ptext)
    note = None
    for _ in range(max_attempts):
        text = mock_llm.generate(prompt)        # ConnectionError propagates on purpose
        out["attempts"] += 1
        note, gate = validate(text, pid)
        if gate == 0:
            break
        out["failures"].append(gate)
        note = None
    if note is None:
        note = {"summary": "Automated case note unavailable; manual review required.",
                "priority": "high", "recommended_team": "customer_care",
                "recommended_action": "hold_for_review", "policy_id": pid}
        out["safe_default"] = True
    if note["recommended_action"] == "approve" and fraud_probability >= 0.5:
        note["recommended_action"] = "hold_for_review"
        out["guardrail_override"] = True
    out["case_note"] = note
    return out

if __name__ == "__main__":
    mock_llm.reset()
    print("scenario | policy | attempts | failures | status | safe | action | override")
    calls = 0
    for i, s in enumerate(json.load(open("case_scenarios.json")), 1):
        r = run_case(s["txn_id"], s["fraud_probability"], s["amount_inr"], s["question"])
        calls += r["attempts"]
        pid = r["case_note"]["policy_id"] if r["case_note"] else None
        act = r["case_note"]["recommended_action"] if r["case_note"] else None
        print(f"S{i} ({s['txn_id']}) | {pid} | {r['attempts']} | {r['failures']} | {r['status']} | "
              f"{'yes' if r['safe_default'] else 'no'} | {act} | {'yes' if r['guardrail_override'] else 'no'}")
    print("total model calls:", calls)   # expect 19