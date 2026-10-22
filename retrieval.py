import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICIES = json.load(open("fraud_policies.json"))
_vec = TfidfVectorizer()
_matrix = _vec.fit_transform([p["text"] for p in POLICIES])
MIN_SCORE = 0.45   # above 0.4251 (best out-of-scope), at most 0.4862 (lowest in-scope)

def retrieve(query):
    sims = cosine_similarity(_vec.transform([query]), _matrix)[0]
    i = int(sims.argmax())
    return POLICIES[i]["policy_id"], POLICIES[i]["text"], float(sims[i])

if __name__ == "__main__":
    hits = 0
    for q in json.load(open("retrieval_queries.json")):
        pid, _, score = retrieve(q["query"])
        hits += q["expected_policy"] is not None and pid == q["expected_policy"]
        print(f"{q['query'][:55]:55} exp={q['expected_policy']} got={pid} {score:.4f}")
    print("in-scope hits:", hits, "/ 8 | MIN_SCORE", MIN_SCORE)