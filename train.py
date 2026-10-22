import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from prepare import clean_transactions, add_features, EXCLUDE

CAT = ["merchant_category", "city", "payment_mode"]
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
GRID = {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]}

def make_rf(**kw):
    return RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                  random_state=42, n_jobs=-1, **kw)

def make_lr(**kw):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, **kw))

def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), TN=tn, FP=fp, FN=fn, TP=tp)

def get_splits():
    raw = pd.read_csv("transactions_raw.csv")
    df = add_features(clean_transactions(raw))
    y = df["is_fraud"]
    X = pd.get_dummies(df.drop(columns=EXCLUDE), columns=CAT, dtype=int)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=42)
    med = float(Xtr["distance_from_home_km"].median())
    Xtr, Xva, Xte = [d.assign(distance_from_home_km=d["distance_from_home_km"].fillna(med))
                     for d in (Xtr, Xva, Xte)]
    return dict(df=df, X=X, Xtr=Xtr, Xva=Xva, Xte=Xte, ytr=ytr, yva=yva, yte=yte, median=med)

def tune(Xtr, ytr):
    gs = GridSearchCV(make_rf(), GRID, cv=SKF, scoring="f1")
    gs.fit(Xtr, ytr)
    return gs

def pick_threshold(model, Xva, yva):
    proba = model.predict_proba(Xva)[:, 1]
    rows, t_star = [], 0.1
    for k in range(1, 10):
        t = round(k / 10, 1)
        m = metrics(yva, (proba >= t).astype(int))
        m.update(t=t, flagged=int((proba >= t).sum()))
        rows.append(m)
        if m["recall"] >= 0.65:
            t_star = t          # grid ascends, so the last qualifying t is the highest
    return t_star, rows

def build_final_model():
    s = get_splits()
    gs = tune(s["Xtr"], s["ytr"])
    t_star, _ = pick_threshold(gs.best_estimator_, s["Xva"], s["yva"])
    return {"model": gs.best_estimator_, "columns": list(s["X"].columns),
            "median": s["median"], "threshold": t_star}

if __name__ == "__main__":
    s = get_splits()
    Xtr, Xva, Xte, ytr, yva, yte = (s[k] for k in ["Xtr", "Xva", "Xte", "ytr", "yva", "yte"])
    print("columns:", list(s["X"].columns), len(s["X"].columns))
    print("sizes:", len(Xtr), len(Xva), len(Xte), "| fraud:", ytr.sum(), yva.sum(), yte.sum(),
          "| train median:", round(s["median"], 1))

    # imbalance treatments
    plain = make_lr().fit(Xtr, ytr)
    bal = make_lr(class_weight="balanced").fit(Xtr, ytr)
    fr = Xtr[ytr == 1].sample(n=int((ytr == 0).sum()), replace=True, random_state=42)
    Xos = pd.concat([Xtr[ytr == 0], fr]); yos = pd.Series([0]*int((ytr == 0).sum()) + [1]*len(fr))
    over = make_lr().fit(Xos, yos)
    print("oversampled rows:", len(Xos))
    for name, m in [("plain", plain), ("balanced", bal), ("oversampled", over)]:
        print(name, {k: round(float(v), 3) for k, v in metrics(yva, m.predict(Xva)).items()})

    # CV and tuning (training split only)
    for name, est in [("LR balanced", make_lr(class_weight="balanced")), ("RF untuned", make_rf())]:
        sc = cross_val_score(est, Xtr, ytr, cv=SKF, scoring="f1")
        print(name, "CV F1 mean %.3f sd %.3f" % (sc.mean(), sc.std()))
    gs = tune(Xtr, ytr)
    print("best:", gs.best_params_, "CV F1 %.3f" % gs.best_score_, "| combos:", len(gs.cv_results_["params"]))
    best = gs.best_estimator_

    t_star, rows = pick_threshold(best, Xva, yva)
    print(pd.DataFrame(rows)[["t", "precision", "recall", "f1", "flagged"]].round(3).to_string(index=False))
    print("t_star:", t_star)

    # one test evaluation
    proba_te = best.predict_proba(Xte)[:, 1]
    flagged = (proba_te >= t_star).astype(int)
    print("TEST", {k: round(float(v), 3) for k, v in metrics(yte, flagged).items()}, "flagged", flagged.sum())

    # leakage experiment
    cb = s["df"]["chargeback_filed"]
    leaky = make_rf(**gs.best_params_).fit(Xtr.assign(chargeback_filed=cb.loc[Xtr.index]), ytr)
    f_leaky = f1_score(yva, leaky.predict(Xva.assign(chargeback_filed=cb.loc[Xva.index])))
    f_clean = f1_score(yva, best.predict(Xva))
    print("val F1 @0.5 clean %.3f leaky %.3f" % (f_clean, f_leaky))

    imp = pd.Series(best.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    print(imp.head(5).round(4))