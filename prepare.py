import pandas as pd

EXCLUDE = ["txn_id", "customer_id", "chargeback_filed", "is_fraud"]

def clean_transactions(raw_df):
    df = raw_df.drop_duplicates().copy()                       # (a)
    for c in ["merchant_category", "city"]:                    # (b)
        df[c] = df[c].str.strip().str.title()
    df = df[df["amount_inr"].notna()]                          # (c)
    df = df[(df["amount_inr"] >= 1) & (df["amount_inr"] <= 1_000_000)]  # (d)
    return df.reset_index(drop=True)                           # (e)

def add_features(df):
    out = df.copy()
    out["amount_vs_avg_ratio"] = out["amount_inr"] / out["avg_amount_30d_inr"]
    out["is_night"] = (out["hour_of_day"] <= 5).astype(int)
    out["high_value"] = (out["amount_inr"] > 50_000).astype(int)
    return out

if __name__ == "__main__":
    raw = pd.read_csv("transactions_raw.csv")
    print("rows, cols:", raw.shape)
    print("exact duplicates:", raw.duplicated().sum())
    dd = raw.drop_duplicates()
    print("after dedup rows:", len(dd))
    print("missing amount:", dd["amount_inr"].isna().sum(),
          "| missing distance:", dd["distance_from_home_km"].isna().sum())
    print("out-of-range amounts:", ((dd["amount_inr"] < 1) | (dd["amount_inr"] > 1_000_000)).sum())
    print("raw spellings: category", raw["merchant_category"].nunique(), "city", raw["city"].nunique())

    clean = clean_transactions(raw)
    f = add_features(clean)
    print("\nclean rows:", len(clean))
    print(clean["merchant_category"].value_counts().to_dict())
    print(clean["payment_mode"].value_counts().to_dict())
    print("amount min/max:", clean["amount_inr"].min(), clean["amount_inr"].max())
    print("missing distance:", clean["distance_from_home_km"].isna().sum())
    print("is_night sum:", f["is_night"].sum(), "| high_value sum:", f["high_value"].sum(),
          "| max ratio:", round(f["amount_vs_avg_ratio"].max(), 2))
    print("\nchargeback leakage table:")
    print(clean.groupby("chargeback_filed")["is_fraud"].agg(rows="size", fraud="sum"))
    n, fr = len(clean), int(clean["is_fraud"].sum())
    print(f"\nfraud rows {fr}/{n} = {fr/n:.2%}; 'never fraud' accuracy {1-fr/n:.2%}, recall 0")
    # one-row and input-unchanged checks
    one = raw.head(1).copy(); before = raw.copy()
    clean_transactions(raw); add_features(clean_transactions(one) if len(clean_transactions(one)) else one)
    assert raw.equals(before)