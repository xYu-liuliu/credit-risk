import pandas as pd
import numpy as np
from pathlib import Path


CORR_THR = 0.97                      
EXCLUDE  = ["WEEK_NUM", "case_id", "target"]    
DIR      = Path("data/processed")

PREF = {
    "mean":   1,
    "median": 1,
    "max":    2,
    "min":    2,
    "std":    3,
    "var":    3,
    "p95":    3,
    "last":   4,
}

def pref_rank(colname: str, default: int = 99) -> int:
    """Return priority rank """
    agg = colname.split("_")[0]          
    return PREF.get(agg, default)

def choose_to_drop(col_a: str, col_b: str) -> str:
    """Decide which column to DROP among two highly correlated ones."""
    return col_b if pref_rank(col_a) < pref_rank(col_b) else col_a


def corr_filter_pref(df: pd.DataFrame,
                     thresh: float = 0.97,
                     exclude: list[str] | None = None):
    """
    solve duplicates.
    """
    if exclude is None:
        exclude = []

    use_df = df.drop(columns=exclude, errors="ignore")
    corr   = use_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop: set[str] = set()
    for col in upper.columns:
        if col in to_drop or col in exclude:
            continue
        partners = upper.index[upper[col] > thresh].tolist()
        for other in partners:
            if other in to_drop or other in exclude:
                continue
            drop = choose_to_drop(col, other)
            to_drop.add(drop)

    kept = [c for c in df.columns if c not in to_drop]
    return kept, sorted(to_drop)

# load 
train = pd.read_parquet(DIR / "df_train_encoded.parquet")
val   = pd.read_parquet(DIR / "df_val_encoded.parquet")
test  = pd.read_parquet(DIR / "df_test_encoded.parquet")

train["target"] = train["target"].astype("Int64")  
val["target"] = val["target"].astype("Int64")
test["target"] = test["target"].astype("Int64")



kept_cols, drop_cols = corr_filter_pref(train,
                                        thresh=CORR_THR,
                                        exclude=EXCLUDE)

print(f"🔗 |ρ| > {CORR_THR} → {len(drop_cols)} columns dropped:")
for c in drop_cols:
    print("   -", c)

# apply ans save
train_cln = train[kept_cols]
val_cln   = val[kept_cols]
test_cln  = test[kept_cols]

out = Path(r"E:/Home Credit Processed Feature")
out.mkdir(exist_ok=True)

train_cln["target"] = train_cln["target"].astype("Int64")  
val_cln["target"] = val_cln["target"].astype("Int64")
test_cln["target"] = test_cln["target"].astype("Int64")

train_cln.to_parquet(out / "train_corr.parquet", compression="zstd")
val_cln.to_parquet(out / "val_corr.parquet",   compression="zstd")
test_cln .to_parquet(out / "test_corr.parquet",  compression="zstd")

print(f"✅ Correlation filtering complete. Cleaned files saved to: {out}")


