
import pandas as pd
import numpy as np
from pathlib import Path


IV_THR   = 0.005                     
EXCLUDE  = ["WEEK_NUM", "case_id"]   
DIR      = Path(r"E:/Home Credit Processed Feature")  


def iv_qcut(
        x: pd.Series,
        y: pd.Series,
        n_bins: int = 10,
        min_bin: int = 30,
        nan_label="MISSING",
        eps: float = 1e-6
    ) -> float:
    """
    Compute Information Value (IV) using quantile / equal-frequency binning.
    """

    if x.nunique(dropna=False) <= n_bins:            
        bins = x.fillna(nan_label).astype(str)
    else:                                            
        x_rank = x.fillna(x.median()).rank(method="first")
        bins   = pd.qcut(
            x_rank, n_bins, duplicates="drop", labels=False
        ).astype("object")
        bins[x.isna()] = nan_label

    # Merge 
    if min_bin and bins.value_counts().min() < min_bin and bins.nunique() > 1:
        counts = bins.value_counts().sort_index()
        for b in counts[counts < min_bin].index:
            # merge into the previous bin (or the first bin if b == 0)
            target_bin = b - 1 if isinstance(b, int) and b > 0 else list(counts.index)[0]
            bins.replace(b, target_bin, inplace=True)

    
    ct   = pd.crosstab(bins, y)
    bad  = ct[1]
    good = ct[0]

    bad_rate  = bad  / max(bad.sum(),  eps)
    good_rate = good / max(good.sum(), eps)

    woe = np.log((bad_rate + eps) / (good_rate + eps))
    iv  = ((bad_rate - good_rate) * woe).sum()

    return iv

train = pd.read_parquet(DIR / "train_corr.parquet")
val   = pd.read_parquet(DIR / "val_corr.parquet")
test  = pd.read_parquet(DIR / "test_corr.parquet")

y_train = train.pop("target")
y_val   = val.pop("target")
y_test  = test.pop("target")


drop_iv   = []     
iv_records = []     

for col in train.columns:
    if col in EXCLUDE:               
        continue

    uniq_cnt = train[col].nunique(dropna=False)

   
    if uniq_cnt <= 2:
        iv_val = iv_qcut(train[col], y_train,
                         n_bins=2,     
                         min_bin=1)     
        iv_records.append((col, iv_val))
        continue                        


    iv_val = iv_qcut(train[col], y_train,
                     n_bins=10,
                     min_bin=30)
    iv_records.append((col, iv_val))
    if iv_val < IV_THR:
        drop_iv.append(col)

# Save 
iv_df = (pd.DataFrame(iv_records, columns=["feature", "IV"])
         .sort_values("IV", ascending=False))
iv_df.to_excel(DIR / "iv_summary.xlsx", index=False)


# Drop low-IV 

train.drop(columns=drop_iv, inplace=True, errors="ignore")
val.drop  (columns=drop_iv, inplace=True, errors="ignore")
test.drop (columns=drop_iv, inplace=True, errors="ignore")


# Save filtered datasets

out = Path(r"E:/Home Credit Processed Feature")

train.assign(target=y_train).to_parquet(out / "train_iv.parquet", compression="zstd")
val.assign  (target=y_val)  .to_parquet(out / "val_iv.parquet",   compression="zstd")
test.assign (target=y_test) .to_parquet(out / "test_iv.parquet",  compression="zstd")


print(f"🗑  IV < {IV_THR} — dropped {len(drop_iv)} columns (excluded {EXCLUDE})")
print(drop_iv)
print(f"✅ IV filtering complete. Files saved to: {out}")
