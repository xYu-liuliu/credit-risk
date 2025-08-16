import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
from pandas.api.types import CategoricalDtype
import re


DATA_DIR   = Path(r"E:/Home Credit Processed Feature")
MODEL_FILE = DATA_DIR / "lgbm_timecv_prauc.txt"
VAL_PARQ   = DATA_DIR / "val_sel.parquet"     
TEST_PARQ  = DATA_DIR / "test_sel.parquet"    
OUT_DIR    = DATA_DIR / "feature analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Features to analyze
FEATURES = [
    "mean_refreshdate_3813885D",
    "max_amount_4527230A",
    "pmtnum_254L",
]

N_BINS = 20          
MIN_BIN = 10         
CAT_MIN_BIN = 8     

plt.style.use("seaborn-v0_8") 


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-feature cols and restore categorical dtypes."""
    drop = ["target", "WEEK_NUM", "case_id", "y_prob"]
    X = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore").copy()

    # restore categorical dtypes 
    if "month_decision" in X.columns:
        X["month_decision"] = (X["month_decision"].astype("int32")
                               .astype(CategoricalDtype(categories=list(range(1,13)), ordered=False)))
    if "weekday_decision" in X.columns:
        X["weekday_decision"] = (X["weekday_decision"].astype("int32")
                                 .astype(CategoricalDtype(categories=list(range(1,8)), ordered=False)))

    # cast numerics to float32 
    for c in X.columns:
        if str(X[c].dtype) != "category":
            X[c] = X[c].astype("float32")
    return X

def shap_matrix(booster: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Full SHAP matrix via LightGBM pred_contrib"""
    contrib = booster.predict(X, pred_contrib=True)
    return contrib[:, :-1]

def sanitize(name: str) -> str:
    """Safe filename component."""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name)

def binned_dep_overlay(x_val, s_val, x_test, s_test, n_bins=N_BINS,
                       min_bin=MIN_BIN, cat_min=CAT_MIN_BIN, force_zero_bin=True):
    """
    Build bins using VAL+TEST, then compute mean SHAP per bin for each split.
    """
    # categorical
    if str(getattr(x_val, "dtype", "")) == "category" or str(getattr(x_test, "dtype", "")) == "category":
        # Convert to codes
        val_codes = pd.Series(x_val).astype("category")
        test_codes= pd.Series(x_test).astype("category")
        cats = sorted(set(val_codes.cat.categories.tolist()) | set(test_codes.cat.categories.tolist()))
        # map to integers
        map_idx = {c:i for i,c in enumerate(cats)}
        v_code = val_codes.map(map_idx).to_numpy()
        t_code = test_codes.map(map_idx).to_numpy()

        dfv = pd.DataFrame({"x": v_code, "y": s_val})
        dft = pd.DataFrame({"x": t_code, "y": s_test})
        gv = dfv.groupby("x").agg(n=("y","size"), xmean=("x","mean"), ymean=("y","mean"))
        gt = dft.groupby("x").agg(n=("y","size"), xmean=("x","mean"), ymean=("y","mean"))
        # keep bins with enough samples in each split 
        idx_keep = sorted(set(gv.index[gv["n"]>=cat_min]).union(set(gt.index[gt["n"]>=cat_min])))
        xg = np.array(idx_keep, dtype=float)
        yv = gv.reindex(idx_keep)["ymean"].to_numpy()
        yt = gt.reindex(idx_keep)["ymean"].to_numpy()
        return xg, yv, yt, "category_codes"

    # numeric
    x_all = np.concatenate([np.asarray(x_val), np.asarray(x_test)], axis=0)
    qs = np.quantile(x_all, np.linspace(0, 1, n_bins+1))
    qs = np.unique(qs)
    if force_zero_bin:
        qs = np.unique(np.concatenate([qs, np.array([0.0])]))  
    if len(qs) <= 2:
        xc = np.array([np.nanmean(x_all)])
        return xc, np.array([np.nanmean(s_val)]), np.array([np.nanmean(s_test)]), "numeric"

    # bin each split with shared edges
    bv = np.digitize(x_val, qs[1:-1], right=True)
    bt = np.digitize(x_test, qs[1:-1], right=True)
    dfv = pd.DataFrame({"bin": bv, "x": x_val, "y": s_val})
    dft = pd.DataFrame({"bin": bt, "x": x_test, "y": s_test})

    gv = dfv.groupby("bin").agg(n=("y","size"), xmean=("x","mean"), ymean=("y","mean"))
    gt = dft.groupby("bin").agg(n=("y","size"), xmean=("x","mean"), ymean=("y","mean"))

    idx_keep = [b for b in sorted(set(gv.index) | set(gt.index))
                if (gv.loc[b,"n"] if b in gv.index else 0) >= min_bin
                and (gt.loc[b,"n"] if b in gt.index else 0) >= min_bin]
    if not idx_keep:
        idx_keep = [b for b in sorted(set(gv.index) | set(gt.index))
                    if ((gv.loc[b,"n"] if b in gv.index else 0) >= min_bin) or
                       ((gt.loc[b,"n"] if b in gt.index else 0) >= min_bin)]

    gvs = gv.reindex(idx_keep).sort_values("xmean")
    gts = gt.reindex(idx_keep).sort_values("xmean")
    xg  = np.nanmean([gvs["xmean"].to_numpy(), gts["xmean"].to_numpy()], axis=0)
    return xg, gvs["ymean"].to_numpy(), gts["ymean"].to_numpy(), "numeric"



def main():
    # load data & model
    booster = lgb.Booster(model_file=str(MODEL_FILE))
    val_df  = pd.read_parquet(VAL_PARQ)
    test_df = pd.read_parquet(TEST_PARQ)

    Xv = prepare_X(val_df)
    Xt = prepare_X(test_df)

    # align feature order with model
    feat_order = booster.feature_name()
    Xv = Xv.reindex(columns=feat_order)
    Xt = Xt.reindex(columns=feat_order)
    assert list(Xv.columns) == feat_order == list(Xt.columns), "Feature order mismatch with model."

    # full SHAP matrices 
    shap_v = shap_matrix(booster, Xv)
    shap_t = shap_matrix(booster, Xt)

    # weeks for weekly mean|SHAP|
    weeks_v = np.sort(val_df["WEEK_NUM"].unique()) if "WEEK_NUM" in val_df.columns else []
    weeks_t = np.sort(test_df["WEEK_NUM"].unique()) if "WEEK_NUM" in test_df.columns else []

    for f in FEATURES:
        if f not in Xv.columns:
            print(f"[warn] feature not found: {f}")
            continue

        j = Xv.columns.get_loc(f)
        xv = Xv[f].values
        xt = Xt[f].values
        sv = shap_v[:, j]
        st = shap_t[:, j]
        
        for name, s in [("VAL", Xv[f]), ("TEST", Xt[f])]:
            s = pd.Series(s)
            print(name, 
                  "n=", s.size, 
                  "na%=", s.isna().mean(), 
                  "unique=", s.nunique(dropna=True),
                  "min/50%/max=", s.min(), s.median(), s.max())
            print(s.value_counts(dropna=False).head(10), "\n")
            
            
        # binned dependence for Val vs Test 
        xgrid, yv, yt, kind = binned_dep_overlay(xv, sv, xt, st)
        plt.figure(figsize=(8,5))
        plt.plot(xgrid, yv, label="Validation", marker="o", markersize=4, linewidth=1.0)
        plt.plot(xgrid, yt, label="Test", marker="o", markersize=4, linewidth=1.0)
        plt.axhline(0.0, ls="--", lw=0.8)
        plt.title(f"{f} — Binned dependence (mean SHAP per bin)")
        plt.xlabel("Feature value" + (" (category codes)" if kind=="category_codes" else ""))
        plt.ylabel("Mean SHAP per bin")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{sanitize(f)}_dep_val_vs_test.png", dpi=160)

        # Weekly mean|SHAP|
        if len(weeks_v) and len(weeks_t):
            # val weekly
            val_week_vals = []
            for w in weeks_v:
                idx = (val_df["WEEK_NUM"].values == w)
                val_week_vals.append(np.abs(sv[idx]).mean() if idx.any() else np.nan)
            # test weekly
            test_week_vals = []
            for w in weeks_t:
                idx = (test_df["WEEK_NUM"].values == w)
                test_week_vals.append(np.abs(st[idx]).mean() if idx.any() else np.nan)

            plt.figure(figsize=(9,5))
            plt.plot(weeks_v, val_week_vals, marker="o", markersize=4, linewidth=1.0, label="Validation")
            plt.plot(weeks_t, test_week_vals, marker="o", markersize=4, linewidth=1.0, label="Test")
            plt.title(f"{f} — Weekly mean |SHAP|")
            plt.xlabel("WEEK_NUM")
            plt.ylabel("mean |SHAP|")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"{sanitize(f)}_weekly_meanabs_val_vs_test.png", dpi=160)
        else:
            print(f"[info] WEEK_NUM missing; skip weekly plot for {f}")

        print(f"saved: {OUT_DIR / (sanitize(f)+'_dep_val_vs_test.png')}")
        print(f"saved: {OUT_DIR / (sanitize(f)+'_weekly_meanabs_val_vs_test.png')}")

if __name__ == "__main__":
    main()




