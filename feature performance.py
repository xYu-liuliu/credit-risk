import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("data/feature_selection")
PRED_DIR = Path("data/prediction")
MODEL_DIR = Path("data/model")
OUT_DIR   = Path("data/analysis/feature")   
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PARQ = DATA_DIR / "train_sel.parquet"   
VAL_PARQ   = DATA_DIR / "val_sel.parquet"    
TEST_PARQ  = DATA_DIR / "test_sel.parquet"    

VAL_PRED   = PRED_DIR / "val_pred.csv"        
TEST_PRED  = PRED_DIR / "test_pred.csv"      

MODEL_FILE = MODEL_DIR / "lgbm_timecv_prauc.txt"
plt.style.use("seaborn-v0_8") 

OUT_DIR    = DATA_DIR / "feature analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP10_FILE = OUT_DIR / "shap_top10_validation.txt"

BAD_CAP    = 0.018     
PSI_BINS   = 10
EPS        = 1e-6
TOPN       = 10        
IV_BINS = 10  

def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """preprocess"""
    df = df.copy()
    X = df.drop(columns=[c for c in ["target","WEEK_NUM","case_id","y_prob"] if c in df.columns],
                errors="ignore")
    if "month_decision" in X.columns:
        X["month_decision"] = (X["month_decision"].astype("int32")
                               .astype(CategoricalDtype(categories=list(range(1,13)), ordered=False)))
    if "weekday_decision" in X.columns:
        X["weekday_decision"] = (X["weekday_decision"].astype("int32")
                                 .astype(CategoricalDtype(categories=list(range(1,8)), ordered=False)))
    return X

def split_num_cat_subset(X: pd.DataFrame, feat_list):
    """Split given feature list into numeric vs categorical"""
    cat_cols = [c for c in feat_list if c in X.columns and str(X[c].dtype) == "category"]
    num_cols = [c for c in feat_list if c in X.columns and c not in cat_cols]
    return num_cols, cat_cols

def pick_threshold_by_bad_cap(df_pred: pd.DataFrame, bad_cap=0.018):
    """
    Choose the highest threshold t such that the cumulative bad rate among
    """
    tmp = df_pred[["y_prob","target"]].dropna().sort_values("y_prob", ascending=True).reset_index(drop=True)
    cum_bads = tmp["target"].astype(int).cumsum()
    n = np.arange(1, len(tmp)+1)
    cum_bad_rate = cum_bads / n
    ok = np.where(cum_bad_rate <= bad_cap)[0]
    if len(ok) == 0: 
        return None
    k = int(ok[-1])
    return float(tmp.loc[k, "y_prob"])

#  PSI 
def psi_numeric(expected: np.ndarray, actual: np.ndarray, bins=PSI_BINS, edges=None, eps=EPS):
    """
    Numeric PSI with quantile binning defined on 'expected'
    """
    e = expected[~pd.isna(expected)]
    a = actual[~pd.isna(actual)]
    if edges is None:
        qs = np.quantile(e, np.linspace(0,1,bins+1))
        qs[0], qs[-1] = -np.inf, np.inf
    else:
        qs = edges
    e_hist, _ = np.histogram(e, bins=qs)
    a_hist, _ = np.histogram(a, bins=qs)
    e_rate = e_hist / max(1, len(e))
    a_rate = a_hist / max(1, len(a))
    contrib = (a_rate - e_rate) * np.log((a_rate + eps) / (e_rate + eps))
    return float(np.sum(contrib)), qs

def psi_categorical(expected: pd.Series, actual: pd.Series, base_cats=None, eps=EPS):
    """
    Categorical PSI aligned on 'base_cats' 
    """
    cats = pd.Index(base_cats).astype("object") if base_cats is not None \
           else pd.Index(pd.Series(expected).dropna().unique()).astype("object")
    e_cnt = pd.Series(expected).astype("object").value_counts()
    a_cnt = pd.Series(actual).astype("object").value_counts()
    e_rate = np.array([e_cnt.get(c,0) for c in cats], dtype=float); e_rate = e_rate / max(1, e_rate.sum())
    a_rate = np.array([a_cnt.get(c,0) for c in cats], dtype=float); a_rate = a_rate / max(1, a_rate.sum())
    contrib = (a_rate - e_rate) * np.log((a_rate + eps) / (e_rate + eps))
    return float(np.sum(contrib)), cats.tolist()


def shap_matrix_on(df_feat: pd.DataFrame, booster: lgb.Booster) -> np.ndarray:
    """
    Full SHAP matrix via LightGBM predict(pred_contrib=True).
    """
    contrib = booster.predict(df_feat, pred_contrib=True)
    return contrib[:, :-1]




def ks_numeric(train_vals: np.ndarray, other_vals: np.ndarray) -> float:
    """Two-sample KS for numeric features """
    x = np.asarray(train_vals)
    y = np.asarray(other_vals)
    x = x[~pd.isna(x)]; y = y[~pd.isna(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # max difference
    xs = np.sort(x); ys = np.sort(y)
    allv = np.unique(np.concatenate([xs, ys]))
    # search the clostest cdf
    x_cdf = np.searchsorted(xs, allv, side="right") / len(xs)
    y_cdf = np.searchsorted(ys, allv, side="right") / len(ys)
    return float(np.max(np.abs(x_cdf - y_cdf)))

def iv_numeric(values: np.ndarray, target: np.ndarray, edges=None, bins=IV_BINS, eps=1e-6):
    """
    calculate numeric feature IV based on train's bin edge
    """
    v = np.asarray(values); y = np.asarray(target).astype(int)
    m = ~pd.isna(v); v = v[m]; y = y[m]
    if edges is None:
        qs = np.quantile(v, np.linspace(0,1,bins+1))
        qs[0], qs[-1] = -np.inf, np.inf
    else:
        qs = edges
    b = np.digitize(v, qs[1:-1], right=True)
    # each bin
    bad = y.sum()
    good = len(y) - bad
    if good + bad == 0:
        return np.nan, qs
    iv = 0.0
    for bi in range(len(qs)-1):
        mask = (b == bi)
        nb = int(y[mask].sum())
        ng = int(mask.sum()) - nb
        bad_rate  = (nb + eps) / (bad  + eps* (len(qs)-1))
        good_rate = (ng + eps) / (good + eps* (len(qs)-1))
        woe = np.log(good_rate / bad_rate)
        iv += (good_rate - bad_rate) * woe
    return float(iv), qs

def iv_categorical(values: pd.Series, target: np.ndarray, levels=None, eps=1e-6):
    """
    calculate object feature IV based on train's bin edge
    """
    s = pd.Series(values).astype("object")
    y = np.asarray(target).astype(int)
    m = ~s.isna()
    s = s[m]; y = y[m]
    if levels is None:
        levels = list(pd.Index(s.dropna().unique()).astype("object"))
    total_bad  = int(y.sum())
    total_good = int(len(y) - total_bad)
    if total_good + total_bad == 0:
        return np.nan, levels
    iv = 0.0
    for lv in levels:
        mask = (s == lv)
        nb = int(y[mask].sum())
        ng = int(mask.sum()) - nb
        bad_rate  = (nb + eps) / (total_bad  + eps*len(levels))
        good_rate = (ng + eps) / (total_good + eps*len(levels))
        woe = np.log(good_rate / bad_rate)
        iv += (good_rate - bad_rate) * woe
    return float(iv), levels




def main():
    # ---- load data & preds ----
    train_df = pd.read_parquet(TRAIN_PARQ)
    val_df   = pd.read_parquet(VAL_PARQ)
    test_df  = pd.read_parquet(TEST_PARQ)

    val_pred  = pd.read_csv(VAL_PRED)
    test_pred = pd.read_csv(TEST_PRED)
    assert len(val_df)==len(val_pred), "val rows mismatch between parquet and preds"
    assert len(test_df) == len(test_pred), "test rows mismatch between parquet and preds"
    
    X_tr = prepare_X(train_df)
    X_va = prepare_X(val_df)
    X_te = prepare_X(test_df)

    # load 
    booster = lgb.Booster(model_file=str(MODEL_FILE))
    feat_order = booster.feature_name()
    X_tr = X_tr.reindex(columns=feat_order)
    X_va = X_va.reindex(columns=feat_order)
    X_te = X_te.reindex(columns=feat_order)
    for X in (X_tr, X_va, X_te):
        assert list(X.columns) == feat_order, "Feature order mismatch"

    # calculate SHAP Top-10 on validation 
    if TOP10_FILE.exists():
        top10 = [ln.strip() for ln in TOP10_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        shap_val = shap_matrix_on(X_va, booster)                 # full validation SHAP
        mean_abs = np.abs(shap_val).mean(axis=0)
        shap_imp = (pd.DataFrame({"feature": feat_order, "mean_abs_shap": mean_abs})
                    .sort_values("mean_abs_shap", ascending=False)
                    .reset_index(drop=True))
        shap_imp.to_csv(OUT_DIR / "shap_importance_validation.csv", index=False)
        top10 = shap_imp["feature"].head(TOPN).tolist()
        TOP10_FILE.write_text("\n".join(top10), encoding="utf-8")
    print("Top-10 (validation):", top10)

    # Feature PSI ONLY Top-10 
    num_cols, cat_cols = split_num_cat_subset(X_va, top10)
    rows, num_edges_cache, cat_levels_cache = [], {}, {}
    for f in top10:
        if f in num_cols:
            psi_tv, edges = psi_numeric(X_tr[f].to_numpy(), X_va[f].to_numpy(), bins=PSI_BINS, edges=None, eps=EPS)
            psi_tt, _     = psi_numeric(X_tr[f].to_numpy(), X_te[f].to_numpy(), bins=PSI_BINS, edges=edges, eps=EPS)
            num_edges_cache[f] = edges
            rows.append({"feature": f, 
                         "psi_train_to_val": psi_tv, "psi_train_to_test": psi_tt})
        else:
            base_cats = pd.Index(X_tr[f].astype("object").dropna().unique()).tolist()
            psi_tv, cats = psi_categorical(X_tr[f], X_va[f], base_cats=base_cats, eps=EPS)
            psi_tt, _    = psi_categorical(X_tr[f], X_te[f], base_cats=base_cats, eps=EPS)
            cat_levels_cache[f] = cats
            rows.append({"feature": f, 
                         "psi_train_to_val": psi_tv, "psi_train_to_test": psi_tt})
    psi_top10 = pd.DataFrame(rows)
    
    
    iv_rows, ks_rows = [], []
    y_tr = train_df["target"].values
    y_va = val_df["target"].values
    y_te = test_df["target"].values

    for f in top10:
        if f in num_cols:
            # KS for numeric features (Train vs Val/Test)
            ks_tv = ks_numeric(X_tr[f].to_numpy(), X_va[f].to_numpy())
            ks_tt = ks_numeric(X_tr[f].to_numpy(), X_te[f].to_numpy())
            ks_rows.append({
                "feature": f, 
                "ks_train_vs_val": ks_tv,
                "ks_train_vs_test": ks_tt
            })
            # IV for Train/Val/Test using Train-defined bin edges
            iv_tr, edges = iv_numeric(X_tr[f].to_numpy(), y_tr, edges=None)
            iv_va, _     = iv_numeric(X_va[f].to_numpy(), y_va, edges=edges)
            iv_te, _     = iv_numeric(X_te[f].to_numpy(), y_te, edges=edges)
        else:
            # Categorical: KS not defined
            ks_rows.append({
                "feature": f, 
                "ks_train_vs_val": np.nan,
                "ks_train_vs_test": np.nan
            })
            levels = list(pd.Index(X_tr[f].astype("object").dropna().unique()).astype("object"))
            iv_tr, levels = iv_categorical(X_tr[f], y_tr, levels=levels)
            iv_va, _      = iv_categorical(X_va[f], y_va, levels=levels)
            iv_te, _      = iv_categorical(X_te[f], y_te, levels=levels)

        iv_rows.append({
            "feature": f,
            "iv_train": iv_tr,
            "iv_val": iv_va,
            "iv_test": iv_te,
            "iv_val_minus_train": (iv_va - iv_tr) if pd.notna(iv_va) and pd.notna(iv_tr) else np.nan,
            "iv_test_minus_train": (iv_te - iv_tr) if pd.notna(iv_te) and pd.notna(iv_tr) else np.nan,
        })

    df_iv = pd.DataFrame(iv_rows).sort_values("feature").reset_index(drop=True)
    df_ks = pd.DataFrame(ks_rows).sort_values("feature").reset_index(drop=True)
    df_iv.to_csv(OUT_DIR / "feature_iv_train_val_test_top10.csv", index=False)
    df_ks.to_csv(OUT_DIR / "feature_ks_train_vs_val_test_top10.csv", index=False)
    print("✅ saved:", OUT_DIR / "feature_iv_train_val_test_top10.csv")
    print("✅ saved:", OUT_DIR / "feature_ks_train_vs_val_test_top10.csv")
   
    
    stability = (psi_top10
                 .merge(df_ks, on=["feature"], how="left")
                 .merge(df_iv, on=["feature"], how="left"))
    stability = stability[[
        "feature",
        "psi_train_to_val","psi_train_to_test",
        "ks_train_vs_val","ks_train_vs_test",
        "iv_train","iv_val","iv_test",
        "iv_val_minus_train","iv_test_minus_train"
    ]]
    stability.to_csv(OUT_DIR / "feature_stability_top10.csv", index=False)
    print("✅ saved:", OUT_DIR / "feature_stability_top10.csv")
    
    # Approved-subset weekly Feature PSI (Val→Test), ONLY Top-10 
    # pick threshold 
    thr = pick_threshold_by_bad_cap(val_pred[["y_prob","target"]], bad_cap=BAD_CAP)
    if thr is None:
        raise RuntimeError("No feasible threshold under BAD_CAP; try a higher cap.")

    # attach predictions to dfs for masking & grouping
    val_df = val_df.copy(); test_df = test_df.copy()
    for c in ["y_prob","WEEK_NUM","target"]:
        val_df[c]  = val_pred[c].values
        test_df[c] = test_pred[c].values

    # approved subsets
    val_app  = val_df[val_df["y_prob"] <= thr]
    test_app = test_df[test_df["y_prob"] <= thr]

    # feature frames for approved subsets
    val_appX  = prepare_X(val_app)
    test_appX = prepare_X(test_app)

    # baseline for weekly PSI
    weeks = sorted(test_app["WEEK_NUM"].unique())
    weekly_rows = []

    # precompute baseline spec for each feature 
    baseline_spec = {}
    for f in top10:
        if f in num_cols:
            base_vals = val_appX[f].to_numpy()
            if base_vals.size == 0:
                baseline_spec[f] = ("num", None)
            else:
                _, edges = psi_numeric(base_vals, base_vals, bins=PSI_BINS, edges=None, eps=EPS)
                baseline_spec[f] = ("num", edges)
        else:
            base_levels = pd.Index(val_appX[f].astype("object").dropna().unique()).tolist()
            baseline_spec[f] = ("cat", base_levels)

    # weekly PSI: VAL-approved (expected) vs TEST-approved-in-week (actual)
    for w in weeks:
        sub = test_app.loc[test_app["WEEK_NUM"] == w]
        if sub.empty:
            for f in top10:
                weekly_rows.append({"WEEK_NUM": int(w), "feature": f, "psi_val_to_test_approved": np.nan})
            continue
        subX = prepare_X(sub)
        for f in top10:
            kind, spec = baseline_spec[f]
            if kind == "num":
                exp_vals = val_appX[f].to_numpy()
                act_vals = subX[f].to_numpy()
                if (exp_vals.size == 0) or (act_vals.size == 0) or (spec is None):
                    psi_w = np.nan
                else:
                    psi_w, _ = psi_numeric(exp_vals, act_vals, bins=PSI_BINS, edges=spec, eps=EPS)
            else:
                exp_vals = val_appX[f]
                act_vals = subX[f]
                psi_w, _ = psi_categorical(exp_vals, act_vals, base_cats=spec, eps=EPS)
            weekly_rows.append({"WEEK_NUM": int(w), "feature": f, "psi_val_to_test_approved": float(psi_w)})

    weekly_psi_top10 = pd.DataFrame(weekly_rows).sort_values(["feature","WEEK_NUM"]).reset_index(drop=True)
    weekly_psi_top10.to_csv(OUT_DIR / "weekly_feature_psi_val_to_test_approved_top10.csv", index=False)
    print("✅ saved:", OUT_DIR / "weekly_feature_psi_val_to_test_approved_top10.csv")

    # summary 
    weekly_summary = (weekly_psi_top10
                      .groupby("feature", as_index=False)["psi_val_to_test_approved"]
                      .agg(median_week_psi="median", max_week_psi="max"))
    weekly_summary.to_csv(OUT_DIR / "weekly_feature_psi_val_to_test_approved_summary_top10.csv", index=False)
    print("✅ saved:", OUT_DIR / "weekly_feature_psi_val_to_test_approved_summary_top10.csv")

if __name__ == "__main__":
    main()




