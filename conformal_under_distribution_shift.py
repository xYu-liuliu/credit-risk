from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

# =========================
# Config
# =========================
DATA_DIR = Path(r"E:/Home Credit Processed Feature")

VAL_X_PATH  = DATA_DIR / "val_sel.parquet"
TEST_X_PATH = DATA_DIR / "test_sel.parquet"

CAL_DIR   = DATA_DIR / "model analysis" / "calibration_compare"
VAL_PRED  = CAL_DIR / "val_pred_with_cal_platt.csv"
TEST_PRED = CAL_DIR / "test_pred_with_cal_platt.csv"

OUT_DIR = DATA_DIR / "model analysis" / "rolling_conformal_w4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# columns in pred files
COL_WEEK = "WEEK_NUM"
COL_Y    = "target"
COL_P    = "y_prob_cal"     # calibrated prob; if you want raw, change here

# rolling window
W = 8
alpha = 0.05
EPS = 1e-12
CLIP_Q = 0.99

# risk-score binning for grouped Mondrian
N_BINS_X = 5          # start with 5; try 8 if stable and you want finer
MIN_BIN_N = 200       # min total samples in bin (otherwise fallback to global)
MIN_BIN_CLASS_N = 20  # min samples per class within bin (otherwise fallback per class)

# domain clf settings
DOMAIN_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    metric="auc",
    verbosity=-1,
    seed=42,
)

MIN_CLASS_N = 20   # if calibration window has too few y=1, Mondrian q1 becomes unstable

# =========================
# Helpers
# =========================
def find_id_col(df1: pd.DataFrame, df2: pd.DataFrame):
    candidates = ["SK_ID_CURR", "case_id", "id", "ID", "application_id"]
    for c in candidates:
        if c in df1.columns and c in df2.columns:
            return c
    return None

def weighted_quantile(values, quantile, sample_weight):
    v = np.asarray(values, float)
    w = np.asarray(sample_weight, float)
    if len(v) == 0:
        return np.nan
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted)
    cutoff = quantile * cum_w[-1]
    idx = np.searchsorted(cum_w, cutoff, side="left")
    idx = min(idx, len(v_sorted) - 1)
    return float(v_sorted[idx])

def score_symmetric(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    return np.where(y == 1, 1 - p, p)

def score_logloss(y, p):
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.where(y == 1, -np.log(p), -np.log(1 - p))

def build_sets_symmetric(p, q):
    p = np.asarray(p, float)
    inc0 = (p <= q)          # score0 = p
    inc1 = ((1 - p) <= q)    # score1 = 1-p
    sz = inc0.astype(int) + inc1.astype(int)
    return inc0, inc1, sz

def build_sets_logloss(p, q0, q1):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    inc1 = (-np.log(p) <= q1)
    inc0 = (-np.log(1 - p) <= q0)
    sz = inc0.astype(int) + inc1.astype(int)
    return inc0, inc1, sz

def eval_sets(y_true, inc0, inc1, sz):
    y_true = np.asarray(y_true).astype(int)
    true_in_set = np.where(y_true == 1, inc1, inc0)

    out = {
        "coverage_overall": float(true_in_set.mean()),
        "empty_rate(review)": float((sz == 0).mean()),
        "singleton_rate": float((sz == 1).mean()),
        "ambiguity_rate{0,1}": float((sz == 2).mean()),
        "avg_set_size": float(sz.mean()),
        "n": int(len(y_true)),
    }
    for c in [0, 1]:
        idx = (y_true == c)
        out[f"n_y={c}"] = int(idx.sum())
        out[f"coverage_y={c}"] = float(true_in_set[idx].mean()) if idx.any() else np.nan
    return out

def get_feature_cols(df: pd.DataFrame, id_col: str | None):
    drop = {COL_WEEK, COL_Y, COL_P}
    if id_col is not None:
        drop.add(id_col)
    cols = [c for c in df.columns if c not in drop]
    return cols

def domain_weights_dr(X_cal: pd.DataFrame, X_tgt: pd.DataFrame):
    """Train domain clf: cal=0, tgt=1, return weights on cal points (clipped)."""
    X_cal = X_cal.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_tgt = X_tgt.replace([np.inf, -np.inf], np.nan).fillna(0)

    y_cal = np.zeros(len(X_cal), dtype=int)
    y_tgt = np.ones(len(X_tgt), dtype=int)
    X = pd.concat([X_cal, X_tgt], axis=0, ignore_index=True)
    y = np.concatenate([y_cal, y_tgt], axis=0)

    # simple shuffle split
    rng = np.random.RandomState(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(0.8 * len(X))
    tr, va = idx[:split], idx[split:]

    dtr = lgb.Dataset(X.iloc[tr], y[tr])
    dva = lgb.Dataset(X.iloc[va], y[va])

    model = lgb.train(
        DOMAIN_PARAMS,
        dtr,
        num_boost_round=2000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    g = model.predict(X_cal)
    g = np.clip(g, 1e-6, 1 - 1e-6)

    nC, nT = len(X_cal), len(X_tgt)
    w = (g / (1 - g)) * (nC / nT)

    cap = float(np.quantile(w, CLIP_Q))
    w = np.minimum(w, cap)

    info = dict(
        w_mean=float(np.mean(w)),
        w_p95=float(np.quantile(w, 0.95)),
        w_p99=float(np.quantile(w, 0.99)),
        cap=cap
    )
    return w, info

# ===== Grouped Mondrian by risk score bins (B(X)=bin(p)) =====
def make_bins_by_quantiles(p_cal: np.ndarray, n_bins: int):
    """
    Use calibration probabilities to define bin edges by quantiles.
    Returns edges of length n_bins+1, with edges[0]=-inf, edges[-1]=inf.
    """
    p_cal = np.asarray(p_cal, float)
    if n_bins <= 1:
        return np.array([-np.inf, np.inf], dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    interior = np.quantile(p_cal, qs[1:-1])
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    edges = np.maximum.accumulate(edges)  # handle ties
    return edges

def assign_bins(p: np.ndarray, edges: np.ndarray):
    """
    Assign each p into bin id in {0,...,K-1} using edges length K+1.
    """
    p = np.asarray(p, float)
    if len(edges) == 2:
        return np.zeros(len(p), dtype=int)
    b = np.digitize(p, edges[1:-1], right=True)  # 0..K-1
    return b.astype(int)

def compute_mondrian_thresholds_by_bin(
    y_cal: np.ndarray,
    s_ll_cal: np.ndarray,
    bins_cal: np.ndarray,
    alpha: float,
    weights: np.ndarray | None,
    q0_global: float,
    q1_global: float,
    n_bins: int,
    min_bin_n: int,
    min_bin_class_n: int,
):
    """
    For each bin b, compute q0[b], q1[b] on calibration points in that bin.
    Fallback logic:
      - If bin total < min_bin_n: use global thresholds.
      - Else if class count < min_bin_class_n: use global for that class.
    Supports optional weights (for weighted quantile).
    """
    y_cal = np.asarray(y_cal).astype(int)
    s_ll_cal = np.asarray(s_ll_cal, float)
    bins_cal = np.asarray(bins_cal).astype(int)
    if weights is not None:
        weights = np.asarray(weights, float)

    q0 = np.full(n_bins, q0_global, dtype=float)
    q1 = np.full(n_bins, q1_global, dtype=float)

    fallback_bin = np.zeros(n_bins, dtype=int)
    fallback0 = np.zeros(n_bins, dtype=int)
    fallback1 = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        idx_b = (bins_cal == b)
        nb = int(idx_b.sum())
        if nb < min_bin_n:
            fallback_bin[b] = 1
            continue

        yb = y_cal[idx_b]
        sb = s_ll_cal[idx_b]
        wb = weights[idx_b] if weights is not None else None

        # y=0
        idx0 = (yb == 0)
        n0 = int(idx0.sum())
        if n0 >= min_bin_class_n:
            if wb is None:
                q0[b] = float(np.quantile(sb[idx0], 1 - alpha))
            else:
                q0[b] = float(weighted_quantile(sb[idx0], 1 - alpha, wb[idx0]))
        else:
            fallback0[b] = 1

        # y=1
        idx1 = (yb == 1)
        n1 = int(idx1.sum())
        if n1 >= min_bin_class_n:
            if wb is None:
                q1[b] = float(np.quantile(sb[idx1], 1 - alpha))
            else:
                q1[b] = float(weighted_quantile(sb[idx1], 1 - alpha, wb[idx1]))
        else:
            fallback1[b] = 1

    info = {
        "fallback_bin_rate": float(fallback_bin.mean()),
        "fallback0_rate": float(fallback0.mean()),
        "fallback1_rate": float(fallback1.mean()),
    }
    return q0, q1, info

def build_sets_logloss_by_bin(p_tgt: np.ndarray, bins_tgt: np.ndarray, q0: np.ndarray, q1: np.ndarray):
    p = np.clip(np.asarray(p_tgt, float), EPS, 1 - EPS)
    b = np.asarray(bins_tgt).astype(int)
    q0_i = q0[b]
    q1_i = q1[b]
    inc1 = (-np.log(p) <= q1_i)
    inc0 = (-np.log(1 - p) <= q0_i)
    sz = inc0.astype(int) + inc1.astype(int)
    return inc0, inc1, sz

# =========================
# Build master table: (VAL+TEST) with features + (week,y,p)
# =========================
def load_and_merge():
    Xv = pd.read_parquet(VAL_X_PATH)
    Xt = pd.read_parquet(TEST_X_PATH)
    Pv = pd.read_csv(VAL_PRED)
    Pt = pd.read_csv(TEST_PRED)

    for df, tag in [(Pv, "VAL_PRED"), (Pt, "TEST_PRED")]:
        need = {COL_WEEK, COL_Y, COL_P}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"[{tag}] missing columns: {miss}")

    id_col = find_id_col(Xv, Pv)

    def _row_align_join(P: pd.DataFrame, X: pd.DataFrame, tag: str):
        P_small = P[[COL_WEEK, COL_Y, COL_P]].reset_index(drop=True)
        X2 = X.reset_index(drop=True)

        if len(P_small) != len(X2):
            raise ValueError(
                f"[{tag}] row-align failed: len(P)={len(P_small)} != len(X)={len(X2)}. "
                f"Pred CSV and feature parquet are not in the same row order. "
                f"Fix by exporting id column into pred files and merge on id."
            )

        dup = set(P_small.columns).intersection(set(X2.columns))
        if dup:
            X2 = X2.drop(columns=list(dup), errors="ignore")

        return pd.concat([P_small, X2], axis=1)

    if id_col is not None:
        val = Pv[[id_col, COL_WEEK, COL_Y, COL_P]].merge(Xv, on=id_col, how="inner")
        test = Pt[[id_col, COL_WEEK, COL_Y, COL_P]].merge(Xt, on=id_col, how="inner")
    else:
        val = _row_align_join(Pv, Xv, "VAL")
        test = _row_align_join(Pt, Xt, "TEST")
        id_col = None

    all_df = pd.concat([val, test], axis=0, ignore_index=True)

    all_df[COL_WEEK] = pd.to_numeric(all_df[COL_WEEK], errors="coerce").astype(int)
    all_df[COL_Y] = pd.to_numeric(all_df[COL_Y], errors="coerce").astype(int)
    all_df[COL_P] = pd.to_numeric(all_df[COL_P], errors="coerce").astype(float)

    all_df = all_df.dropna(subset=[COL_WEEK, COL_Y, COL_P])
    return all_df, id_col

# =========================
# Rolling evaluation
# =========================
def run():
    df, id_col = load_and_merge()
    feat_cols = get_feature_cols(df, id_col)

    test_weeks = sorted(df.loc[df[COL_WEEK] >= 75, COL_WEEK].unique().tolist())

    # Global fallback for Mondrian thresholds (use VAL only)
    val_df = df[(df[COL_WEEK] >= 70) & (df[COL_WEEK] <= 74)].copy()
    yv = val_df[COL_Y].to_numpy()
    pv = val_df[COL_P].to_numpy()
    sv = score_logloss(yv, pv)
    s0g = sv[yv == 0]
    s1g = sv[yv == 1]
    q0_global = float(np.quantile(s0g, 1 - alpha)) if len(s0g) else np.inf
    q1_global = float(np.quantile(s1g, 1 - alpha)) if len(s1g) else np.inf

    per_week_rows = []

    for wk in test_weeks:
        cal_weeks = list(range(wk - W, wk))
        cal = df[df[COL_WEEK].isin(cal_weeks)].copy()
        tgt = df[df[COL_WEEK] == wk].copy()

        if len(cal) == 0 or len(tgt) == 0:
            continue

        y_cal = cal[COL_Y].to_numpy()
        p_cal = cal[COL_P].to_numpy()
        y_tgt = tgt[COL_Y].to_numpy()
        p_tgt = tgt[COL_P].to_numpy()

        # weights for this week (cal -> tgt)
        w_cal, w_info = domain_weights_dr(cal[feat_cols], tgt[feat_cols])

        # -------------------------
        # (1) Unconditional + Symmetric (Unweighted)
        # -------------------------
        s = score_symmetric(y_cal, p_cal)
        q = float(np.quantile(s, 1 - alpha))
        inc0, inc1, sz = build_sets_symmetric(p_tgt, q)
        m = eval_sets(y_tgt, inc0, inc1, sz)
        per_week_rows.append({"week": wk, "method": "Uncond+Sym(Unw)", "alpha": alpha, "q": q} | m)

        # -------------------------
        # (2) Unconditional + Symmetric (Weighted DR)
        # -------------------------
        q_w = weighted_quantile(s, 1 - alpha, w_cal)
        inc0, inc1, sz = build_sets_symmetric(p_tgt, q_w)
        m = eval_sets(y_tgt, inc0, inc1, sz)
        per_week_rows.append({
            "week": wk, "method": "Uncond+Sym(W-DR)", "alpha": alpha, "q": q_w,
            "w_mean": w_info["w_mean"], "w_p99": w_info["w_p99"], "w_cap": w_info["cap"]
        } | m)

        # -------------------------
        # (3) Mondrian + LogLoss (Unweighted)
        # -------------------------
        s_ll = score_logloss(y_cal, p_cal)
        s0 = s_ll[y_cal == 0]
        s1 = s_ll[y_cal == 1]
        q0 = float(np.quantile(s0, 1 - alpha)) if len(s0) >= MIN_CLASS_N else q0_global
        q1 = float(np.quantile(s1, 1 - alpha)) if len(s1) >= MIN_CLASS_N else q1_global

        inc0, inc1, sz = build_sets_logloss(p_tgt, q0, q1)
        m = eval_sets(y_tgt, inc0, inc1, sz)
        per_week_rows.append({
            "week": wk, "method": "Mond+LogLoss(Unw)", "alpha": alpha, "q0": q0, "q1": q1,
            "fallback_q0": int(len(s0) < MIN_CLASS_N), "fallback_q1": int(len(s1) < MIN_CLASS_N)
        } | m)

        # -------------------------
        # (4) Mondrian + LogLoss (Weighted DR)
        # -------------------------
        w0 = w_cal[y_cal == 0]
        w1 = w_cal[y_cal == 1]
        q0w = weighted_quantile(s0, 1 - alpha, w0) if len(s0) >= MIN_CLASS_N else q0_global
        q1w = weighted_quantile(s1, 1 - alpha, w1) if len(s1) >= MIN_CLASS_N else q1_global

        inc0, inc1, sz = build_sets_logloss(p_tgt, q0w, q1w)
        m = eval_sets(y_tgt, inc0, inc1, sz)
        per_week_rows.append({
            "week": wk, "method": "Mond+LogLoss(W-DR)", "alpha": alpha, "q0": q0w, "q1": q1w,
            "fallback_q0": int(len(s0) < MIN_CLASS_N), "fallback_q1": int(len(s1) < MIN_CLASS_N),
            "w_mean": w_info["w_mean"], "w_p99": w_info["w_p99"], "w_cap": w_info["cap"]
        } | m)

        # -------------------------
        # (5) Grouped Mondrian by risk-score bins + LogLoss (Unweighted)
        # -------------------------
        edges = make_bins_by_quantiles(p_cal, N_BINS_X)
        bins_cal = assign_bins(p_cal, edges)
        bins_tgt = assign_bins(p_tgt, edges)

        q0_bin, q1_bin, bin_info_unw = compute_mondrian_thresholds_by_bin(
            y_cal=y_cal,
            s_ll_cal=s_ll,
            bins_cal=bins_cal,
            alpha=alpha,
            weights=None,
            q0_global=q0_global,
            q1_global=q1_global,
            n_bins=N_BINS_X,
            min_bin_n=MIN_BIN_N,
            min_bin_class_n=MIN_BIN_CLASS_N,
        )

        inc0, inc1, sz = build_sets_logloss_by_bin(p_tgt, bins_tgt, q0_bin, q1_bin)
        m = eval_sets(y_tgt, inc0, inc1, sz)
        per_week_rows.append({
            "week": wk, "method": f"MondBin{N_BINS_X}+LogLoss(Unw)", "alpha": alpha,
            "fallback_bin_rate": bin_info_unw["fallback_bin_rate"],
            "fallback0_rate": bin_info_unw["fallback0_rate"],
            "fallback1_rate": bin_info_unw["fallback1_rate"],
        } | m)

        # -------------------------
        # (6) Grouped Mondrian by risk-score bins + LogLoss (Weighted DR)
        # -------------------------
        q0w_bin, q1w_bin, bin_info_w = compute_mondrian_thresholds_by_bin(
            y_cal=y_cal,
            s_ll_cal=s_ll,
            bins_cal=bins_cal,
            alpha=alpha,
            weights=w_cal,  # weighted quantiles inside each bin
            q0_global=q0_global,
            q1_global=q1_global,
            n_bins=N_BINS_X,
            min_bin_n=MIN_BIN_N,
            min_bin_class_n=MIN_BIN_CLASS_N,
        )

        inc0, inc1, sz = build_sets_logloss_by_bin(p_tgt, bins_tgt, q0w_bin, q1w_bin)
        m = eval_sets(y_tgt, inc0, inc1, sz)
        per_week_rows.append({
            "week": wk, "method": f"MondBin{N_BINS_X}+LogLoss(W-DR)", "alpha": alpha,
            "fallback_bin_rate": bin_info_w["fallback_bin_rate"],
            "fallback0_rate": bin_info_w["fallback0_rate"],
            "fallback1_rate": bin_info_w["fallback1_rate"],
            "w_mean": w_info["w_mean"], "w_p99": w_info["w_p99"], "w_cap": w_info["cap"]
        } | m)

    per_week = pd.DataFrame(per_week_rows)
    per_week_csv = OUT_DIR / f"per_week_W{W}_alpha{alpha:.2f}_bins{N_BINS_X}.csv"
    per_week.to_csv(per_week_csv, index=False)

    # =========================
    # Summary: aggregate over weeks (weighted by n)
    # =========================
    def agg(g: pd.DataFrame):
        w = g["n"].to_numpy()

        def wavg(col):
            x = g[col].to_numpy()
            return float(np.sum(w * x) / np.sum(w))

        out = {
            "weeks": int(g["week"].nunique()),
            "n_total": int(g["n"].sum()),
            "coverage_overall": wavg("coverage_overall"),
            "empty_rate(review)": wavg("empty_rate(review)"),
            "singleton_rate": wavg("singleton_rate"),
            "ambiguity_rate{0,1}": wavg("ambiguity_rate{0,1}"),
            "avg_set_size": wavg("avg_set_size"),
            "coverage_y=0": wavg("coverage_y=0"),
            "coverage_y=1": wavg("coverage_y=1"),
        }
        return pd.Series(out)

    summary = per_week.groupby("method", as_index=False).apply(agg).reset_index(drop=True)
    summary_csv = OUT_DIR / f"summary_W{W}_alpha{alpha:.2f}_bins{N_BINS_X}.csv"
    summary.to_csv(summary_csv, index=False)

    print("Saved per-week ->", per_week_csv)
    print("Saved summary  ->", summary_csv)
    print("\n=== SUMMARY ===")
    print(summary.sort_values("coverage_overall", ascending=False))

if __name__ == "__main__":
    run()