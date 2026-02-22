from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

# ======================
# Paths
# ======================
DATA_DIR = Path(r"E:/Home Credit Processed Feature")
VAL_PRED  = DATA_DIR / "val_pred.csv"
TEST_PRED = DATA_DIR / "test_pred.csv"
OUT_DIR   = DATA_DIR / "calibration_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8")

REQ_COLS = {"target", "y_prob"}
# 你要保留的额外列（可以继续加，比如 SK_ID_CURR / case_id）
KEEP_COLS = ["WEEK_NUM"]

def read_pred(path: Path, name: str, keep_cols=None) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQ_COLS - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] missing required columns: {missing}")

    keep_cols = keep_cols or []
    existing_keep = [c for c in keep_cols if c in df.columns]

    # 只保留必要列 + 你想保留的列（避免无关列污染输出）
    df = df[list(REQ_COLS) + existing_keep].copy()

    df["target"] = pd.to_numeric(df["target"], errors="coerce").astype("Int64")
    df["y_prob"] = pd.to_numeric(df["y_prob"], errors="coerce")

    # WEEK_NUM 统一做成整数（如果存在）
    if "WEEK_NUM" in df.columns:
        df["WEEK_NUM"] = pd.to_numeric(df["WEEK_NUM"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["target", "y_prob"])
    df["target"] = df["target"].astype(int)

    # clip just in case of tiny numerical out-of-range
    df["y_prob"] = df["y_prob"].clip(0.0, 1.0)

    return df

# ----------------------
# Metrics
# ----------------------
def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True)
    bin_ids[bin_ids == 0] = 1

    total = len(y_true)
    ece = 0.0
    for b in range(1, n_bins + 1):
        idx = bin_ids == b
        cnt = int(idx.sum())
        if cnt == 0:
            continue
        avg_conf = float(y_prob[idx].mean())
        emp_freq = float(y_true[idx].mean())
        ece += (cnt / total) * abs(emp_freq - avg_conf)
    return float(ece)

def reliability_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True)
    bin_ids[bin_ids == 0] = 1

    xs, ys, ns = [], [], []
    for b in range(1, n_bins + 1):
        idx = bin_ids == b
        if idx.sum() == 0:
            continue
        xs.append(float(y_prob[idx].mean()))
        ys.append(float(y_true[idx].mean()))
        ns.append(int(idx.sum()))
    return np.array(xs), np.array(ys), np.array(ns)

def plot_reliability(y_true, y_prob, title, out_path, n_bins=20):
    xs, ys, ns = reliability_points(y_true, y_prob, n_bins=n_bins)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    plt.scatter(xs, ys)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical positive rate")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# ----------------------
# Calibrators
# ----------------------
def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))

class PlattCalibrator:
    """Sigmoid calibration: fit LR on logit(p_raw)."""
    def __init__(self):
        self.lr = LogisticRegression(solver="lbfgs", max_iter=300)

    def fit(self, p_raw, y):
        z = _logit(p_raw)
        self.lr.fit(z.reshape(-1, 1), np.asarray(y).astype(int))
        return self

    def predict(self, p_raw):
        z = _logit(p_raw)
        return self.lr.predict_proba(z.reshape(-1, 1))[:, 1]

class IsotonicCalibratorWrap:
    """Isotonic calibration: nonparametric monotone mapping."""
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p_raw, y):
        self.iso.fit(np.asarray(p_raw, dtype=float), np.asarray(y).astype(int))
        return self

    def predict(self, p_raw):
        return self.iso.predict(np.asarray(p_raw, dtype=float))

# ----------------------
# Compare helper
# ----------------------
def summarize(y, p, tag, n_bins=20):
    return {
        "tag": tag,
        "n": int(len(y)),
        "pos_rate": float(np.mean(y)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece_score(y, p, n_bins=n_bins)),
    }

def dist_compare(p_raw, p_cal):
    p_raw = np.asarray(p_raw, dtype=float)
    p_cal = np.asarray(p_cal, dtype=float)
    corr = float(np.corrcoef(p_raw, p_cal)[0, 1]) if len(p_raw) > 1 else np.nan
    mae = float(np.mean(np.abs(p_cal - p_raw)))
    q = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    raw_q = np.quantile(p_raw, q)
    cal_q = np.quantile(p_cal, q)
    qdf = pd.DataFrame({"q": q, "raw": raw_q, "cal": cal_q, "cal_minus_raw": cal_q - raw_q})
    return corr, mae, qdf

def metrics_by_week(df: pd.DataFrame, prob_col: str, tag_prefix: str, n_bins: int = 20) -> pd.DataFrame:
    # 没有 WEEK_NUM 就直接返回空
    if "WEEK_NUM" not in df.columns:
        return pd.DataFrame()

    rows = []
    for wk, g in df.groupby("WEEK_NUM"):
        y = g["target"].to_numpy()
        p = g[prob_col].to_numpy()
        rows.append({
            "tag": f"{tag_prefix}_W{int(wk)}",
            "WEEK_NUM": int(wk),
            "n": int(len(g)),
            "pos_rate": float(np.mean(y)),
            "brier": float(brier_score_loss(y, p)),
            "ece": float(ece_score(y, p, n_bins=n_bins)),
        })
    return pd.DataFrame(rows).sort_values(["WEEK_NUM"])

def main():
    val = read_pred(VAL_PRED, "VAL", keep_cols=KEEP_COLS)
    test = read_pred(TEST_PRED, "TEST", keep_cols=KEEP_COLS)

    y_val = val["target"].values
    p_val_raw = val["y_prob"].values
    y_test = test["target"].values
    p_test_raw = test["y_prob"].values

    # ===== choose method =====
    METHOD = "platt"   # "platt" or "isotonic"
    N_BINS = 20

    if METHOD == "platt":
        cal = PlattCalibrator().fit(p_val_raw, y_val)
    elif METHOD == "isotonic":
        cal = IsotonicCalibratorWrap().fit(p_val_raw, y_val)
    else:
        raise ValueError("METHOD must be 'platt' or 'isotonic'")

    # apply
    p_val_cal = cal.predict(p_val_raw)
    p_test_cal = cal.predict(p_test_raw)

    # 写回 df（关键：WEEK_NUM 会被一起保留）
    val_out = val.copy()
    test_out = test.copy()
    val_out["y_prob_cal"] = p_val_cal
    test_out["y_prob_cal"] = p_test_cal

    # ===== metrics compare =====
    rows = []
    rows.append(summarize(y_val,  p_val_raw,  "VAL_RAW",  n_bins=N_BINS))
    rows.append(summarize(y_val,  p_val_cal,  "VAL_CAL",  n_bins=N_BINS))
    rows.append(summarize(y_test, p_test_raw, "TEST_RAW", n_bins=N_BINS))
    rows.append(summarize(y_test, p_test_cal, "TEST_CAL", n_bins=N_BINS))
    metrics_df = pd.DataFrame(rows)

    metrics_csv = OUT_DIR / f"metrics_raw_vs_cal_{METHOD}.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print("Saved:", metrics_csv)
    print(metrics_df)

    # ===== reliability plots (TEST) =====
    plot_reliability(y_test, p_test_raw,
                     f"Reliability (RAW, TEST) [{METHOD}]",
                     OUT_DIR / f"reliability_test_raw_{METHOD}.png",
                     n_bins=N_BINS)
    plot_reliability(y_test, p_test_cal,
                     f"Reliability (CAL, TEST) [{METHOD}]",
                     OUT_DIR / f"reliability_test_cal_{METHOD}.png",
                     n_bins=N_BINS)
    print("Saved reliability plots to:", OUT_DIR)

    # ===== how much changed? (distribution comparison) =====
    corr, mae, qdf = dist_compare(p_val_raw, p_val_cal)
    q_csv = OUT_DIR / f"val_prob_shift_quantiles_{METHOD}.csv"
    qdf.to_csv(q_csv, index=False)
    with open(OUT_DIR / f"val_prob_shift_summary_{METHOD}.txt", "w", encoding="utf-8") as f:
        f.write(f"METHOD={METHOD}\n")
        f.write(f"corr(raw, cal) = {corr:.6f}\n")
        f.write(f"mean_abs_diff  = {mae:.6f}\n")
        f.write("See quantiles CSV for details.\n")
    print("Saved:", q_csv)
    print(f"corr(raw,cal)={corr:.6f} | mean_abs_diff={mae:.6f}")

    # ===== save calibrated preds (with WEEK_NUM if exists) =====
    val_out.to_csv(OUT_DIR / f"val_pred_with_cal_{METHOD}.csv",
                   index=False, float_format="%.7g")
    test_out.to_csv(OUT_DIR / f"test_pred_with_cal_{METHOD}.csv",
                    index=False, float_format="%.7g")
    print("Saved calibrated CSVs.")

    # ===== optional: metrics by week (raw vs cal) =====
    wk1 = metrics_by_week(test_out.assign(y_prob=test_out["y_prob"]), "y_prob", "TEST_RAW", n_bins=N_BINS)
    wk2 = metrics_by_week(test_out, "y_prob_cal", "TEST_CAL", n_bins=N_BINS)
    if len(wk1) and len(wk2):
        wk_df = wk1.merge(
            wk2[["WEEK_NUM", "brier", "ece", "pos_rate", "n"]],
            on="WEEK_NUM",
            suffixes=("_raw", "_cal"),
            how="outer",
        ).sort_values("WEEK_NUM")
        wk_csv = OUT_DIR / f"metrics_by_week_test_{METHOD}.csv"
        wk_df.to_csv(wk_csv, index=False)
        print("Saved:", wk_csv)

if __name__ == "__main__":

    main()
