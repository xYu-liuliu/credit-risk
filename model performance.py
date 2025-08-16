import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, roc_curve
import matplotlib.pyplot as plt


DATA_DIR  = Path(r"E:/Home Credit Processed Feature")
TRAIN_PRED= DATA_DIR / "train_pred.csv"
VAL_PRED  = DATA_DIR / "val_pred.csv"
TEST_PRED = DATA_DIR / "test_pred.csv"
OUT_DIR   = DATA_DIR / "model analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8") 

REQ_COLS = {"WEEK_NUM", "target", "y_prob"}

def read_pred(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQ_COLS - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] missing required columns: {missing}")
    df = df.copy()
    df["WEEK_NUM"] = pd.to_numeric(df["WEEK_NUM"], errors="coerce")
    df["target"]   = pd.to_numeric(df["target"],   errors="coerce").astype(int)
    df["y_prob"]   = pd.to_numeric(df["y_prob"],   errors="coerce")
    df = df.dropna(subset=["WEEK_NUM", "target", "y_prob"])
    return df

def ks_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(np.abs(tpr - fpr)))

def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20):
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

def evaluate_split(df: pd.DataFrame, tag: str):
    """
    Evaluate overall metrics for one split and also compute weekly metrics.
    """
    y = df["target"].values
    p = df["y_prob"].values

    pr_auc  = average_precision_score(y, p)
    roc_auc = roc_auc_score(y, p)
    ks      = ks_score(y, p)
    brier   = brier_score_loss(y, p)
    ece     = ece_score(y, p, n_bins=20)

    summary = {
        "split": tag,
        "n": len(df),
        "positive_rate": float(np.mean(y)),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "ks": float(ks),
        "brier": float(brier),
        "ece": float(ece),
    }

    # Weekly stability metrics
    def _weekly_metrics(g: pd.DataFrame):
        yy = g["target"].values
        pp = g["y_prob"].values
        uniq = np.unique(yy)
        if len(uniq) < 2:
            # If only one class present in that week, metrics are undefined
            return pd.Series({
                "n": len(g),
                "pos_rate": float(np.mean(yy)),
                "pr_auc": np.nan,
                "roc_auc": np.nan,
                "ks": np.nan,
            })
        return pd.Series({
            "n": len(g),
            "pos_rate": float(np.mean(yy)),
            "pr_auc": float(average_precision_score(yy, pp)),
            "roc_auc": float(roc_auc_score(yy, pp)),
            "ks": float(ks_score(yy, pp)),
        })

    weekly = (df.groupby("WEEK_NUM", as_index=False)
                .apply(_weekly_metrics)
                .sort_values("WEEK_NUM"))
    return summary, weekly

def plot_weekly(weekly_df: pd.DataFrame, title: str, out_path: Path):
    """
    Plot KS, ROC-AUC, and PR-AUC curves across weeks.
    """
    x = weekly_df["WEEK_NUM"].values
    plt.figure(figsize=(9, 5))
    for col, label in [("ks","KS"), ("roc_auc","ROC-AUC"), ("pr_auc","PR-AUC")]:
        y = weekly_df[col].values
        ok = ~np.isnan(y)
        if ok.any():
            plt.plot(x[ok], y[ok], marker="o", markersize=4, linewidth=1.0, label=label)

    plt.title(title)
    plt.xlabel("WEEK_NUM")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def psi_from_scores(expected_scores, actual_scores, bins=10, eps=1e-6, fixed_edges=None):
    """
    PSI between expected and actual score distributions.
    """
    e = np.asarray(expected_scores, dtype=float)
    a = np.asarray(actual_scores, dtype=float)
    if fixed_edges is None:
        qs = np.quantile(e, np.linspace(0, 1, bins + 1))
        qs[0], qs[-1] = -np.inf, np.inf
    else:
        qs = fixed_edges

    e_hist, _ = np.histogram(e, bins=qs)
    a_hist, _ = np.histogram(a, bins=qs)
    e_rate = e_hist / max(1, len(e))
    a_rate = a_hist / max(1, len(a))
    contrib = (a_rate - e_rate) * np.log((a_rate + eps) / (e_rate + eps))
    psi_val = float(np.sum(contrib))

    detail = pd.DataFrame({
        "bin_left": qs[:-1], "bin_right": qs[1:],
        "exp_rate": e_rate,  "act_rate": a_rate,
        "contrib": contrib
    })
    return psi_val, detail, qs

def weekly_model_psi(actual_df: pd.DataFrame, expected_scores: np.ndarray, edges, min_week_n: int = 30):
    """
    Compute weekly PSI (all applications) vs expected distribution.
    """
    rows = []
    for w, g in actual_df.groupby("WEEK_NUM"):
        a = g["y_prob"].to_numpy()
        if len(a) >= min_week_n:
            v, _, _ = psi_from_scores(expected_scores, a, bins=10, fixed_edges=edges)
        else:
            v = np.nan
        rows.append({"WEEK_NUM": int(w), "n": int(len(a)), "psi": float(v)})
    return pd.DataFrame(rows).sort_values("WEEK_NUM").reset_index(drop=True)

def plot_weekly_psi(weekly_df: pd.DataFrame, title: str, out_path: Path):
    """
    Plot weekly PSI curve.
    """
    x = weekly_df["WEEK_NUM"].values
    y = weekly_df["psi"].values
    ok = ~np.isnan(y)
    plt.figure(figsize=(9, 4))
    if ok.any():
        plt.plot(x[ok], y[ok], marker="o", markersize=4, linewidth=1.0)
    plt.title(title)
    plt.xlabel("WEEK_NUM")
    plt.ylabel("PSI")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    
    
def main():
    # Load 
    splits = {
        "TRAIN": read_pred(TRAIN_PRED, "TRAIN"),
        "VAL":   read_pred(VAL_PRED,   "VAL"),
        "TEST":  read_pred(TEST_PRED,  "TEST"),
    }

    # Overall metrics (AUC/PR-AUC/KS/Brier/ECE) + Weekly curves 
    summaries, weekly_map = [], {}
    for tag, df in splits.items():
        s, w = evaluate_split(df, tag)
        summaries.append(s)
        weekly_map[tag] = w

    overall_csv = OUT_DIR / "overall_metrics.csv"
    pd.DataFrame(summaries).to_csv(overall_csv, index=False)
    print(f"Saved overall metrics → {overall_csv}")

    val_png  = OUT_DIR / "VAL_weekly.png"
    test_png = OUT_DIR / "TEST_weekly.png"
    plot_weekly(weekly_map["VAL"],  "Weekly Stability (VAL)",  val_png)
    plot_weekly(weekly_map["TEST"], "Weekly Stability (TEST)", test_png)
    print(f"Saved weekly plots → {val_png}")
    print(f"Saved weekly plots → {test_png}")

    
    train_scores = splits["TRAIN"]["y_prob"].to_numpy()
    val_scores   = splits["VAL"]["y_prob"].to_numpy()
    test_scores  = splits["TEST"]["y_prob"].to_numpy()

    # overall PSI
    psi_train_val, _, edges = psi_from_scores(train_scores, val_scores, bins=20, fixed_edges=None)
    psi_train_test, _, _    = psi_from_scores(train_scores, test_scores, bins=20, fixed_edges=edges)

    overall_psi = pd.DataFrame([
        {"pair": "TRAIN→VAL",  "psi": psi_train_val},
        {"pair": "TRAIN→TEST", "psi": psi_train_test},
    ])
    overall_psi_csv = OUT_DIR / "model_psi_overall.csv"
    overall_psi.to_csv(overall_psi_csv, index=False)
    print(f"Saved model PSI (overall) → {overall_psi_csv}")

    # weekly PSI
    val_weekly_psi  = weekly_model_psi(splits["VAL"],  train_scores, edges)
    test_weekly_psi = weekly_model_psi(splits["TEST"], train_scores, edges)

    val_weekly_psi_csv  = OUT_DIR / "VAL_weekly_model_psi.csv"
    test_weekly_psi_csv = OUT_DIR / "TEST_weekly_model_psi.csv"
    val_weekly_psi.to_csv(val_weekly_psi_csv, index=False)
    test_weekly_psi.to_csv(test_weekly_psi_csv, index=False)
    print(f"Saved weekly model PSI → {val_weekly_psi_csv}")
    print(f"Saved weekly model PSI → {test_weekly_psi_csv}")

    # weekly PSI plots
    val_psi_png  = OUT_DIR / "VAL_weekly_model_psi.png"
    test_psi_png = OUT_DIR / "TEST_weekly_model_psi.png"
    plot_weekly_psi(val_weekly_psi,  "Weekly Model PSI (VAL vs TRAIN)",  val_psi_png)
    plot_weekly_psi(test_weekly_psi, "Weekly Model PSI (TEST vs TRAIN)", test_psi_png)
    print(f"Saved weekly PSI plots → {val_psi_png}")
    print(f"Saved weekly PSI plots → {test_psi_png}")

if __name__ == "__main__":
    main()