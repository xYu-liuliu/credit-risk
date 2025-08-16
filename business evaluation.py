import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


DATA_DIR  = Path(r"E:/Home Credit Processed Feature")
TRAIN_PRED= DATA_DIR / "train_pred.csv"  
VAL_PRED  = DATA_DIR / "val_pred.csv"    
TEST_PRED = DATA_DIR / "test_pred.csv"   
OUT_DIR   = DATA_DIR / "business analysis"
OUT_DIR.mkdir(exist_ok=True, parents=True)

CAPS = (0.016, 0.018, 0.020)     
PR_KS = (0.05, 0.10, 0.15)

plt.style.use("seaborn-v0_8") 

def threshold_curves(y_true, y_prob, grid=None):
    
    if grid is None:
        grid = np.linspace(0, 1, 201)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    rows = []
    for t in grid:
        approved = y_prob <= t          
        n_app = int(approved.sum())
        app_rate = n_app / n
        bad_rate = y_true[approved].mean() if n_app > 0 else np.nan
        rows.append((t, app_rate, bad_rate, n_app))
    return pd.DataFrame(rows, columns=["threshold", "approval_rate", "bad_rate_in_approved", "n_approved"])

def pr_at_k(y_true, y_prob, ks=PR_KS):
    """Precision@K / Recall@K """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    order = np.argsort(-y_prob)   
    y_sorted = y_true[order]
    res, total_pos = [], y_true.sum()
    for k in ks:
        top = max(1, int(np.floor(n * k)))
        sel = y_sorted[:top]
        precision = sel.mean()
        recall = sel.sum() / total_pos if total_pos > 0 else np.nan
        res.append({"K%": f"{int(k*100)}%", "topN": top,
                    "precision_at_K": precision, "recall_at_K": recall})
    return pd.DataFrame(res)

def lift_curve(y_true, y_prob, n_bins=20):
    """Cumulative gain / lift """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    bin_sizes = [n // n_bins + (1 if i < n % n_bins else 0) for i in range(n_bins)]
    starts = np.cumsum([0] + bin_sizes[:-1])
    ends   = np.cumsum(bin_sizes)
    total_pos = y_true.sum()
    base_rate = total_pos / n if n>0 else np.nan
    rows, cum_pos = [], 0
    for i, (s,e) in enumerate(zip(starts, ends), start=1):
        seg = y_sorted[s:e]
        cum_pos += seg.sum()
        cum_recall = cum_pos / total_pos if total_pos > 0 else np.nan
        lift = (cum_pos / e) / base_rate if base_rate>0 else np.nan
        rows.append((i, e, cum_recall, lift))
    return pd.DataFrame(rows, columns=["bin_idx","cum_n","cum_recall","lift"])

def psi_from_scores(expected_scores, actual_scores, bins=10, eps=1e-6, fixed_edges=None):
    """
    PSI between expected and actual distributions.
    """
    e = np.asarray(expected_scores)
    a = np.asarray(actual_scores)
    if fixed_edges is None:
        qs = np.quantile(e, np.linspace(0,1,bins+1))
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

def pick_threshold_by_bad_cap(df, prob_col="y_prob", target_col="target", bad_cap=0.02):
    """
    Given a bad-rate cap, return the threshold that maximizes approval while
    """
    tmp = df[[prob_col, target_col]].dropna().sort_values(prob_col, ascending=True).reset_index(drop=True)
    cum_bads = tmp[target_col].astype(int).cumsum()
    n = np.arange(1, len(tmp) + 1)
    cum_bad_rate = cum_bads / n
    ok = np.where(cum_bad_rate <= bad_cap)[0]
    if len(ok) == 0:
        return None
    k = ok[-1]
    return {
        "threshold": float(tmp.loc[k, prob_col]),
        "approve_rate": float((k + 1) / len(tmp)),
        "bad_rate_at_threshold": float(cum_bad_rate[k])
    }

def do_all(name, df, out_prefix):
    """Dump all"""
    df = df.copy()
    assert {"y_prob","target"}.issubset(df.columns), f"{name} lacks y_prob/target"
    df["target"] = df["target"].astype(int)
    # threshold curves
    thr_df = threshold_curves(df["target"], df["y_prob"])
    thr_df.to_csv(OUT_DIR / f"{out_prefix}_threshold_curve.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(7,4))
    ax1.plot(thr_df["threshold"], thr_df["approval_rate"])
    ax1.set_xlabel("Threshold t (approve if y_prob ≤ t)")
    ax1.set_ylabel("Approval rate")
    ax2 = ax1.twinx()
    ax2.plot(thr_df["threshold"], thr_df["bad_rate_in_approved"])
    ax2.set_ylabel("Bad rate (among approved)")
    plt.title(f"{name}: Threshold → Approval & Bad rate")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{out_prefix}_curve_threshold_approval_badrate.png", dpi=150)

    # PR@K
    prk = pr_at_k(df["target"], df["y_prob"])
    prk.to_csv(OUT_DIR / f"{out_prefix}_pr_at_k.csv", index=False)
    # lift & cumulative gain
    lift_df = lift_curve(df["target"], df["y_prob"], n_bins=20)
    lift_df.to_csv(OUT_DIR / f"{out_prefix}_lift_detail.csv", index=False)
    plt.figure(figsize=(7,4))
    plt.plot(lift_df["bin_idx"]/20, lift_df["cum_recall"], marker=".")
    plt.xlabel("Population proportion (from worst to better)")
    plt.ylabel("Cumulative recall")
    plt.title(f"{name}: Cumulative Gain")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{out_prefix}_curve_cum_gain.png", dpi=150)

    plt.figure(figsize=(7,4))
    plt.plot(lift_df["bin_idx"]/20, lift_df["lift"], marker=".")
    plt.xlabel("Population proportion (from worst to better)")
    plt.ylabel("Lift")
    plt.title(f"{name}: Lift Curve")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{out_prefix}_curve_lift.png", dpi=150)
    
    base_bad = df["target"].mean()
    print(f"[{name}] base bad rate = {base_bad:.3%}")
    return {"base_bad_rate": base_bad}

def weekly_stability(df, threshold, *,
                     baseline_scores_for_psi=None,
                     bin_edges=None,
                     approved_only_for_psi=True,
                     min_week_n=30):
    """
    Per-week stability:
      approval_rate, bad_rate_in_approved, bad_capture_rate
      weekly PSI 
    """
    rows = []
    for w, g in df.groupby("WEEK_NUM"):
        g = g.copy()
        g["target"] = g["target"].astype(int)
        n = len(g)
        approve_mask = (g["y_prob"] <= threshold)

        # approval rate & bad rate among approved
        app_rate = float(approve_mask.mean())
        bad_in_approved = float(g.loc[approve_mask, "target"].mean()) if approve_mask.any() else np.nan

        # capture rate = rejected bads / total bads in week
        tot_bad = int(g["target"].sum())
        rej_bad = int(g.loc[~approve_mask, "target"].sum())
        capture_rate = (rej_bad / tot_bad) if tot_bad > 0 else np.nan

        # weekly PSI (approved subset) vs baseline, if provided
        psi = np.nan
        if baseline_scores_for_psi is not None:
            scores_for_week = g.loc[approve_mask, "y_prob"].to_numpy() if approved_only_for_psi else g["y_prob"].to_numpy()
            if len(scores_for_week) >= min_week_n:
                psi, _, _ = psi_from_scores(baseline_scores_for_psi, scores_for_week,
                                            bins=10, fixed_edges=bin_edges)

        rows.append({
            "WEEK_NUM": int(w),
            "n": int(n),
            "approve_rate": app_rate,
            "bad_rate_in_approved": bad_in_approved,
            "bad_capture_rate": capture_rate,
            "psi": psi
        })
    return pd.DataFrame(rows).sort_values("WEEK_NUM").reset_index(drop=True)

def plot_week_lines(week_df, cap, title_prefix, out_prefix, show_psi=True):
    """Plot weekly bad-in-approved, weekly approval, and weekly PSI."""
    # weekly bad rate in approved
    plt.figure(figsize=(8,4))
    plt.plot(week_df["WEEK_NUM"], week_df["bad_rate_in_approved"], marker="o")
    plt.axhline(cap, linestyle="--", label=f"cap={cap:.2%}")
    plt.legend()
    plt.title(f"{title_prefix}: Weekly bad rate (in approved)")
    plt.xlabel("WEEK_NUM"); plt.ylabel("Bad rate in approved")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{out_prefix}_weekly_bad_in_approved.png", dpi=150)

    # weekly approval rate
    plt.figure(figsize=(8,4))
    plt.plot(week_df["WEEK_NUM"], week_df["approve_rate"], marker="o")
    plt.title(f"{title_prefix}: Weekly approval rate")
    plt.xlabel("WEEK_NUM"); plt.ylabel("Approval rate")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{out_prefix}_weekly_approve.png", dpi=150)

    # weekly PSI (if any)
    if show_psi and "psi" in week_df and not week_df["psi"].isna().all():
        plt.figure(figsize=(8,4))
        plt.plot(week_df["WEEK_NUM"], week_df["psi"], marker="o")
        plt.title(f"{title_prefix}: Weekly PSI (approved vs VAL-approved baseline)")
        plt.xlabel("WEEK_NUM"); plt.ylabel("PSI")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{out_prefix}_weekly_psi.png", dpi=150)

def build_strategy_table_with_stability(train, val, test, caps=(0.016, 0.018, 0.020)):
    rows = []
    for cap in caps:
        # calculate threshold from VAL under this bad_cap
        res = pick_threshold_by_bad_cap(val, "y_prob", "target", bad_cap=cap)
        if res is None:
            continue
        t = res["threshold"]

        # VAL approved scores as PSI baseline 
        val_approved_scores = val.loc[val["y_prob"] <= t, "y_prob"].to_numpy()
        if len(val_approved_scores) == 0:
            continue
        _, _, bin_edges = psi_from_scores(val_approved_scores, val_approved_scores, bins=10)

        # weekly stability
        #    VAL: PSI vs VAL-approved
        val_w  = weekly_stability(val,  t,
                                  baseline_scores_for_psi=val_approved_scores,
                                  bin_edges=bin_edges,
                                  approved_only_for_psi=True)
        # TEST: PSI vs VAL-approved
        test_w = weekly_stability(test, t,
                                  baseline_scores_for_psi=val_approved_scores,
                                  bin_edges=bin_edges,
                                  approved_only_for_psi=True)

        # plots
        val_w.to_csv(OUT_DIR / f"val_weekly_cap{cap:.3f}.csv", index=False)
        test_w.to_csv(OUT_DIR / f"test_weekly_cap{cap:.3f}.csv", index=False)
        plot_week_lines(val_w,  cap, f"VAL cap={cap:.3%}",              f"val_cap{cap:.3f}",  show_psi=True)
        plot_week_lines(test_w, cap, f"TEST cap={cap:.3%} (Val→Test)",  f"test_cap{cap:.3f}", show_psi=True)

        # 4) overall aggregates
        val_mask  = (val["y_prob"]  <= t)
        test_mask = (test["y_prob"] <= t)
        val_approve = float(val_mask.mean());  test_approve = float(test_mask.mean())
        val_bad_in  = float(val.loc[val_mask,  "target"].mean())
        test_bad_in = float(test.loc[test_mask, "target"].mean())
        val_cap_rate  = float(val.loc[~val_mask,  "target"].sum() / max(1, val["target"].sum()))
        test_cap_rate = float(test.loc[~test_mask, "target"].sum() / max(1, test["target"].sum()))
        
        val_appr  = val.loc[val["y_prob"]  <= t, "y_prob"].to_numpy()
        test_appr = test.loc[test["y_prob"] <= t, "y_prob"].to_numpy()
        overall_psi, _, _ = psi_from_scores(val_appr, test_appr, bins=20, fixed_edges=bin_edges)
        
        rows.append({
            "bad_cap": cap,
            "threshold": t,
            "val_approve_rate": val_approve,
            "val_bad_rate_in_approved": val_bad_in,
            "val_approved_n": int(val_mask.sum()),
            "val_bad_capture_rate": val_cap_rate,
            "test_approve_rate": test_approve,
            "test_bad_rate_in_approved": test_bad_in,
            "test_approved_n": int(test_mask.sum()),
            "test_bad_capture_rate": test_cap_rate,
            "val_%weeks_over_cap": float((val_w["bad_rate_in_approved"] > cap).mean()) if len(val_w) else np.nan,
            "val_worst_week_bad":  float(val_w["bad_rate_in_approved"].max()) if len(val_w) else np.nan,
            "val_std_bad":         float(val_w["bad_rate_in_approved"].std(ddof=0)) if len(val_w) else np.nan,
            "test_%weeks_over_cap": float((test_w["bad_rate_in_approved"] > cap).mean()) if len(test_w) else np.nan,
            "test_worst_week_bad":  float(test_w["bad_rate_in_approved"].max()) if len(test_w) else np.nan,
            "test_std_bad":         float(test_w["bad_rate_in_approved"].std(ddof=0)) if len(test_w) else np.nan,
            "test_overall_PSI_approved_vsVAL":     float(overall_psi),
            "test_median_week_PSI_approved_vsVAL": float(test_w["psi"].median()) if "psi" in test_w else np.nan,
            "test_max_week_PSI_approved_vsVAL":    float(test_w["psi"].max())    if "psi" in test_w else np.nan,
        })

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_DIR / "strategy_table_with_weekly_stability.csv", index=False)
    print("✅ saved:", OUT_DIR / "strategy_table_with_weekly_stability.csv")
    return tab


if __name__ == "__main__":
    # load 
    train = pd.read_csv(TRAIN_PRED)
    val   = pd.read_csv(VAL_PRED)
    test  = pd.read_csv(TEST_PRED)

    # per-split evaluation (threshold curves / PR@K / Lift)
    print("→ VAL")
    do_all("VAL",  val,  "val")
    print("→ TEST")
    do_all("TEST", test, "test")

    strategy_tab = build_strategy_table_with_stability(train, val, test, caps=CAPS)
    

