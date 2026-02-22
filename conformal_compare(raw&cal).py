from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(r"E:/Home Credit Processed Feature")
CAL_DIR  = DATA_DIR / "calibration_compare"

alpha = 0.05

# RAW preds
VAL_RAW  = DATA_DIR / "val_pred.csv"
TEST_RAW = DATA_DIR / "test_pred.csv"

# CAL preds (from your platt outputs)
VAL_CAL  = CAL_DIR / "val_pred_with_cal_platt.csv"
TEST_CAL = CAL_DIR / "test_pred_with_cal_platt.csv"

def conformal_binary(val_df, test_df, prob_col, alpha):
    y_val = val_df["target"].to_numpy().astype(int)
    p_val = val_df[prob_col].to_numpy().astype(float)

    y_test = test_df["target"].to_numpy().astype(int)
    p_test = test_df[prob_col].to_numpy().astype(float)

    # scores on VAL
    scores = np.where(y_val == 1, 1 - p_val, p_val)
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    q = float(np.sort(scores)[k - 1])

    # prediction sets on TEST
    include_1 = (1 - p_test) <= q
    include_0 = (p_test) <= q
    set_size = include_0.astype(int) + include_1.astype(int)

    true_in_set = np.where(y_test == 1, include_1, include_0)
    coverage = float(true_in_set.mean())

    empty_rate = float((set_size == 0).mean())
    singleton_rate = float((set_size == 1).mean())
    ambiguity_rate = float((set_size == 2).mean())
    avg_set_size = float(set_size.mean())

    # “middle region” that produces empty when q < 0.5
    # empty happens when q < p < 1-q
    q_low = q
    q_high = 1 - q

    return {
        "alpha": alpha,
        "n_val": n,
        "q": q,
        "q_low": q_low,
        "q_high": q_high,
        "test_coverage": coverage,
        "empty_rate(review)": empty_rate,
        "singleton_rate": singleton_rate,
        "ambiguity_rate{0,1}": ambiguity_rate,
        "avg_set_size": avg_set_size,
    }

def main():
    val_raw  = pd.read_csv(VAL_RAW)
    test_raw = pd.read_csv(TEST_RAW)
    val_cal  = pd.read_csv(VAL_CAL)
    test_cal = pd.read_csv(TEST_CAL)

    out = []
    out.append({"system": "RAW"} | conformal_binary(val_raw, test_raw, "y_prob", alpha))
    out.append({"system": "CAL(Platt)"} | conformal_binary(val_cal, test_cal, "y_prob_cal", alpha))

    res = pd.DataFrame(out)
    print(res)

    out_csv = CAL_DIR / f"conformal_compare_raw_vs_cal_alpha_{alpha:.2f}.csv"
    res.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

if __name__ == "__main__":

    main()
