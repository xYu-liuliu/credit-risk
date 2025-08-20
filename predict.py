from pathlib import Path
import pandas as pd
import lightgbm as lgb
from pandas.api.types import CategoricalDtype

DATA_DIR = Path(r"E:/Home Credit Processed Feature")
MODEL_F  = DATA_DIR / "lgbm_timecv_prauc.txt"

INPUTS = {
    "TRAIN": DATA_DIR / "train_sel.parquet",
    "VAL":   DATA_DIR / "val_sel.parquet",
    "TEST":  DATA_DIR / "test_sel.parquet",
}

OUTPUTS = {
    "TRAIN": DATA_DIR / "train_pred.csv",
    "VAL":   DATA_DIR / "val_pred.csv",
    "TEST":  DATA_DIR / "test_pred.csv",
}

NON_FEATURES = {"target", "WEEK_NUM", "case_id"}

# load
booster = lgb.Booster(model_file=str(MODEL_F))
feat_names = booster.feature_name() 

def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-features, cast categoricals, and align column order to the model."""
    X = df.drop(columns=[c for c in NON_FEATURES if c in df.columns], errors="ignore").copy()

    # cast categorical month/weekday if present
    if "month_decision" in X.columns:
        X["month_decision"] = X["month_decision"].astype("int32").astype(
            CategoricalDtype(categories=list(range(1, 13)), ordered=False)
        )
    if "weekday_decision" in X.columns:
        X["weekday_decision"] = X["weekday_decision"].astype("int32").astype(
            CategoricalDtype(categories=list(range(1, 8)), ordered=False)
        )

    
    missing = [f for f in feat_names if f not in X.columns]
    if missing:
        raise ValueError(f"Missing features required by model: {missing[:10]}{'...' if len(missing)>10 else ''}")
    X = X[feat_names]
    return X

def predict_split(tag: str, in_path: Path, out_path: Path):
    if not in_path.exists():
        print(f"[{tag}] skip: file not found -> {in_path}")
        return

    df = pd.read_parquet(in_path)
    X  = prepare_X(df)
    y_prob = booster.predict(X)

    keep_cols = [c for c in ["case_id", "WEEK_NUM", "target"] if c in df.columns]
    out = df[keep_cols].copy() if keep_cols else pd.DataFrame(index=df.index)
    out["y_prob"] = y_prob

    out.to_csv(out_path, index=False, float_format="%.7g")
    print(f"[{tag}] saved: {out_path} | shape: {out.shape}")


    try:
        print(out.head(3))
    except Exception:
        pass

def main():
    for tag in ["TRAIN", "VAL", "TEST"]:
        predict_split(tag, INPUTS[tag], OUTPUTS[tag])

if __name__ == "__main__":
    main()










