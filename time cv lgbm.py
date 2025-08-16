# timecv_optuna_pr_auc.py
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna, joblib
from pathlib import Path
from sklearn.metrics import average_precision_score
from pandas.api.types import CategoricalDtype
import math

SEED = 42
DATA_DIR = Path(r"E:/Home Credit Processed Feature")
TRAIN_F  = DATA_DIR / "train_sel.parquet"    
OUT_PKL  = DATA_DIR / "best_params_timecv_prauc.pkl"
OUT_MODEL= DATA_DIR / "lgbm_timecv_prauc.txt"


def make_expanding_folds(df: pd.DataFrame,
                         time_col: str = "WEEK_NUM",
                         start_week: int = 0,
                         end_week: int = 69,
                         n_folds: int = 4,
                         val_span: int = 4):
    """
    Each validation window is 4 weeks, and the ends of validation windows are evenly distributed near end_week.
    """
    df = df.reset_index(drop=False).rename(columns={"index": "_idx"})
    folds = []
    for i in range(n_folds):
        val_end   = end_week - (n_folds - 1 - i) * val_span
        val_start = val_end - val_span + 1
        tr_idx = df.loc[(df[time_col] >= start_week) & (df[time_col] <  val_start), "_idx"].to_numpy()
        va_idx = df.loc[(df[time_col] >= val_start)   & (df[time_col] <= val_end),   "_idx"].to_numpy()
        if len(tr_idx) == 0 or len(va_idx) == 0:
            raise ValueError(f"Fold {i} empty. Check weeks and filters.")
        folds.append((tr_idx, va_idx))
    return folds

# Optuna 
def run_study(df: pd.DataFrame, n_trials: int = 100):
    # y / X
    y = df["target"]
    DROP_COLS = ["WEEK_NUM", "case_id", "target"]
    X = df.drop(columns=DROP_COLS, errors="ignore").copy()

    
    cat_feats = [c for c in ["month_decision", "weekday_decision"] if c in X.columns]
    if "month_decision" in X:
        mon_dtype  = CategoricalDtype(categories=list(range(1, 13)), ordered=False)
        X["month_decision"] = X["month_decision"].astype("int32").astype(mon_dtype)
    if "weekday_decision" in X:
        wday_dtype = CategoricalDtype(categories=list(range(1, 8)),  ordered=False)
        X["weekday_decision"] = X["weekday_decision"].astype("int32").astype(wday_dtype)



    feats = X.columns.tolist()
    
    folds = make_expanding_folds(df, time_col="WEEK_NUM",
                                 start_week=0, end_week=69,
                                 n_folds=4, val_span=4)



    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "average_precision",   
            "boosting_type": "gbdt",
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 128, 512),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 300, 1500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 8),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
            "seed": SEED,
            "num_threads": -1,
        }

        cv_scores, cv_iters = [], []
        for tr_idx, va_idx in folds:
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
            
                 
            pos = max(1, (y_tr == 1).sum())
            neg = max(1, (y_tr == 0).sum())
            params["scale_pos_weight"] = max(1.0, neg / pos)
   
   
            dtr = lgb.Dataset(X_tr, y_tr, feature_name=feats, categorical_feature=cat_feats)
            dva = lgb.Dataset(X_va, y_va, feature_name=feats, categorical_feature=cat_feats)

            model = lgb.train(
                params,
                dtr,
                valid_sets=[dva],
                valid_names=["val"],
                num_boost_round=3000,
                early_stopping_rounds=80,
                verbose_eval=False,
            )

            y_prob = model.predict(X_va, num_iteration=model.best_iteration)
            pr_auc = average_precision_score(y_va, y_prob)  
            cv_scores.append(pr_auc)
            cv_iters.append(model.best_iteration)

        trial.set_user_attr("best_iterations", cv_iters)
        return float(np.mean(cv_scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def main():
    df = pd.read_parquet(TRAIN_F)

    study = run_study(df, n_trials=100)
    bt = study.best_trial
    best_params = bt.params
    best_iters  = bt.user_attrs["best_iterations"] 

    #  Take the early stopping rounds of the “last fold” and multiply by 1.1 
    if best_iters and len(best_iters) > 0:
        last_iter = int(best_iters[-1])            
        train_rounds = int(math.ceil(last_iter * 1.10))
    else:
        last_iter = 1000
        train_rounds = 1100

    train_rounds = max(100, min(train_rounds, 6000))  
    print(f"CV iters per fold: {best_iters}  → last={last_iter}, final num_boost_round={train_rounds}")

    #  Retrain on full data
    y = df["target"]
    X = df.drop(columns=["target", "WEEK_NUM", "case_id"], errors="ignore").copy()

    cat_feats = [c for c in ["month_decision", "weekday_decision"] if c in X.columns]
    if "month_decision" in X:
        X["month_decision"] = X["month_decision"].astype("int32").astype(
            CategoricalDtype(categories=list(range(1, 13)), ordered=False)
        )
    if "weekday_decision" in X:
        X["weekday_decision"] = X["weekday_decision"].astype("int32").astype(
            CategoricalDtype(categories=list(range(1, 8)), ordered=False)
        )

    dtrain = lgb.Dataset(X, y, feature_name=X.columns.tolist(), categorical_feature=cat_feats)

    final_params = {
        **best_params,
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "scale_pos_weight": max(1.0, (y == 0).sum() / max(1, (y == 1).sum())),
        "seed": SEED,
    }

    final_model = lgb.train(
        final_params,
        dtrain,
        num_boost_round=train_rounds,   
    )

    joblib.dump(
    {
        "params": best_params,
        "best_cv_pr_auc": bt.value,
        "best_iterations": best_iters,
        "last_fold_iter": last_iter,
        "final_num_boost_round": train_rounds,
    },
    OUT_PKL
)
    final_model.save_model(str(OUT_MODEL))

if __name__ == "__main__":
    main()




