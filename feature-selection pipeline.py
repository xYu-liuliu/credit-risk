from __future__ import annotations
from pathlib import Path
from typing  import Sequence

import numpy  as np
import pandas as pd
import lightgbm as lgb
import shap
from collections import Counter

from pandas.api.types import CategoricalDtype



class GainShapSelector:

    def __init__(self,
                 k_gain: int = 276,
                 k_shap: int = 201,
                 lgb_params: dict | None = None):
        self.k_gain  = k_gain
        self.k_shap  = k_shap
        self.lgb_params = lgb_params or {}
        self.selected_: list[str] = []
        self.model_:    lgb.Booster | None = None


    def _train_model(self, X: pd.DataFrame, y: pd.Series, cat_feats: list[str]) -> lgb.Booster:
        """LightGBM quick model for importance extraction."""
        neg, pos = Counter(y)[0], Counter(y)[1]
        scale = neg / pos
        default_params = dict(
            objective        = "binary",
            metric           = "auc",
            learning_rate    = 0.02,
            num_leaves       = 256,
            max_depth        = 10,
            min_data_in_leaf = 25,
            feature_fraction = 0.9,
            bagging_fraction = 0.8,
            bagging_freq     = 1,
            scale_pos_weight = scale,
            seed             = 42,
            verbosity        = -1,
        )
        default_params.update(self.lgb_params)
        dtrain = lgb.Dataset(X, y, categorical_feature=cat_feats)
        return lgb.train(default_params, dtrain, num_boost_round=1000)


    def fit(self, df_train: pd.DataFrame, target_col: str = "target") -> "GainShapSelector":
        """ Fit on training data  → build Gain-Top-K + SHAP-Top-K feature list """
        
        df = df_train.copy()      
        mon_dtype  = CategoricalDtype(categories=list(range(1, 13)), ordered=False)
        wday_dtype = CategoricalDtype(categories=list(range(1, 8)),  ordered=False)
        if "month_decision" in df:
            df["month_decision"] = df["month_decision"].astype("int32").astype(mon_dtype)
        if "weekday_decision" in df:
            df["weekday_decision"] = df["weekday_decision"].astype("int32").astype(wday_dtype)
                     
        y  = df[target_col]                              
        DROP_COLS = ["WEEK_NUM", "case_id", target_col]
        X = df.drop(columns=DROP_COLS, errors="ignore")
        
        cat_feats = [c for c in ["month_decision", "weekday_decision"] if c in X.columns]

        self.model_ = self._train_model(X, y, cat_feats)

        # Gain Top-K
        gain_imp = self.model_.feature_importance("gain")
        gain_df  = (pd.DataFrame({"f": self.model_.feature_name(),
                                  "g": gain_imp})
                    .sort_values("g", ascending=False)
                    .head(self.k_gain))
        gain_set = set(gain_df["f"])

        # SHAP Top-K
        explainer = shap.TreeExplainer(self.model_)
        raw       = explainer.shap_values(X, check_additivity=False)
        shap_vals = raw[1] if isinstance(raw, list) else raw
        shap_mean = np.abs(shap_vals).mean(axis=0)
        shap_df   = (pd.DataFrame({"f": X.columns, "s": shap_mean})
                     .sort_values("s", ascending=False)
                     .head(self.k_shap))
        
        print("\nTop-10 features by mean |SHAP|:")
        print(shap_df.head(10))      
        shap_set  = set(shap_df["f"])

        self.selected_ = sorted(gain_set | shap_set)
        print(f"Gain top-{self.k_gain}: {len(gain_set)}, "
              f"SHAP top-{self.k_shap}: {len(shap_set)}, "
              f"union → {len(self.selected_)} features")

        return self


    def transform(self,
                  df: pd.DataFrame,
                  keep_extra: Sequence[str] = ("target", "WEEK_NUM")
                 ) -> pd.DataFrame:
        """ output keep target, WEEK_NUM + selected union """
        assert self.selected_, "fit() must be called first."
        df = df.copy()
        cols = [c for c in keep_extra if c in df.columns] + self.selected_
        return df.loc[:, cols]


    def fit_transform(self, df_train: pd.DataFrame,
                      target_col: str = "target",
                      keep_extra: Sequence[str] = ("target", "WEEK_NUM")
                     ) -> pd.DataFrame:
        return self.fit(df_train, target_col).transform(df_train, keep_extra)



def main():
    data_dir = Path(r"E:/Home Credit Processed Feature")
    train_f, val_f, test_f = [data_dir / f for f in
        ("train_corr.parquet", "val_corr.parquet", "test_corr.parquet")]

    train_df = pd.read_parquet(train_f)

    selector = GainShapSelector(k_gain=276, k_shap=201)
    train_sel = selector.fit_transform(train_df)

    # save train
    out_dir = Path(r"E:/Home Credit Processed Feature")
    
    train_sel.to_parquet(out_dir / "train_sel.parquet", compression="zstd", use_dictionary=False)

    # process val / test
    for name, path in [("val",  val_f), ("test", test_f)]:
        df_in = pd.read_parquet(path)
        
        df_out = selector.transform(df_in)
        
        df_out.to_parquet(out_dir / f"{name}_sel.parquet", compression="zstd", use_dictionary=False)
        print(f"{name} saved:", df_out.shape)

    print("Done.  Files in:", out_dir)


if __name__ == "__main__":
    main()
