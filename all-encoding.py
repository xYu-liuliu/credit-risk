import pandas as pd
from pathlib import Path

class CategoricalEncoder:
    def __init__(self, min_samples=50, trend_window=100, onehot_thr=20, alpha=100, beta=10):
        """
        One-Hot + Time-Window Target Encoding (with dynamic smoothing)
        """
        self.min_samples = min_samples
        self.trend_window = trend_window
        self.onehot_thr = onehot_thr
        self.alpha = alpha
        self.beta = beta
        
        self.low_card_cols_ = []
        self.high_card_cols_ = []
        self.ohe_cols_ = None 
        self.global_mean_ = None
        self.running_stats_ = {}

    def _time_window_encode(self, df, col, target, time_col, init_stats, allow_current=True):
        """
        Time-window Target Encoding with dynamic smoothing.
        """
        df = df.copy()
        # Sort by week + weekday_decision
        df = df.sort_values([time_col, "weekday_decision"]).reset_index(drop=True) 
        encoded, trend_encoded = [], []
        running_stats = {k: [v[0], v[1], v[2][:]] for k, v in init_stats.items()}

        for _, row in df.iterrows():
            cat = row[col]

            if cat in running_stats and running_stats[cat][1] >= self.min_samples:
                s, c, recent = running_stats[cat]
                w = self.alpha / (c + self.beta)  # Dynamic weight

                if allow_current and not pd.isna(row[target]):
                    s_tmp, c_tmp = s + row[target], c + 1
                    mean = (s_tmp + self.global_mean_ * w) / (c_tmp + w)
                else:
                    mean = (s + self.global_mean_ * w) / (c + w)

                encoded.append(mean)

                if recent:
                    recent_mean = sum(recent[-self.trend_window:]) / min(len(recent), self.trend_window)
                    trend_encoded.append(recent_mean - mean)
                else:
                    trend_encoded.append(0)
            else:
                encoded.append(self.global_mean_)
                trend_encoded.append(0)

            # Update running stats
            if not pd.isna(row[target]):
                if cat not in running_stats:
                    running_stats[cat] = [row[target], 1, [row[target]]]
                else:
                    running_stats[cat][0] += row[target]
                    running_stats[cat][1] += 1
                    running_stats[cat][2].append(row[target])

        df[col] = encoded
        df[col + "_trend"] = trend_encoded
        return df, running_stats

    def fit(self, df: pd.DataFrame, target_col: str, time_col="WEEK_NUM"):
        """
        Fit the encoder on the training set.
        """
        df = df.copy()
        self.global_mean_ = df[target_col].mean()

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        nunique_dict = df[cat_cols].nunique()

        self.low_card_cols_ = [c for c in nunique_dict.index if nunique_dict[c] <= self.onehot_thr and c != time_col]
        self.high_card_cols_ = [c for c in nunique_dict.index if nunique_dict[c] > self.onehot_thr and c != time_col]
        
        print(f"✅ OneHot columns: {self.low_card_cols_}")
        print(f"✅ Target Encoding columns: {self.high_card_cols_}")

        self.running_stats_ = {}
        for col in self.high_card_cols_:
            _, stats = self._time_window_encode(df, col, target_col, time_col, {}, allow_current=True)
            self.running_stats_[col] = stats

        return self

    def transform(self, df: pd.DataFrame, target_col: str, time_col="WEEK_NUM", use_val=False):
        """
        Transform the dataset using fitted statistics.
        """
        df = df.copy()

        # One-Hot Encoding
        if self.low_card_cols_:
            df = pd.get_dummies(df, columns=self.low_card_cols_, drop_first=True)
            
            # record the complete dummy set 
            if self.ohe_cols_ is None:
                self.ohe_cols_ = df.columns.tolist()
 
            # add missing dummy cols 
            missing = [c for c in self.ohe_cols_ if c not in df.columns]
            for m in missing:
                df[m] = 0

        df = df[self.ohe_cols_]

        for col in self.high_card_cols_:
            init_stats = self.running_stats_[col]
            df, new_stats = self._time_window_encode(
                df, col, target_col, time_col,
                init_stats, allow_current=False
            )
            if use_val:  
                self.running_stats_[col] = new_stats

        return df







if __name__ == "__main__":
    # load
    IN_DIR  = Path("data/processed") 
    OUT_DIR = Path("data/processed") 
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_file = IN_DIR / "df_train_cut_0_69_processed.parquet"
    val_file   = IN_DIR / "df_val_cut_70_74_processed.parquet"
    test_file  = IN_DIR / "df_test_cut_75_91_processed.parquet"

    print("📥 Loading datasets...")
    df_train = pd.read_parquet(train_file)
    df_val   = pd.read_parquet(val_file)
    df_test  = pd.read_parquet(test_file)
    
    
    df_train["target"] = df_train["target"].astype("Int64")  
    df_val["target"] = df_val["target"].astype("Int64")
    df_test["target"] = df_test["target"].astype("Int64")



    # encoder
    encoder = CategoricalEncoder(
        min_samples=50, 
        trend_window=100, 
        onehot_thr=20, 
        alpha=100, 
        beta=10
    )

    # train
    print("🚀 Fitting encoder on training set...")
    df_train_enc = encoder.fit(df_train, target_col="target").transform(df_train, target_col="target")

    # validation(update)
    print("🔍 Encoding validation set...")
    df_val_enc = encoder.transform(df_val, target_col="target", use_val=True)

    # test 
    print("🧪 Encoding test set...")
    df_test_enc = encoder.transform(df_test, target_col="target", use_val=True)

    # Save 
    df_train_enc["target"] =df_train_enc["target"].astype("Int64")  
    df_train_enc.to_parquet(OUT_DIR / "df_train_encoded.parquet", engine="pyarrow", index=False, compression="zstd")
    df_val_enc["target"] = df_val_enc["target"].astype("Int64")
    df_val_enc.to_parquet(OUT_DIR / "df_val_encoded.parquet", engine="pyarrow", index=False, compression="zstd")
    df_test_enc["target"] = df_test_enc["target"].astype("int64")
    df_test_enc.to_parquet(OUT_DIR / "df_test_encoded.parquet", engine="pyarrow", index=False, compression="zstd")

    print("✅ Encoding complete! Files saved in:", OUT_DIR)
    


