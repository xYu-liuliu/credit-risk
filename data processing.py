import pandas as pd
import numpy as np
from pathlib import Path

class RiskDataPipeline:
    def __init__(self, nan_thr=0.97, high_card=5000, 
                 upper_q=0.998):
        self.nan_thr = nan_thr
        self.high_card = high_card
        self.upper_q = upper_q

        self.drop_cols_ = None
        self.quantiles_ = {}


    def fit(self, df: pd.DataFrame):
        df = df.copy()

        # Hard filter
        rm = []
        for c in df.columns:
            if df[c].isna().mean() > self.nan_thr or df[c].nunique(dropna=True) <= 1:
                rm.append(c); continue
            if df[c].dtype == "object" and df[c].nunique(dropna=True) > self.high_card:
                rm.append(c)
        self.drop_cols_ = rm
        print(f"[HardFilter] Drop {len(rm)} cols")
    
        df = df.drop(columns=self.drop_cols_, errors="ignore")

        # Soft impute
        df = self.soft_impute(df)

        # Add combos
        df = self.add_combos(df)

        # Winsorize quantiles
        self.quantiles_ = {}
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                # flag
                uniq_vals = set(df[c].dropna().unique())
                if uniq_vals <= {0,1}:
                    continue

                if c.lower().endswith(("p","a")):
                    q = df[c].quantile([self.upper_q]).to_dict()
                    self.quantiles_[c] = {"type": "PA", "high": q[self.upper_q]}

                elif c.lower().endswith("o"):
                    q_val = df[c].quantile(self.upper_q)
                    self.quantiles_[c] = {"type": "O", "high": max(1.0, q_val)}

        print(f"[Winsorize] Stored quantiles for {len(self.quantiles_)} cols")
        return self


    def transform(self, df: pd.DataFrame):
        df = df.copy()

        # Hard filter
        df = df.drop(columns=self.drop_cols_, errors="ignore")

        # Soft impute
        df = self.soft_impute(df)

        # Add combos
        df = self.add_combos(df)

        # ④ Winsorize with stored quantiles
        for c, info in self.quantiles_.items():
            if c in df.columns:
                if info["type"] == "PA":
                    df[c] = df[c].clip(upper=info["high"])
                elif info["type"] == "O":
                    df[c] = df[c].clip(lower=0, upper=info["high"])
    
        return df


    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    def soft_impute(self,df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if "_" not in col:
                continue
            suffix = col.split("_")[-1][-1]
            is_num = pd.api.types.is_numeric_dtype(df[col])

            if suffix in {"A", "P", "D", "o",  "g"}:  
                df[col].fillna(0, inplace=True)

            elif suffix == "M":  
                if "zip" in col.lower() and is_num:
                    df[col].fillna(-1, inplace=True)
                else:
                    if pd.api.types.is_categorical_dtype(df[col]):
                        if "missing" not in df[col].cat.categories:
                            df[col] = df[col].cat.add_categories(["missing"])
                        df[col].fillna("missing", inplace=True)
                    else:
                        df[col] = df[col].astype(str).fillna("missing")

            elif suffix in {"T", "L"}:  
                if is_num:
                    df[col].fillna(0, inplace=True)
                else:
                    if pd.api.types.is_categorical_dtype(df[col]):
                        if "UNKNOWN" not in df[col].cat.categories:
                            df[col] = df[col].cat.add_categories(["UNKNOWN"])
                        df[col].fillna("UNKNOWN", inplace=True)
                    else:
                        df[col] = df[col].astype(str).fillna("UNKNOWN")

        return df

    def add_combos(self,df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        
        # High utilization + severe delinquency (dpd >= 60) indicator
        out["highutil_and_dpd60_ratio"] = (
            (out["last_cc_burden_ratio"] > 0.8) &
            (out["last_actualdpd_gt60_flag"] == 1)
        ).astype("int8")

        # Trend / relief
        out["dpd_worsen_delta_P"] = out["last_actualdpd_943P"] - out["mean_actualdpd_943P"]
        out["cc_burden_relief_ratio"] = 1 - (out["last_cc_burden_ratio"] /
                                       (out["max_cc_burden_ratio"] + 1e-3))

        
        # Debt pressure
        out["dti_ex_ratio"] = out["last_annuity_853A"] / (out["last_mainoccupationinc_437A"] + 1e-3)
        out["arrears_progress_P"] = (
            1 - out["repayment_progress_ratio"]
        ) * out["last_actualdpd_943P"]

        # Residual debt
        out["residual_amt_flag"] = (
            out["max_residualamount_856A"] * out["max_residual_closed_flag"]
        )

        # Credit limit surge
        out["credlmt_delta_A"] = out["last_credlmt_230A"] - out["mean_credlmt_230A"]

        # Missingness indicators for key ratios
        for c in ["payment_ability_ratio", "dti_ratio"]:
            out[c + "_miss"] = (out[c] == 0).astype("int8")

        return out



# load
IN_DIR  = Path(r"E:/Home Credit Processed Feature")
OUT_DIR = Path(r"E:/Home Credit Processed Feature")
OUT_DIR.mkdir(parents=True, exist_ok=True)

train_file = IN_DIR / "df_train_cut_0_69_ADDCOMBO.parquet"
val_file   = IN_DIR / "df_val_cut_70_74_ADDCOMBO.parquet"
test_file  = IN_DIR / "df_test_cut_75_91_ADDCOMBO.parquet"

df_train = pd.read_parquet(train_file)
df_val   = pd.read_parquet(val_file)
df_test  = pd.read_parquet(test_file)


# Preserve "target" column
TARGET = "target"                       
y_train = df_train.pop(TARGET)
y_val   = df_val.pop(TARGET)
y_test  = df_test.pop(TARGET)


# Run Pipeline 
pipeline = RiskDataPipeline()

# fit on trainset
df_train_proc = pipeline.fit_transform(df_train)

# transfrom on validationset and testset
df_val_proc   = pipeline.transform(df_val)
df_test_proc  = pipeline.transform(df_test)


#  Add "target" column back
df_train_proc[TARGET] = y_train
df_val_proc[TARGET]   = y_val
df_test_proc[TARGET]  = y_test


# save
df_train_proc.to_parquet(OUT_DIR / "df_train_cut_0_69_processed.parquet", engine="pyarrow", index=False, compression="zstd", use_dictionary=False)
df_val_proc.to_parquet(OUT_DIR / "df_val_cut_70_74_processed.parquet", engine="pyarrow", index=False, compression="zstd", use_dictionary=False)
df_test_proc.to_parquet(OUT_DIR / "df_test_cut_75_91_processed.parquet", engine="pyarrow", index=False, compression="zstd", use_dictionary=False)




