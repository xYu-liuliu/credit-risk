from pathlib import Path
import pandas as pd
import joblib
import numpy as np


OUT_DIR = Path("E:/Home Credit Processed Feature")
INPUT_PATH = Path(r"E:/all_data.parquet")
df = pd.read_parquet(INPUT_PATH, engine="pyarrow")




KEY = ("WEEK_NUM", "weekday_decision")
TARGET = "target"

def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """ convert dtype"""
    df = df.copy()

    # Convert "WEEK_NUM", "weekday_decision" to int32
    for c in KEY:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise").astype("int32")

    # Convert "target" to int64
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype("int64")

    # Convert all intergers below int32 to int32
    small_int_cols = [
        c for c, dt in df.dtypes.items()
        if any(tok in str(dt).lower() for tok in ("int8", "uint8", "int16", "uint16"))
    ]
    if small_int_cols:
        df[small_int_cols] = df[small_int_cols].astype("int32")

    # Convert float16 to float32
    f16 = df.select_dtypes(include=["float16"]).columns
    if len(f16):
        df[f16] = df[f16].astype("float32")

    # Sort by time
    if set(KEY) <= set(df.columns):
        df = df.sort_values(list(KEY), kind="mergesort").reset_index(drop=True)

    return df

df_sorted = sanitize(df)


# Split 

TRAIN_END = 69  
VAL_END   = 74   
TEST_BEG  = 75   

train_df = df_sorted[df_sorted["WEEK_NUM"] <= TRAIN_END]
val_df   = df_sorted[(df_sorted["WEEK_NUM"] > TRAIN_END) & (df_sorted["WEEK_NUM"] <= VAL_END)]
test_df  = df_sorted[df_sorted["WEEK_NUM"] >= TEST_BEG]


is_time_sorted = train_df.sort_values(["WEEK_NUM","weekday_decision"]).index.equals(train_df.index)

is_time_sorted = val_df.sort_values(["WEEK_NUM","weekday_decision"]).index.equals(val_df.index)

is_time_sorted = test_df.sort_values(["WEEK_NUM","weekday_decision"]).index.equals(test_df.index)


train_df.to_parquet(OUT_DIR / "df_train_cut_0_69_ADDCOMBO.parquet", engine="pyarrow", index=False, use_dictionary=False)
val_df  .to_parquet(OUT_DIR / "df_val_cut_70_74_ADDCOMBO.parquet", engine="pyarrow",  index=False, use_dictionary=False)
test_df["target"] = test_df["target"].astype("int64")
test_df .to_parquet(OUT_DIR / "df_test_cut_75_91_ADDCOMBO.parquet", engine="pyarrow", index=False, use_dictionary=False)




