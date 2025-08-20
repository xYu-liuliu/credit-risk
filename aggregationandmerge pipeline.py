import sys
import subprocess
import os
import gc
from pathlib import Path
from glob import glob
import random
import numpy as np
import pandas as pd
import polars as pl

from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

import joblib

import warnings
warnings.filterwarnings('ignore')
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedGroupKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from Combo_feature import _optional_rules, build_combo_exprs
from fill import Filler



class Pipeline:

    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    def handle_dates(df: pl.LazyFrame | pl.DataFrame):
        # Find all columns that end with _D
        date_cols = [c for c in df.columns if c.endswith("D")]

        # Use list comprehension to build pl.Expr
        exprs = [
        (pl.col(c) - pl.col("date_decision"))
        .dt.total_days()
        .alias(c)
        for c in date_cols
    ]

        # Drop extra columns
        return (
        df.with_columns(exprs)
          .drop("date_decision", "MONTH")
    )

    def filter_cols(df):
        
        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if (freq == 1) | (freq > 200):
                    df = df.drop(col)
        
        return df
    
    
class Aggregator:
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        expr_max    = [pl.max(col).alias(f"max_{col}")    for col in cols]
        expr_last   = [pl.last(col).alias(f"last_{col}")   for col in cols]
        expr_mean   = [pl.mean(col).alias(f"mean_{col}")   for col in cols]
        return expr_max + expr_last + expr_mean

    def date_expr(df):
        cols = [col for col in df.columns if col[-1] == "D"]
        expr_max    = [pl.max(col).alias(f"max_{col}")    for col in cols]
        expr_last   = [pl.last(col).alias(f"last_{col}")   for col in cols]
        expr_mean   = [pl.mean(col).alias(f"mean_{col}")   for col in cols]
        return expr_max + expr_last + expr_mean

    def str_expr(df):
        cols = [col for col in df.columns if col[-1] == "M"]
        expr_max  = [pl.max(col).alias(f"max_{col}")  for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        return expr_max + expr_last

    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max  = [pl.max(col).alias(f"max_{col}")  for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        return expr_max + expr_last

    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]
        expr_max  = [pl.max(col).alias(f"max_{col}")  for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        return expr_max + expr_last

    def keyword_expr(df):
        exprs = []
        for col in df.columns:
            if col.endswith(("flag", "ratio", "delta")):
                exprs.extend([
                pl.mean(col).alias(f"mean_{col}"),
                pl.max(col).alias(f"max_{col}"),
                pl.last(col).alias(f"last_{col}"),
            ])
        return exprs

    def get_exprs(df):
        return (
            Aggregator.num_expr(df)
          + Aggregator.date_expr(df)
          + Aggregator.str_expr(df)
          + Aggregator.other_expr(df)
          + Aggregator.count_expr(df)
          + Aggregator.keyword_expr(df)
        )
    
    
def read_file(path, depth=None):
    df = (
        pl.scan_parquet(path, low_memory=True)
          .pipe(Pipeline.set_table_dtypes) 
          .pipe(Filler.fill_by_suffix)          
    )

    # Add combo feature
    combo_exprs = build_combo_exprs(df, _optional_rules)
    if combo_exprs:                            
        df = df.with_columns(combo_exprs)
    if depth in [1,2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df)) 
    return df


def read_files(pattern, depth=None):
    lf = (
        pl.scan_parquet(str(pattern), low_memory=True)      
          .pipe(Pipeline.set_table_dtypes) 
          .pipe(Filler.fill_by_suffix)                      
    )

    # Add combo feature
    combo_exprs = build_combo_exprs(lf, _optional_rules)
    if combo_exprs:
        lf = lf.with_columns(combo_exprs)

    # The same kind of tables do one aggregation 
    if depth in (1, 2):
        lf = lf.group_by("case_id").agg(Aggregator.get_exprs(lf))

    # unique
    lf = lf.unique(subset=["case_id"])

    return lf    



def feature_eng(df_base, depth_0, depth_1, depth_2):
    # Add date feature
    lf = df_base.with_columns(
        month_decision   = pl.col("date_decision").dt.month(),
        weekday_decision = pl.col("date_decision").dt.weekday(),
    )

    # Join sub-table in sequence
    for lf_new in depth_0 + depth_1 + depth_2:
        # Left join and  add suffix "_new" to conflicting columns from right table
        joined = lf.join(lf_new, how="left", on="case_id", suffix="_new")

        # caleasce conflicting columns
        dup = [c for c in lf.columns if f"{c}_new" in joined.columns]
        coalesced = [pl.coalesce([pl.col(c), pl.col(f"{c}_new")]).alias(c)for c in dup]

        # Merge columns back and drop columns with "_new"
        lf = (
            joined
            .with_columns(*coalesced)
            .drop([f"{c}_new" for c in dup])
        )

    # Compute date difference
    return lf.pipe(Pipeline.handle_dates)


def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)=="category":
            continue
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            continue
    end_mem = df.memory_usage().sum() / 1024**2    
    return df

def reduce_group(grps):
    use = []
    for g in grps:
        mx = 0; vx = g[0]
        for gg in g:
            n = df_train[gg].nunique()
            if n>mx:
                mx = n
                vx = gg
        use.append(vx)
    return use






TRAIN_DIR = Path("data/raw")

data_store = {
    "df_base": read_file(TRAIN_DIR / "train_base.parquet"),
    "depth_0": [
        read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
        read_files(TRAIN_DIR / "train_static_0_*.parquet"),
    ],
    "depth_1": [
        read_files(TRAIN_DIR / "train_applprev_1_*.parquet", depth=1),
        read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1),
        read_files(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet", depth=1),
        read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1),
        read_file(TRAIN_DIR / "train_other_1.parquet", 1),
        read_file(TRAIN_DIR / "train_person_1.parquet", 1),
        read_file(TRAIN_DIR / "train_deposit_1.parquet", 1),
        read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1),
    ],
    "depth_2": [
        read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2),
    ]
}



# Filter
df_train = feature_eng(**data_store).collect(streaming=True)
del data_store; gc.collect()

df_train = Pipeline.filter_cols(df_train)
print("Columns after filter_cols:", df_train.columns)

combo_names = set(_optional_rules.keys())        
present_combo = [c for c in df_train.columns if c in combo_names]
print("Combo features in df_train:", present_combo)


# Convert to Pandas
df_train, cat_cols = to_pandas(df_train)
df_train = reduce_mem_usage(df_train)

OUT_DIR = Path("data/processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Write Parquet
df_train.to_parquet(
    f"{OUT_DIR}/all_data.parquet",
    index=False,
    engine="pyarrow",   
    compression="zstd"  
)











