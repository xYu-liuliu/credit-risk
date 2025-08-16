import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from pathlib import Path
from kneed import KneeLocator
from collections import Counter

# load 
PATH = Path(r"E:/Home Credit Processed Feature/train_corr.parquet")
df   = pd.read_parquet(PATH, dtype_backend="pyarrow")   
df["target"] = df["target"].astype("Int64")

df = df.convert_dtypes(dtype_backend="numpy_nullable")  

# transform
bool_cols = df.select_dtypes("boolean").columns         
df[bool_cols] = df[bool_cols].astype("int8")            

hf_cols = df.select_dtypes("float16").columns          
df[hf_cols] = df[hf_cols].astype("float32")             

f64_cols = df.select_dtypes("float64").columns          
df[f64_cols] = df[f64_cols].astype("float32")          

df["WEEK_NUM"] = (
    pd.to_numeric(df["WEEK_NUM"], errors="coerce")  
      .fillna(0)                                     
      .astype("int16")                               
)

arrow_cols = df.columns[df.dtypes.astype(str).str.contains(r"\[pyarrow\]")]

for col in arrow_cols:
    kind = str(df[col].dtype)              
    if   kind.startswith("float"):
        df[col] = df[col].astype("float32") 
    elif kind.startswith("int8"):
        df[col] = df[col].astype("int8")
    elif kind.startswith("int16"):
        df[col] = df[col].astype("int16")
    elif kind.startswith("int32"):
        df[col] = df[col].astype("int32")
    else:
        raise ValueError(f"Unhandled Arrow dtype: {kind}")
 
int_ext = df.select_dtypes(include=["Int64"]).columns
if len(int_ext):
   df[int_ext] = df[int_ext].astype("int64")

                                                                                               
print("✓ dtype conversion done:")
print(df.dtypes.value_counts())
    
y  = df.pop("target")
DROP_COLS = ["WEEK_NUM", "case_id"]
X = df.drop(columns=DROP_COLS, errors="ignore").copy()


dtype_cnt = X.dtypes.value_counts()
print("\n>>> Column count by dtype")
print(dtype_cnt)

    
    
    
is_zero_based = (df.index[0] == 0)
is_contiguous = (df.index.is_monotonic_increasing and df.index.to_series().diff().fillna(1).eq(1).all())
print(f"index starts at 0: {is_zero_based},  contiguous: {is_contiguous}")

# quick LightGBM for Gain importance 
neg, pos = Counter(y)[0], Counter(y)[1]
scale = neg / pos
print(f"Positive:Negative = 1:{scale:.1f}")

params = dict(
    objective   = "binary",
    metric      = "auc",
    learning_rate = 0.02,
    num_leaves    = 256,
    max_depth     = 10,
    min_data_in_leaf = 25,
    feature_fraction = 0.9,
    bagging_fraction = 0.8,
    bagging_freq     = 1,
    scale_pos_weight = scale,   
    seed  = 42,
    verbosity = -1,
)

dtrain = lgb.Dataset(X, y)
model  = lgb.train(
    params, dtrain,
    num_boost_round = 1000,
)

gain = model.feature_importance(importance_type="gain")
gain_df = (pd.DataFrame({"feature": X.columns, "gain": gain})
           .sort_values("gain", ascending=False)
           .reset_index(drop=True))

# elbow detection (first gain < 1 %) 
gain_df["gain_pct"]     = gain_df["gain"] / gain_df["gain"].sum()
gain_df["cum_gain_pct"] = gain_df["gain_pct"].cumsum()
elbow_idx = gain_df[gain_df["gain_pct"] < 0.01].index[0]  

# plot
plt.figure(figsize=(8, 4))
plt.plot(gain_df.index + 1, gain_df["gain"], marker=".", linewidth=1)
plt.yscale("log")
plt.axvline(elbow_idx + 1, color="red", ls="--",
            label=f"elbow ≈ Top-{elbow_idx+1}")
plt.title("Gain-importance curve")
plt.xlabel("Feature rank (Top-N)")
plt.ylabel("Gain importance (log scale)")
plt.legend()
plt.tight_layout()
plt.show()


# top 95%
k95 = gain_df[gain_df["cum_gain_pct"] >= 0.95].index[0] + 1
print("Top-K (95% cumulative):", k95)

# Kneedle 
knee = KneeLocator(
    x=gain_df.index + 1,
    y=gain_df["gain"],
    S=1.0, curve="convex", direction="decreasing"
).knee
print("Kneedle knee point:", knee)


# save
gain_df.to_csv("gain_importance.csv", index=False)
print(f"✅ gain_importance.csv saved — elbow at roughly Top-{elbow_idx+1}")
