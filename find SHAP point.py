from collections import Counter
import lightgbm as lgb
import shap, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path
from pandas.api.types import CategoricalDtype


# load 
PATH = Path(r"E:/Home Credit Processed Feature/train_corr.parquet")
df   = pd.read_parquet(PATH)   


    
mon_dtype  = CategoricalDtype(categories=list(range(1, 13)), ordered=False)
wday_dtype = CategoricalDtype(categories=list(range(1, 8)),  ordered=False)

NON_FEATURES = {"target","WEEK_NUM","case_id"}
feat_cols = [c for c in df.columns if c not in NON_FEATURES]
cat_feats = [c for c in ["month_decision","weekday_decision"] if c in feat_cols]
num_cols  = [c for c in feat_cols if c not in cat_feats]
    

y  = df["target"]
X = df[feat_cols].copy()

    



neg, pos = Counter(y)[0], Counter(y)[1]
scale = neg / pos
print(f"Positive:Negative = 1:{scale:.1f}")

params = dict(
    objective   = "binary",
    metric      = "average_precision",
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

dtrain = lgb.Dataset(X, y, categorical_feature=cat_feats)
model  = lgb.train(
    params, dtrain,
    num_boost_round = 1000,
)
y_pred = model.predict(X)
print("train AUC:", roc_auc_score(y, y_pred))

# calculate  SHAP
import shap, numpy as np, pandas as pd
explainer = shap.TreeExplainer(model)
raw       = explainer.shap_values(X, check_additivity=False)


if isinstance(raw, list):          
    shap_values = raw[1]           
else:                              
    shap_values = raw              

print("shape:", shap_values.shape) 

shap_mean   = np.abs(shap_values).mean(axis=0)

shap_df = (pd.DataFrame({"feature": X.columns, "shap": shap_mean})
           .sort_values("shap", ascending=False)
           .reset_index(drop=True))

print("Top-10 features by mean |SHAP|")
print(shap_df.head(10))


shap_df.to_csv("E:/Home Credit Processed Feature/shap_importance_full.csv", index=False)

# plot
import matplotlib.pyplot as plt

shap_df["shap_pct"]     = shap_df["shap"] / shap_df["shap"].sum()
shap_df["cum_shap_pct"] = shap_df["shap_pct"].cumsum()

k90 = shap_df[shap_df["cum_shap_pct"] >= 0.90].index[0] + 1
print(f"Top-{k90} features explain 90 % of total |SHAP|.")

fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(shap_df.index + 1, shap_df["shap"], marker=".", lw=1)
ax1.set_yscale("log")
ax1.set_xlabel("Feature rank (Top-N)")
ax1.set_ylabel("mean |SHAP|  (log scale)")
ax1.set_title("SHAP importance curve")

ax2 = ax1.twinx()
ax2.plot(shap_df.index + 1, shap_df["cum_shap_pct"],
         color="orange", lw=1, label="Cumulative share")
ax2.set_ylabel("Cumulative share")
ax2.set_ylim(0, 1.05)

ax1.axvline(k90, color="red", ls="--", label=f"90 % @ Top-{k90}")
ax1.legend(loc="upper right")
plt.tight_layout(); plt.show()


fig.savefig("shap_importance_curve.png", dpi=300)
