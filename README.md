credit-risk-modeling/
│
├─ data/
│   ├─ raw/               # Kaggle 原始 parquet
│   ├─ aggregation/       # Stage 1 输出 (all_data.parquet)
│   ├─ splits/            # Stage 2 输出 (*_ADDCOMBO.parquet)
│   ├─ processed/         # Stage 3 输出 (*_processed.parquet)
│   ├─ feature_selection/ # Stage 4 输出 (*_sel.parquet, importance files)
│   └─ model/             # Stage 5 输出 (LightGBM 模型、最佳参数 pkl)
│
├─ fill.py
├─ Combo_feature.py
├─ aggregationandmerge_pipeline.py   # Stage 1
├─ split_data.py                     # Stage 2
├─ data processing.py                # Stage 3
├─ all-encoding.py                   # Stage 4
├─ feature-selection pipeline.py     # Stage 4
├─ time cv lgbm.py                   # Stage 5
└─ README.md
