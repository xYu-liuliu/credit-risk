```text
credit-risk-modeling/
│
├─ data/
│    ├─ raw/                # Original Kaggle parquet files  
│    ├─ aggregation/        # Stage 1 output (all_data.parquet)  
│    ├─ splits/             # Stage 2 output (*_ADDCOMBO.parquet)  
│    ├─ processed/          # Stage 3 output (*_processed.parquet)  
│    ├─ feature_selection/  # Stage 4 outputs (encoded parquet, correlation/IV/SHAP/Gain)  
│    ├─ model/              # Stage 5 trained models  
│    ├─ prediction/         # Stage 6 predictions for analysis  
│    └─ analysis/           # Stage 7 analysis results  
│         ├─ model/         # Model performance metrics  
│         ├─ business/      # Business evaluation results  
│         └─ feature/       # Feature interpretation & stability  
│  
├─ fill.py                           # Utility: missing value filling rules  
├─ Combo_feature.py                  # Utility: row-level combo features  
├─ aggregationandmerge_pipeline.py   # Stage 1: Aggregate & merge tables  
├─ split_data.py                     # Stage 2: Split dataset into train/val/test  
├─ data processing.py                # Stage 3: Data cleaning & preprocessing  
├─ all-encoding.py                   # Stage 3: Encoding  
├─ correlation.py                    # Stage 4: Feature correlation analysis  
├─ iv.py                             # Stage 4: Information Value (IV) calculation (not used in this project)  
├─ find SHAP point.py                # Stage 4: SHAP importance exploration  
├─ find gain importance point.py     # Stage 4: Gain importance exploration  
├─ feature-selection pipeline.py     # Stage 4: Final feature selection pipeline  
├─ time cv lgbm.py                   # Stage 5: Model training (time-series CV, LightGBM)  
├─ predict.py                        # Stage 6: Run prediction for analysis  
├─ model performance.py              # Stage 7: Model performance metrics (ROC, PR, KS, etc.)  
├─ business evaluation.py            # Stage 7: Business evaluation (PR@K, Lift, Bad Rate)  
├─ feature performance.py            # Stage 7: Feature performance (IV, KS, PSI, Top10 stability)  
├─ feature analysis.py               # Stage 7: Feature interpretation (SHAP dependence, weekly SHAP)  
├─ README.md                         # intro  
└─ credit risk.pdf                   # Full project report  

