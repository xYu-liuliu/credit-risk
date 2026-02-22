## Repository Structure

```text
credit-risk-modeling/
│
├─ data/
│  ├─ raw/                 # Original Kaggle parquet files
│  ├─ aggregation/         # Stage 1 output (all_data.parquet)
│  ├─ splits/              # Stage 2 output (*_ADDCOMBO.parquet)
│  ├─ processed/           # Stage 3 output (*_processed.parquet, *_encoded.parquet)
│  ├─ feature_selection/   # Stage 4 outputs (correlation/IV/SHAP/Gain)
│  ├─ model/               # Stage 5 trained models
│  ├─ prediction/          # Stage 6 predictions for analysis
│  └─ analysis/            # Stage 7–8 analysis results
│     ├─ model/            # Stage 7 Model performance metrics
│     ├─ business/         # Stage 7 Business evaluation results
│     ├─ feature/          # Stage 7 Feature interpretation & stability
│     └─ calibration/    # Stage 8 Probability calibration results
│     └─ conformal/      # Stage 8 Conformal prediction results
│
├─ fill.py                           # Utility: missing value filling rules
├─ Combo_feature.py                  # Utility: row-level combo features
│
├─ aggregationandmerge_pipeline.py   # Stage 1: Aggregate & merge tables
├─ split_data.py                     # Stage 2: Time-based split
├─ data processing.py                # Stage 3: Cleaning & preprocessing
├─ all-encoding.py                   # Stage 3: Encoding
│
├─ correlation.py                    # Stage 4: Correlation analysis
├─ iv.py                             # Stage 4: IV calculation (not used)
├─ find SHAP point.py                # Stage 4: SHAP exploration
├─ find gain importance point.py     # Stage 4: Gain importance exploration
├─ feature-selection pipeline.py     # Stage 4: Final feature selection
│
├─ time cv lgbm.py                   # Stage 5: Model training (time-series CV, LightGBM)
├─ predict.py                        # Stage 6: Prediction generation
│
├─ model performance.py              # Stage 7: Model metrics
├─ business evaluation.py            # Stage 7: Business metrics
├─ feature performance.py            # Stage 7: Feature stability metrics
├─ feature analysis.py               # Stage 7: SHAP analysis
│
├─ calibration.py                    # Stage 8: Probability calibration (Platt / Isotonic)
├─ conformal_compare.py              # Stage 8: Compare RAW vs CAL conformal sets
├─ conformal_under_distribution_shift.py  # Stage 8: deal with distribution shift 
│
├─ README.md                         # Project introduction
└─ credit risk.pdf                   # Full project report

