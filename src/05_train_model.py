import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: LightGBM not installed. Run: pip install lightgbm")
    print("         Continuing with XGBoost + Random Forest only.")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURES, TRAIN_RATIO, VAL_RATIO

print("Loading dataset...")

df = pd.read_csv("processed/premier_league_features_v3.csv")
df["matchDate"] = pd.to_datetime(df["matchDate"])
df = df.sort_values("matchDate").reset_index(drop=True)
df["over25"] = (df["total_goals"] > 2.5).astype(int)

# -------------------------------------------------------
# WALK-FORWARD SPLIT — strict temporal order, no leakage
# -------------------------------------------------------

print("")
print("Running walk-forward validation...")
print("(Training only on past data, never touching future)")
print("")

split_train = int(len(df) * TRAIN_RATIO)
split_val   = int(len(df) * VAL_RATIO)

train_df = df.iloc[:split_train]
val_df   = df.iloc[split_train:split_val]
test_df  = df.iloc[split_val:]

print(f"Train:      {len(train_df)} matches  ({train_df['matchDate'].min().date()} to {train_df['matchDate'].max().date()})")
print(f"Validation: {len(val_df)} matches  ({val_df['matchDate'].min().date()} to {val_df['matchDate'].max().date()})")
print(f"Test:       {len(test_df)} matches  ({test_df['matchDate'].min().date()} to {test_df['matchDate'].max().date()})")
print("")

X_train = train_df[FEATURES]
y_train = train_df["over25"]
X_val   = val_df[FEATURES]
y_val   = val_df["over25"]
X_test  = test_df[FEATURES]
y_test  = test_df["over25"]

# -------------------------------------------------------
# MODEL 1: XGBoost
# -------------------------------------------------------

print("Training Model 1: XGBoost...")

xgb = XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    gamma=1, reg_lambda=2, random_state=42,
    eval_metric="logloss", early_stopping_rounds=30, verbosity=0
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"  Best iteration: {xgb.best_iteration}")

xgb_val_probs  = xgb.predict_proba(X_val)[:, 1]
xgb_test_probs = xgb.predict_proba(X_test)[:, 1]

# -------------------------------------------------------
# MODEL 2: LightGBM
# -------------------------------------------------------

if HAS_LGBM:
    print("Training Model 2: LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
        reg_lambda=2, random_state=42, verbose=-1,
    )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[__import__("lightgbm").early_stopping(30, verbose=False),
                   __import__("lightgbm").log_evaluation(period=-1)]
    )
    lgbm_val_probs  = lgbm.predict_proba(X_val)[:, 1]
    lgbm_test_probs = lgbm.predict_proba(X_test)[:, 1]
    print(f"  Best iteration: {lgbm.best_iteration_}")
else:
    lgbm = None
    lgbm_val_probs  = None
    lgbm_test_probs = None

# -------------------------------------------------------
# MODEL 3: Random Forest
# -------------------------------------------------------

print("Training Model 3: Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=10,
    max_features="sqrt", random_state=42, n_jobs=-1,
)
rf.fit(X_train, y_train)

rf_val_probs  = rf.predict_proba(X_val)[:, 1]
rf_test_probs = rf.predict_proba(X_test)[:, 1]

# -------------------------------------------------------
# INDIVIDUAL MODEL PERFORMANCE
# -------------------------------------------------------

print("")
print("=" * 55)
print("  INDIVIDUAL MODEL PERFORMANCE (test set)")
print("=" * 55)

base_rate      = y_test.mean()
brier_baseline = brier_score_loss(y_test, np.full(len(y_test), base_rate))

models_eval = [("XGBoost", xgb_test_probs), ("Random Forest", rf_test_probs)]
if HAS_LGBM:
    models_eval.insert(1, ("LightGBM", lgbm_test_probs))

for name, probs in models_eval:
    auc   = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    bss   = 1 - (brier / brier_baseline)
    print(f"  {name:<20} AUC={auc:.4f}  BSS={bss:.4f}")

print("=" * 55)

# -------------------------------------------------------
# STACKING META-MODEL
# Learns optimal weights for each model
# -------------------------------------------------------

print("")
print("Training Stacking meta-model...")

if HAS_LGBM:
    stack_val_X  = np.column_stack([xgb_val_probs,  lgbm_val_probs,  rf_val_probs])
    stack_test_X = np.column_stack([xgb_test_probs, lgbm_test_probs, rf_test_probs])
    model_names  = ["XGBoost", "LightGBM", "Random Forest"]
else:
    stack_val_X  = np.column_stack([xgb_val_probs,  rf_val_probs])
    stack_test_X = np.column_stack([xgb_test_probs, rf_test_probs])
    model_names  = ["XGBoost", "Random Forest"]

meta_model = LogisticRegression(C=1.0, random_state=42)
meta_model.fit(stack_val_X, y_val)

ensemble_probs = meta_model.predict_proba(stack_test_X)[:, 1]

print("  Meta-model weights:")
for name, coef in zip(model_names, meta_model.coef_[0]):
    bar = "█" * max(1, int(abs(coef) * 10))
    print(f"    {name:<20} {coef:+.4f}  {bar}")

# -------------------------------------------------------
# CALIBRATE ENSEMBLE
# -------------------------------------------------------

print("")
print("Calibrating ensemble probabilities...")

raw_ens_val = meta_model.predict_proba(stack_val_X)[:, 1].reshape(-1, 1)
calibrator  = LogisticRegression(C=1.0)
calibrator.fit(raw_ens_val, y_val)

final_probs = calibrator.predict_proba(ensemble_probs.reshape(-1, 1))[:, 1]

# -------------------------------------------------------
# FINAL ENSEMBLE PERFORMANCE
# -------------------------------------------------------

auc_ens   = roc_auc_score(y_test, final_probs)
brier_ens = brier_score_loss(y_test, final_probs)
bss_ens   = 1 - (brier_ens / brier_baseline)

print("")
print("=" * 55)
print("  ENSEMBLE MODEL PERFORMANCE")
print("  Trained on past only — zero data leakage")
print("=" * 55)
print(f"  Matches in test:     {len(test_df)}")
print(f"  Over 2.5 base rate:  {base_rate:.1%}")
print(f"  AUC:                 {auc_ens:.4f}")
print(f"  Brier Score:         {brier_ens:.4f}  (baseline: {brier_baseline:.4f})")
print(f"  Brier Skill Score:   {bss_ens:.4f}")
print("=" * 55)

best_ind_auc = max(roc_auc_score(y_test, p) for _, p in models_eval)
if auc_ens >= best_ind_auc:
    print(f"  ✅ Ensemble beats best individual model (AUC +{auc_ens - best_ind_auc:.4f})")
else:
    print(f"  ⚠️  Best individual slightly better — using ensemble for robustness")

print("")
counts, bins = np.histogram(final_probs, bins=10)
print("Probability distribution:")
for i in range(len(counts)):
    bar = "#" * int(counts[i] / 2)
    print(f"  {bins[i]:.2f}-{bins[i+1]:.2f}: {bar} ({counts[i]})")
print(f"  Std dev: {final_probs.std():.3f}")

print("")
print("Top features by importance (XGBoost):")
importances = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in importances.head(12).items():
    bar = "=" * int(imp * 200)
    print(f"  {feat:<35} {bar} {imp:.4f}")

# -------------------------------------------------------
# RETRAIN ALL MODELS ON FULL DATA FOR PRODUCTION
# -------------------------------------------------------

print("")
print("Retraining all models on full data for production...")

full_X = df[FEATURES]
full_y = df["over25"]

final_xgb = XGBClassifier(
    n_estimators=xgb.best_iteration + 1, max_depth=4, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    gamma=1, reg_lambda=2, random_state=42, verbosity=0
)
final_xgb.fit(full_X, full_y)

if HAS_LGBM:
    final_lgbm = LGBMClassifier(
        n_estimators=lgbm.best_iteration_, max_depth=4, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
        reg_lambda=2, random_state=42, verbose=-1
    )
    final_lgbm.fit(full_X, full_y)
else:
    final_lgbm = None

final_rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=10,
    max_features="sqrt", random_state=42, n_jobs=-1
)
final_rf.fit(full_X, full_y)

# Production stacking calibrated on last 30%
cal_start = int(len(df) * 0.70)
cal_X = df.iloc[cal_start:][FEATURES]
cal_y = full_y.iloc[cal_start:]

xgb_c = final_xgb.predict_proba(cal_X)[:, 1]
rf_c  = final_rf.predict_proba(cal_X)[:, 1]

if HAS_LGBM:
    lgbm_c      = final_lgbm.predict_proba(cal_X)[:, 1]
    stack_cal_X = np.column_stack([xgb_c, lgbm_c, rf_c])
else:
    stack_cal_X = np.column_stack([xgb_c, rf_c])

prod_meta = LogisticRegression(C=1.0, random_state=42)
prod_meta.fit(stack_cal_X, cal_y)

raw_prod  = prod_meta.predict_proba(stack_cal_X)[:, 1].reshape(-1, 1)
prod_cal  = LogisticRegression(C=1.0)
prod_cal.fit(raw_prod, cal_y)

# Bundle: (xgb, lgbm, rf, meta_model, calibrator, has_lgbm)
eval_bundle = (xgb,       lgbm,       rf,       meta_model, calibrator, HAS_LGBM)
prod_bundle = (final_xgb, final_lgbm, final_rf, prod_meta,  prod_cal,   HAS_LGBM)

joblib.dump(eval_bundle, "models/over25_model_eval.pkl")
joblib.dump(prod_bundle, "models/over25_model.pkl")

print("")
print("Models saved:")
print("  models/over25_model_eval.pkl  <- honest evaluation model")
print("  models/over25_model.pkl       <- production ensemble (XGB + LGBM + RF + stacking)")