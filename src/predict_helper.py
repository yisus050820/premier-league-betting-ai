"""
Helper: load ensemble model and predict probabilities.
Works with both old bundle (xgb, cal) and new bundle (xgb, lgbm, rf, meta, cal, has_lgbm).
"""
import numpy as np
import joblib

def load_model(path="models/over25_model.pkl"):
    bundle = joblib.load(path)
    if len(bundle) == 2:
        # Old format: (xgb, calibrator)
        xgb, cal = bundle
        return {"type": "single", "xgb": xgb, "cal": cal}
    else:
        # New format: (xgb, lgbm, rf, meta, cal, has_lgbm)
        xgb, lgbm, rf, meta, cal, has_lgbm = bundle
        return {"type": "ensemble", "xgb": xgb, "lgbm": lgbm, "rf": rf,
                "meta": meta, "cal": cal, "has_lgbm": has_lgbm}

def predict_proba(model_bundle, X):
    if model_bundle["type"] == "single":
        raw = model_bundle["xgb"].predict_proba(X)[:, 1].reshape(-1, 1)
        return model_bundle["cal"].predict_proba(raw)[:, 1]
    else:
        xgb_p = model_bundle["xgb"].predict_proba(X)[:, 1]
        rf_p  = model_bundle["rf"].predict_proba(X)[:, 1]
        if model_bundle["has_lgbm"] and model_bundle["lgbm"] is not None:
            lgbm_p  = model_bundle["lgbm"].predict_proba(X)[:, 1]
            stack_X = np.column_stack([xgb_p, lgbm_p, rf_p])
        else:
            stack_X = np.column_stack([xgb_p, rf_p])
        ens_p = model_bundle["meta"].predict_proba(stack_X)[:, 1].reshape(-1, 1)
        return model_bundle["cal"].predict_proba(ens_p)[:, 1]