"""
app/model.py — Expresso Churn Prediction
==========================================
Load model + preprocessor, expose hàm predict cho FastAPI dùng.
"""

import os
import sys
import io
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

# Import preprocessor từ cùng thư mục app/
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import ChurnPreprocessor

# ─────────────────────────────────────────────
# PATHS — tính từ thư mục gốc project
# ─────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.cbm")
PREP_PATH  = os.path.join(BASE_DIR, "preprocessor.pkl")

# ─────────────────────────────────────────────
# LOAD MODEL (singleton — load 1 lần khi khởi động)
# ─────────────────────────────────────────────

print(f"[model.py] Loading model từ {MODEL_PATH} ...")
_model = CatBoostClassifier()
_model.load_model(MODEL_PATH)

print(f"[model.py] Loading preprocessor từ {PREP_PATH} ...")
_prep = ChurnPreprocessor.load(PREP_PATH)

print(f"[model.py] Ready! Features: {_prep.feature_names_}")


# ─────────────────────────────────────────────
# SCHEMA — các fields đầu vào hợp lệ
# ─────────────────────────────────────────────

VALID_REGIONS = [
    "DAKAR", "THIES", "SAINT-LOUIS", "DIOURBEL", "LOUGA",
    "TAMBACOUNDA", "KAOLACK", "ZIGUINCHOR", "FATICK",
    "KOLDA", "MATAM", "KAFFRINE", "KEDOUGOU", "SEDHIOU",
]

VALID_TENURES = [
    "D 3-6 month", "E 6-9 month", "F 9-12 month",
    "G 12-15 month", "H 15-18 month", "I 18-21 month",
    "J 21-24 month", "K > 24 month",
]

# ─────────────────────────────────────────────
# PREDICT — 1 khách hàng
# ─────────────────────────────────────────────

def predict_single(customer: dict) -> dict:
    """
    Predict churn cho 1 khách hàng.

    Args:
        customer: dict với các fields như REGION, TENURE, MONTANT, ...

    Returns:
        {
            "churn_probability": 0.72,
            "churn_prediction": 1,
            "risk_level": "Cao",
            "top_factors": [{"feature": "REGULARITY", "importance": 30.32}, ...]
        }
    """
    df = pd.DataFrame([customer])
    X  = _prep.transform(df)

    cat_indices = _prep.get_cat_feature_indices()
    pool        = Pool(data=X, cat_features=cat_indices)

    prob = float(_model.predict_proba(pool)[0][1])
    pred = int(prob >= 0.5)

    # Feature importance (global từ model, không phải per-sample)
    fi = pd.Series(
        _model.get_feature_importance(),
        index=_prep.feature_names_
    ).sort_values(ascending=False)

    top_factors = [
        {"feature": feat, "importance": round(float(score), 2)}
        for feat, score in fi.head(5).items()
    ]

    return {
        "churn_probability" : round(prob, 4),
        "churn_prediction"  : pred,
        "risk_level"        : _risk_level(prob),
        "top_factors"       : top_factors,
    }


# ─────────────────────────────────────────────
# PREDICT — batch CSV
# ─────────────────────────────────────────────

def predict_batch(csv_bytes: bytes) -> tuple[pd.DataFrame, dict]:
    """
    Predict churn cho nhiều khách hàng từ file CSV.

    Args:
        csv_bytes: raw bytes của CSV file upload

    Returns:
        (result_df, summary)
        result_df: DataFrame gốc + thêm cột churn_probability, churn_prediction, risk_level
        summary  : thống kê tổng hợp
    """
    df_raw = pd.read_csv(io.BytesIO(csv_bytes))

    # Giữ lại user_id nếu có để ghép vào kết quả
    user_ids = df_raw["user_id"].values if "user_id" in df_raw.columns else None

    X = _prep.transform(df_raw)

    cat_indices = _prep.get_cat_feature_indices()
    pool        = Pool(data=X, cat_features=cat_indices)

    probs = _model.predict_proba(pool)[:, 1]
    preds = (probs >= 0.5).astype(int)

    result_df = df_raw.copy()
    result_df["churn_probability"] = np.round(probs, 4)
    result_df["churn_prediction"]  = preds
    result_df["risk_level"]        = [_risk_level(p) for p in probs]

    # Đưa cột kết quả lên đầu
    front_cols = ["churn_probability", "churn_prediction", "risk_level"]
    if user_ids is not None:
        front_cols = ["user_id"] + front_cols
        result_df  = result_df.drop(columns=["user_id"])
        result_df.insert(0, "user_id", user_ids)

    summary = {
        "total"           : len(result_df),
        "churn_count"     : int(preds.sum()),
        "churn_rate"      : round(float(preds.mean()) * 100, 1),
        "high_risk_count" : int((probs >= 0.7).sum()),
        "mid_risk_count"  : int(((probs >= 0.4) & (probs < 0.7)).sum()),
        "low_risk_count"  : int((probs < 0.4).sum()),
    }

    return result_df, summary


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def _risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "Cao"
    elif prob >= 0.4:
        return "Trung bình"
    else:
        return "Thấp"