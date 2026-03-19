"""
train.py — Expresso Churn Prediction
======================================
Train CatBoost model, evaluate, lưu model + preprocessor.

Cách chạy:
    python train.py                      # train full data
    python train.py --sample 0.1         # train trên 10% để test nhanh
    python train.py --no-gpu             # bắt buộc dùng CPU
    python train.py --sample 0.1 --no-gpu

Output:
    model.cbm          — CatBoost model (dùng cho inference)
    preprocessor.pkl   — Fitted preprocessor
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

# Thêm app/ vào path để import preprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))
from preprocessing import ChurnPreprocessor

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

TRAIN_PATH      = "data/Train.csv"
MODEL_OUT       = "model.cbm"
PREP_OUT        = "preprocessor.pkl"


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_data(sample_frac: float = 1.0) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"  EXPRESSO CHURN — TRAINING PIPELINE")
    print(f"{'='*55}")

    if not os.path.exists(TRAIN_PATH):
        print(f"[ERROR] Không tìm thấy {TRAIN_PATH}")
        sys.exit(1)

    print(f"\n[1/5] Đọc data từ {TRAIN_PATH} ...")
    df = pd.read_csv(TRAIN_PATH)
    print(f"      Full data shape : {df.shape}")

    if sample_frac < 1.0:
        df = (
            df.groupby("CHURN", group_keys=False)
            .apply(lambda x: x.sample(frac=sample_frac, random_state=42))
            .reset_index(drop=True)
        )
        print(f"      Sampled shape   : {df.shape}  (frac={sample_frac})")

    churn_rate = df["CHURN"].mean() * 100
    print(f"      Churn rate      : {churn_rate:.1f}%")
    return df


# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    print(f"\n[2/5] Preprocessing ...")

    prep = ChurnPreprocessor()
    X, y = prep.fit_transform(df, return_y=True)

    print(f"      Features : {X.shape[1]} columns")
    print(f"      Cat cols : {prep.cat_features_}")
    print(f"      Samples  : {X.shape[0]:,} rows")
    return X, y, prep


# ─────────────────────────────────────────────
# TRAIN / VAL SPLIT
# ─────────────────────────────────────────────

def split(X: pd.DataFrame, y: np.ndarray):
    print(f"\n[3/5] Train/Val split (80/20, stratified) ...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train : {X_tr.shape[0]:,} rows")
    print(f"      Val   : {X_val.shape[0]:,} rows")
    return X_tr, X_val, y_tr, y_val


# ─────────────────────────────────────────────
# BUILD CATBOOST POOLS
# ─────────────────────────────────────────────

def build_pools(X_tr, y_tr, X_val, y_val, cat_indices: list[int]):
    """
    CatBoost Pool giúp model biết chính xác cột nào là categorical
    và tránh copy data trong RAM nhiều lần.
    """
    train_pool = Pool(data=X_tr, label=y_tr, cat_features=cat_indices)
    val_pool   = Pool(data=X_val, label=y_val, cat_features=cat_indices)
    return train_pool, val_pool


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train(
    train_pool,
    val_pool,
    use_gpu: bool = True,
) -> CatBoostClassifier:
    print(f"\n[4/5] Training CatBoost ...")

    # Detect GPU
    task_type = "GPU" if use_gpu else "CPU"
    try:
        if use_gpu:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            if result.returncode != 0:
                print("      [WARN] nvidia-smi not found → fallback CPU")
                task_type = "CPU"
    except FileNotFoundError:
        task_type = "CPU"

    print(f"      Task type : {task_type}")

    # ── Hyperparameters ──────────────────────
    # scale_pos_weight xử lý class imbalance:
    #   tỉ lệ 81:19 ≈ 4.3 → weight class 1 lên 4.3x
    #   thay vì SMOTE (oversample) vì CatBoost đã handle tốt
    neg_count = int((train_pool.get_label() == 0).sum())
    pos_count = int((train_pool.get_label() == 1).sum())
    scale_pos_weight = neg_count / pos_count
    print(f"      Class ratio 0:1 = {neg_count:,}:{pos_count:,}")
    print(f"      scale_pos_weight = {scale_pos_weight:.2f}")

    params = dict(
        iterations          = 1000,       # số cây — early stopping sẽ dừng sớm hơn
        learning_rate       = 0.05,       # nhỏ hơn → stable hơn, cần nhiều cây hơn
        depth               = 6,          # độ sâu cây — 6 là sweet spot cho tabular
        l2_leaf_reg         = 3.0,        # L2 regularization, giảm overfit
        scale_pos_weight    = scale_pos_weight,
        eval_metric         = "AUC",      # metric monitor khi training
        early_stopping_rounds = 50,       # dừng nếu val AUC không cải thiện 50 rounds
        random_seed         = 42,
        task_type           = task_type,
        verbose             = 100,        # in log mỗi 100 iterations
    )

    # GPU không support l2_leaf_reg + một số params → điều chỉnh
    if task_type == "GPU":
        params["devices"] = "0"  # dùng GPU đầu tiên (RTX 4050)

    model = CatBoostClassifier(**params)

    t0 = time.time()
    model.fit(
        train_pool,
        eval_set    = val_pool,
        use_best_model = True,   # lưu model tại iteration tốt nhất trên val
    )
    elapsed = time.time() - t0
    print(f"\n      Training time : {elapsed:.1f}s")
    print(f"      Best iteration: {model.get_best_iteration()}")

    return model


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

def evaluate(model: CatBoostClassifier, X_val, y_val, cat_indices: list[int]):
    print(f"\n[5/5] Evaluation trên validation set ...")

    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_indices)
    y_prob   = model.predict_proba(val_pool)[:, 1]
    y_pred   = model.predict(val_pool)

    auc      = roc_auc_score(y_val, y_prob)
    f1       = f1_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average="macro")

    print(f"\n  {'─'*40}")
    print(f"  AUC-ROC   : {auc:.4f}   (target > 0.80)")
    print(f"  F1 churn  : {f1:.4f}   (target > 0.55)")
    print(f"  F1 macro  : {f1_macro:.4f}")
    print(f"  {'─'*40}")

    print(f"\n  Classification report:")
    print(classification_report(y_val, y_pred, target_names=["No Churn", "Churn"]))

    cm = confusion_matrix(y_val, y_pred)
    print(f"  Confusion matrix:")
    print(f"               Pred 0   Pred 1")
    print(f"  Actual 0  :  {cm[0][0]:6d}   {cm[0][1]:6d}")
    print(f"  Actual 1  :  {cm[1][0]:6d}   {cm[1][1]:6d}")

    # Feature importance
    print(f"\n  Top 10 feature importance:")
    fi = pd.Series(
        model.get_feature_importance(),
        index=X_val.columns
    ).sort_values(ascending=False)
    for feat, score in fi.head(10).items():
        bar = "█" * int(score / 2)
        print(f"  {feat:<22} {score:5.2f}  {bar}")

    return auc, f1


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def save_artifacts(model: CatBoostClassifier, prep: ChurnPreprocessor):
    model.save_model(MODEL_OUT)
    print(f"\n  Model saved       → {MODEL_OUT}")
    prep.save(PREP_OUT)
    print(f"  Preprocessor saved → {PREP_OUT}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Expresso Churn model")
    parser.add_argument(
        "--sample", type=float, default=1.0,
        help="Fraction of data to use, e.g. 0.1 for 10%% (default: 1.0 = full data)"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Force CPU training (default: auto-detect GPU)"
    )
    args = parser.parse_args()

    use_gpu = not args.no_gpu

    # Pipeline
    df                       = load_data(sample_frac=args.sample)
    X, y, prep               = preprocess(df)
    X_tr, X_val, y_tr, y_val = split(X, y)
    cat_indices              = prep.get_cat_feature_indices()
    train_pool, val_pool     = build_pools(X_tr, y_tr, X_val, y_val, cat_indices)
    model                    = train(train_pool, val_pool, use_gpu=use_gpu)
    auc, f1                  = evaluate(model, X_val, y_val, cat_indices)
    save_artifacts(model, prep)

    print(f"\n{'='*55}")
    print(f"  DONE!  AUC={auc:.4f}  F1={f1:.4f}")
    print(f"  Files: {MODEL_OUT}, {PREP_OUT}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()