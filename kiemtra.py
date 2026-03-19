import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, './app')
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from app.preprocessing import ChurnPreprocessor

print("Loading model + preprocessor...")
model = CatBoostClassifier()
model.load_model("model.cbm")
prep = ChurnPreprocessor.load("preprocessor.pkl")

print("Loading 10% sample...")
df = pd.read_csv("data/Train.csv")
df = df.groupby("CHURN", group_keys=False).apply(
    lambda x: x.sample(frac=0.1, random_state=42)
).reset_index(drop=True)

X, y = prep.transform(df, return_y=True)
cat_idx = prep.get_cat_feature_indices()

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
val_pool   = Pool(X_val, y_val, cat_features=cat_idx)

auc_train = roc_auc_score(y_tr, model.predict_proba(train_pool)[:,1])
auc_val   = roc_auc_score(y_val, model.predict_proba(val_pool)[:,1])

print(f"\nTrain AUC : {auc_train:.4f}")
print(f"Val AUC   : {auc_val:.4f}")
print(f"Gap       : {auc_train - auc_val:.4f}  {'OK' if auc_train - auc_val < 0.02 else 'CẦN XEM LẠI'}")
