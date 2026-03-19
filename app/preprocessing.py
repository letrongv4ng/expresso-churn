"""
preprocessing.py — Expresso Churn Prediction
=============================================
Pipeline xử lý data trước khi đưa vào CatBoost.

Cách dùng:
    from preprocessing import ChurnPreprocessor

    prep = ChurnPreprocessor()
    X_train, y_train = prep.fit_transform(df_train)
    prep.save("preprocessor.pkl")

    # Inference
    prep = ChurnPreprocessor.load("preprocessor.pkl")
    X = prep.transform(df_input)
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Columns bỏ hoàn toàn
DROP_COLS = ["user_id", "ZONE1", "ZONE2", "MRG", "ARPU_SEGMENT"]
# ARPU_SEGMENT bị drop vì tương quan 1.0 với REVENUE → thông tin trùng lặp

# Columns numeric cần impute median
NUMERIC_COLS = [
    "MONTANT", "FREQUENCE_RECH", "REVENUE", "FREQUENCE",
    "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO", "FREQ_TOP_PACK",
]

# Columns có missing >49% → tạo binary flag trước khi impute
# Giữ lại thông tin "khách hàng này có dùng không" thay vì mất đi
HIGH_MISSING_COLS = ["DATA_VOLUME", "TIGO"]

# Columns categorical — CatBoost xử lý natively, không cần encode số
CATEGORICAL_COLS = ["REGION", "TOP_PACK"]

# TENURE là ordinal (có thứ tự): chuỗi ngắn hơn → churn nhiều hơn
# → encode thành số để model hiểu thứ tự
TENURE_ORDER = {
    "D 3-6 month"   : 1,
    "E 6-9 month"   : 2,
    "F 9-12 month"  : 3,
    "G 12-15 month" : 4,
    "H 15-18 month" : 5,
    "I 18-21 month" : 6,
    "J 21-24 month" : 7,
    "K > 24 month"  : 8,
}

# Features bị lệch phải nặng → log1p transform để giảm ảnh hưởng outlier
# (log1p = log(x+1), xử lý được cả x=0)
LOG_TRANSFORM_COLS = [
    "MONTANT", "REVENUE", "FREQUENCE_RECH", "FREQUENCE",
    "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO", "FREQ_TOP_PACK",
]

# Winsorize tại 99th percentile để cap outlier cực đoan
# (áp dụng SAU log transform, trước khi đưa vào model)
WINSORIZE_COLS = [
    "MONTANT", "REVENUE", "FREQUENCE_RECH", "FREQUENCE",
    "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO",
]


# ─────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────

class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible preprocessor cho Expresso churn data.

    fit()       — học median/mode/winsorize caps từ train set
    transform() — áp dụng lên bất kỳ DataFrame nào (train/test/inference)
    fit_transform() — gộp 2 bước trên

    Lưu ý:
        - Khi inference 1 khách hàng, truyền DataFrame 1 dòng vào transform()
        - KHÔNG fit lại trên test set — chỉ fit 1 lần trên train
    """

    def __init__(self):
        self.medians_       = {}   # lưu median của từng numeric col
        self.modes_         = {}   # lưu mode của từng categorical col
        self.winsor_caps_   = {}   # lưu 99th percentile của WINSORIZE_COLS
        self.tenure_map_    = TENURE_ORDER
        self.cat_features_  = []   # tên columns categorical sau transform
        self.feature_names_ = []   # tên tất cả columns sau transform

    # ── FIT ──────────────────────────────────

    def fit(self, df: pd.DataFrame, y=None):
        """
        Học thống kê từ training data.
        df: DataFrame gốc (có thể có cột CHURN hoặc không)
        """
        data = df.copy()

        # 1. Drop ngay các cột không dùng
        data = self._drop_cols(data)

        # 2. Học median cho numeric
        for col in NUMERIC_COLS:
            if col in data.columns:
                self.medians_[col] = data[col].median()

        # 3. Học mode cho categorical
        for col in CATEGORICAL_COLS:
            if col in data.columns:
                mode_val = data[col].mode()
                self.modes_[col] = mode_val[0] if len(mode_val) > 0 else "UNKNOWN"

        # 4. Học winsorize caps (sau log transform)
        temp = data.copy()
        for col in LOG_TRANSFORM_COLS:
            if col in temp.columns:
                temp[col] = temp[col].fillna(0)
                temp[col] = np.log1p(temp[col].clip(lower=0))

        for col in WINSORIZE_COLS:
            if col in temp.columns:
                self.winsor_caps_[col] = temp[col].quantile(0.99)

        return self

    # ── TRANSFORM ────────────────────────────

    def transform(self, df: pd.DataFrame, return_y: bool = False):
        """
        Áp dụng toàn bộ pipeline lên DataFrame.

        Args:
            df          : DataFrame đầu vào
            return_y    : nếu True và df có cột CHURN → trả về (X, y)
                          dùng khi transform train set

        Returns:
            X (pd.DataFrame) — hoặc (X, y) nếu return_y=True
        """
        data = df.copy()

        # Tách target nếu có
        y = None
        if "CHURN" in data.columns:
            y = data["CHURN"].values
            data = data.drop(columns=["CHURN"])

        # ── Bước 1: Drop columns không cần ──
        data = self._drop_cols(data)

        # ── Bước 2: Binary flags cho cột missing nhiều ──
        # Đặt TRƯỚC khi impute để không bị ghi đè
        for col in HIGH_MISSING_COLS:
            if col in data.columns:
                flag_col = f"{col}_missing"
                data[flag_col] = data[col].isna().astype(int)

        # ── Bước 3: Impute numeric với median (học từ train) ──
        for col in NUMERIC_COLS:
            if col in data.columns and col in self.medians_:
                data[col] = data[col].fillna(self.medians_[col])
            elif col in data.columns:
                # fallback nếu chưa fit (không nên xảy ra)
                data[col] = data[col].fillna(data[col].median())

        # ── Bước 4: Impute categorical với mode ──
        for col in CATEGORICAL_COLS:
            if col in data.columns:
                fill_val = self.modes_.get(col, "UNKNOWN")
                data[col] = data[col].fillna(fill_val).replace("", fill_val)

        # ── Bước 5: Encode TENURE ordinal ──
        if "TENURE" in data.columns:
            data["TENURE"] = (
                data["TENURE"]
                .map(self.tenure_map_)
                .fillna(0)          # giá trị lạ không có trong map → 0
                .astype(int)
            )

        # ── Bước 6: Log1p transform cho skewed features ──
        for col in LOG_TRANSFORM_COLS:
            if col in data.columns:
                data[col] = np.log1p(data[col].clip(lower=0))

        # ── Bước 7: Winsorize outlier (99th percentile) ──
        for col in WINSORIZE_COLS:
            if col in data.columns and col in self.winsor_caps_:
                data[col] = data[col].clip(upper=self.winsor_caps_[col])

        # ── Bước 8: Đảm bảo categorical cols đúng kiểu string ──
        for col in CATEGORICAL_COLS:
            if col in data.columns:
                data[col] = data[col].astype(str)

        # ── Lưu metadata ──
        self.feature_names_ = list(data.columns)
        self.cat_features_  = [
            col for col in CATEGORICAL_COLS if col in data.columns
        ]

        if return_y:
            if y is None:
                raise ValueError("DataFrame không có cột CHURN — không thể return y.")
            return data, y

        return data

    def fit_transform(self, df: pd.DataFrame, y=None, return_y: bool = True):
        """
        Fit rồi transform luôn — dùng trên train set.

        Returns:
            (X, y) nếu return_y=True (default)
            X      nếu return_y=False
        """
        return self.fit(df).transform(df, return_y=return_y)

    # ── HELPER ───────────────────────────────

    def _drop_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in DROP_COLS if c in df.columns]
        return df.drop(columns=cols_to_drop)

    def get_cat_feature_indices(self) -> list[int]:
        """
        Trả về list index (số nguyên) của categorical features.
        CatBoost cần cat_features dạng index khi dùng Pool.
        """
        return [
            self.feature_names_.index(col)
            for col in self.cat_features_
            if col in self.feature_names_
        ]

    # ── SAVE / LOAD ───────────────────────────

    def save(self, path: str = "preprocessor.pkl"):
        joblib.dump(self, path)
        print(f"[Preprocessor] Saved → {path}")

    @staticmethod
    def load(path: str = "preprocessor.pkl") -> "ChurnPreprocessor":
        prep = joblib.load(path)
        print(f"[Preprocessor] Loaded ← {path}")
        return prep

    # ── SUMMARY ──────────────────────────────

    def summary(self):
        """In tóm tắt các thống kê đã học được sau fit()."""
        print("=" * 50)
        print("PREPROCESSOR SUMMARY")
        print("=" * 50)
        print(f"\n[Learned medians]")
        for col, val in self.medians_.items():
            print(f"  {col:<20} median = {val:.4f}")
        print(f"\n[Learned modes]")
        for col, val in self.modes_.items():
            print(f"  {col:<20} mode   = {val}")
        print(f"\n[Winsorize caps (99th pct, post log1p)]")
        for col, val in self.winsor_caps_.items():
            print(f"  {col:<20} cap    = {val:.4f}")
        print(f"\n[Features after transform] ({len(self.feature_names_)} total)")
        for i, col in enumerate(self.feature_names_):
            tag = " ← categorical" if col in self.cat_features_ else ""
            print(f"  {i:2d}. {col}{tag}")
        print("=" * 50)


# ─────────────────────────────────────────────
# QUICK TEST — chạy file trực tiếp để verify
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Tìm file Train.csv — thử vài đường dẫn phổ biến
    import os
    candidates = [
        "data/Train.csv",
        "Train.csv",
        "../data/Train.csv",
    ]
    train_path = next((p for p in candidates if os.path.exists(p)), None)

    if train_path is None:
        print("[ERROR] Không tìm thấy Train.csv. Chạy từ thư mục gốc project.")
        sys.exit(1)

    print(f"[Test] Đọc sample từ {train_path} ...")
    df_full = pd.read_csv(train_path)

    # Sample nhỏ để test nhanh
    df = df_full.groupby("CHURN", group_keys=False).apply(
        lambda x: x.sample(frac=0.01, random_state=42)
    ).reset_index(drop=True)
    print(f"[Test] Sample shape: {df.shape}")

    # Fit + transform
    prep = ChurnPreprocessor()
    X, y = prep.fit_transform(df, return_y=True)

    prep.summary()

    print(f"\n[Test] X shape       : {X.shape}")
    print(f"[Test] y shape       : {y.shape}")
    print(f"[Test] y distribution: {pd.Series(y).value_counts().to_dict()}")
    print(f"[Test] Missing values: {X.isna().sum().sum()} (phải = 0)")
    print(f"\n[Test] X preview:")
    print(X.head(3).to_string())

    # Test transform trên 1 dòng (simulate inference)
    print("\n[Test] Inference 1 dòng...")
    single_row = df.drop(columns=["CHURN"]).iloc[[0]]
    X_single = prep.transform(single_row)
    print(f"  Shape: {X_single.shape} ✓")

    # Save / load
    prep.save("/tmp/preprocessor_test.pkl")
    prep2 = ChurnPreprocessor.load("/tmp/preprocessor_test.pkl")
    X2, y2 = prep2.fit_transform(df, return_y=True)
    assert X.shape == X2.shape, "Shape mismatch sau save/load!"
    print("[Test] Save/load OK ✓")

    print("\n[Test] TẤT CẢ PASS ✓")