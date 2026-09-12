import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "40000_yt_videos.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print()


# =========================================================
# 2. PREPROCESSING
# =========================================================

# Convert publish_date menjadi datetime
df["publish_date"] = pd.to_datetime(
    df["publish_date"],
    errors="coerce"
)

# Ambil informasi waktu publikasi
df["publish_hour"] = df["publish_date"].dt.hour
df["publish_dayofweek"] = df["publish_date"].dt.dayofweek
df["publish_month"] = df["publish_date"].dt.month


# =========================================================
# 3. MEMBUAT TARGET
# =========================================================

# Kita menggunakan median views sebagai batas:
#
# views >= median  -> High Engagement (1)
# views < median   -> Low Engagement  (0)

views_threshold = df["views"].median()

df["high_engagement"] = (
    df["views"] >= views_threshold
).astype(int)

print(f"Median views: {views_threshold:,.0f}")

print("\nTarget distribution:")
print(df["high_engagement"].value_counts())
print()


# =========================================================
# 4. MEMILIH FEATURES
# =========================================================

features = [
    "category_id",
    "duration_sec",
    "subscriber_count",
    "publish_hour",
    "publish_dayofweek",
    "publish_month"
]

X = df[features]
y = df["high_engagement"]


# =========================================================
# 5. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data :", X_train.shape)
print("Testing data  :", X_test.shape)
print()


# =========================================================
# 6. PREPROCESSOR
# =========================================================

categorical_features = [
    "category_id",
    "publish_hour",
    "publish_dayofweek",
    "publish_month"
]

numeric_features = [
    "duration_sec",
    "subscriber_count"
]


preprocessor = ColumnTransformer(
    transformers=[

        # Categorical
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            categorical_features
        ),

        # Numerical
        (
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ]),
            numeric_features
        )
    ]
)


# =========================================================
# 7. RANDOM FOREST
# =========================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# =========================================================
# 8. PIPELINE
# =========================================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])


# =========================================================
# 9. TRAIN MODEL
# =========================================================

print("Training Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("Training selesai!")
print()


# =========================================================
# 10. PREDICTION
# =========================================================

y_pred = pipeline.predict(X_test)

# Probabilitas class 1
y_probability = pipeline.predict_proba(X_test)[:, 1]


# =========================================================
# 11. EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_probability
)


print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =========================================================
# 12. CHECK REQUIREMENT
# =========================================================

print("\n" + "=" * 60)
print("REQUIREMENT CHECK")
print("=" * 60)

if accuracy >= 0.70:
    print("✓ Accuracy >= 0.70")
else:
    print("✗ Accuracy < 0.70")

if auc >= 0.70:
    print("✓ ROC-AUC >= 0.70")
else:
    print("✗ ROC-AUC < 0.70")


# =========================================================
# 13. SAVE MODEL
# =========================================================

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "youtube_engagement_pipeline.joblib"

joblib.dump(
    pipeline,
    MODEL_PATH
)


# =========================================================
# 14. SAVE METADATA
# =========================================================

metadata = {
    "views_threshold": float(views_threshold),
    "accuracy": float(accuracy),
    "auc": float(auc),
    "features": features
}

METADATA_PATH = MODEL_DIR / "metadata.joblib"

joblib.dump(
    metadata,
    METADATA_PATH
)


# =========================================================
# 15. SELESAI
# =========================================================

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"Model    : {MODEL_PATH}")
print(f"Metadata : {METADATA_PATH}")