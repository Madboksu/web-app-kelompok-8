import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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

DATA_PATH = BASE_DIR / "toxic_comments_50000.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset awal: {df.shape}")


# =========================================================
# 2. CEK KOLOM
# =========================================================

required_columns = [
    "comment_text",
    "label"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Kolom '{column}' tidak ditemukan!"
        )


# =========================================================
# 3. CLEANING
# =========================================================

df = df[
    ["comment_text", "label"]
].copy()

# Hapus missing
df = df.dropna(
    subset=[
        "comment_text",
        "label"
    ]
)

# Ubah text menjadi string
df["comment_text"] = (
    df["comment_text"]
    .astype(str)
    .str.strip()
)

# Hapus text kosong
df = df[
    df["comment_text"] != ""
]

# Hapus duplicate berdasarkan komentar
before = len(df)

df = df.drop_duplicates(
    subset=["comment_text"]
)

after = len(df)

print(
    f"Duplicate comment dihapus: "
    f"{before - after:,}"
)

print(
    f"Dataset setelah cleaning: "
    f"{df.shape}"
)


# =========================================================
# 4. TARGET DISTRIBUTION
# =========================================================

print("\nDistribusi label:")

print(
    df["label"].value_counts()
)


# =========================================================
# 5. FEATURES & TARGET
# =========================================================

X = df["comment_text"]

y = df["label"]


# =========================================================
# 6. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print(len(X_train))

print("Testing data:")
print(len(X_test))


# =========================================================
# 7. PIPELINE
# =========================================================

pipeline = Pipeline([
    
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# =========================================================
# 8. TRAINING
# =========================================================

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training selesai!")


# =========================================================
# 9. PREDICTION
# =========================================================

y_pred = pipeline.predict(
    X_test
)

y_probability = pipeline.predict_proba(
    X_test
)


# =========================================================
# 10. ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# =========================================================
# 11. MULTICLASS ROC-AUC
# =========================================================

classes = pipeline.classes_

auc = roc_auc_score(
    y_test,
    y_probability,
    multi_class="ovr",
    average="weighted",
    labels=classes
)


# =========================================================
# 12. EVALUATION
# =========================================================

print("\n")
print("=" * 60)
print("MODERATION MODEL EVALUATION")
print("=" * 60)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"ROC-AUC  : {auc:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =========================================================
# 13. REQUIREMENT CHECK
# =========================================================

print("\n")
print("=" * 60)
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
# 14. SAVE MODEL
# =========================================================

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR /
    "moderation_pipeline.joblib"
)

joblib.dump(
    pipeline,
    MODEL_PATH
)


# =========================================================
# 15. SAVE METADATA
# =========================================================

metadata = {
    "accuracy": float(accuracy),
    "auc": float(auc),
    "classes": list(classes),
    "model": "Logistic Regression",
    "vectorizer": "TF-IDF",
    "training_rows": len(X_train),
    "testing_rows": len(X_test),
    "duplicate_removed": before - after
}

METADATA_PATH = (
    MODEL_DIR /
    "moderation_metadata.joblib"
)

joblib.dump(
    metadata,
    METADATA_PATH
)


# =========================================================
# 16. SELESAI
# =========================================================

print("\n")
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    f"Model    : {MODEL_PATH}"
)

print(
    f"Metadata : {METADATA_PATH}"
)