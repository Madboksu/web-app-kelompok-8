import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("=" * 60)
print("1. LOAD DATASET")
print("=" * 60)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "40000_yt_videos.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print()


# =========================================================
# 2. CEK DATA
# =========================================================

print("=" * 60)
print("2. CEK DATA")
print("=" * 60)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print()


# =========================================================
# 3. DATA CLEANING
# =========================================================

print("=" * 60)
print("3. DATA CLEANING")
print("=" * 60)

# Convert publish_date menjadi datetime
df["publish_date"] = pd.to_datetime(
    df["publish_date"],
    errors="coerce"
)

# Hapus baris yang tidak memiliki views
df = df.dropna(
    subset=["views"]
).copy()

print(f"Dataset setelah cleaning: {df.shape}")

print()


# =========================================================
# 4. MEMBUAT TARGET
# =========================================================

print("=" * 60)
print("4. MEMBUAT TARGET")
print("=" * 60)

# Median views digunakan sebagai threshold.
#
# views >= median -> High Engagement (1)
# views < median  -> Low Engagement  (0)

views_threshold = df["views"].median()

df["high_engagement"] = (
    df["views"] >= views_threshold
).astype(int)

print(
    f"Median views: {views_threshold:,.0f}"
)

print("\nTarget distribution:")
print(
    df["high_engagement"].value_counts()
)

print()


# =========================================================
# 5. MEMILIH FEATURES
# =========================================================

print("=" * 60)
print("5. MEMILIH FEATURES")
print("=" * 60)

features = [
    "category_id",
    "duration_sec",
    "subscriber_count",
    "publish_hour",
    "publish_dayofweek",
    "publish_month"
]

# Buat fitur waktu dari publish_date
df["publish_hour"] = (
    df["publish_date"].dt.hour
)

df["publish_dayofweek"] = (
    df["publish_date"].dt.dayofweek
)

df["publish_month"] = (
    df["publish_date"].dt.month
)

X = df[features]
y = df["high_engagement"]

print("\nFeatures yang digunakan:")

for feature in features:
    print(f"- {feature}")

print()


# =========================================================
# 6. TRAIN TEST SPLIT
# =========================================================

print("=" * 60)
print("6. TRAIN TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"Training data : {X_train.shape}"
)

print(
    f"Testing data  : {X_test.shape}"
)

print()


# =========================================================
# 7. DEFINISI MODEL
# =========================================================

print("=" * 60)
print("7. DEFINISI MODEL")
print("=" * 60)

# Logistic Regression
logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# Decision Tree
decision_tree_model = DecisionTreeClassifier(
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

# Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Model yang digunakan:")
print("- Logistic Regression")
print("- Decision Tree")
print("- Random Forest")

print()


# =========================================================
# 8. PREPROCESSING
# =========================================================

print("=" * 60)
print("8. PREPROCESSING")
print("=" * 60)

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

print("\nCategorical features:")

for feature in categorical_features:
    print(f"- {feature}")

print("\nNumeric features:")

for feature in numeric_features:
    print(f"- {feature}")


# ---------------------------------------------------------
# Preprocessing untuk Logistic Regression
# ---------------------------------------------------------

preprocessor_logistic = ColumnTransformer(
    transformers=[

        (
            "categorical",

            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
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

        (
            "numeric",

            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),

                (
                    "scaler",
                    StandardScaler()
                )
            ]),

            numeric_features
        )
    ]
)


# ---------------------------------------------------------
# Preprocessing untuk Decision Tree & Random Forest
# ---------------------------------------------------------

preprocessor_tree = ColumnTransformer(
    transformers=[

        (
            "categorical",

            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
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

        (
            "numeric",

            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]),

            numeric_features
        )
    ]
)


# ---------------------------------------------------------
# Pipeline masing-masing model
# ---------------------------------------------------------

logistic_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor_logistic
    ),

    (
        "classifier",
        logistic_model
    )
])


decision_tree_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor_tree
    ),

    (
        "classifier",
        decision_tree_model
    )
])


random_forest_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor_tree
    ),

    (
        "classifier",
        random_forest_model
    )
])

print("\nPreprocessing selesai!")
print()


# =========================================================
# 9. TRAINING MODEL
# =========================================================

print("=" * 60)
print("9. TRAINING MODEL")
print("=" * 60)


# ---------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------

print("\nTraining: Logistic Regression")

logistic_pipeline.fit(
    X_train,
    y_train
)

logistic_pred = (
    logistic_pipeline.predict(X_test)
)

logistic_probability = (
    logistic_pipeline
    .predict_proba(X_test)[:, 1]
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

logistic_f1 = f1_score(
    y_test,
    logistic_pred
)

logistic_auc = roc_auc_score(
    y_test,
    logistic_probability
)

print(
    f"Accuracy : {logistic_accuracy:.4f}"
)

print(
    f"F1 Score : {logistic_f1:.4f}"
)

print(
    f"ROC-AUC  : {logistic_auc:.4f}"
)


# ---------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------

print("\nTraining: Decision Tree")

decision_tree_pipeline.fit(
    X_train,
    y_train
)

decision_tree_pred = (
    decision_tree_pipeline
    .predict(X_test)
)

decision_tree_probability = (
    decision_tree_pipeline
    .predict_proba(X_test)[:, 1]
)

decision_tree_accuracy = accuracy_score(
    y_test,
    decision_tree_pred
)

decision_tree_f1 = f1_score(
    y_test,
    decision_tree_pred
)

decision_tree_auc = roc_auc_score(
    y_test,
    decision_tree_probability
)

print(
    f"Accuracy : {decision_tree_accuracy:.4f}"
)

print(
    f"F1 Score : {decision_tree_f1:.4f}"
)

print(
    f"ROC-AUC  : {decision_tree_auc:.4f}"
)


# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

print("\nTraining: Random Forest")

random_forest_pipeline.fit(
    X_train,
    y_train
)

random_forest_pred = (
    random_forest_pipeline
    .predict(X_test)
)

random_forest_probability = (
    random_forest_pipeline
    .predict_proba(X_test)[:, 1]
)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_pred
)

random_forest_f1 = f1_score(
    y_test,
    random_forest_pred
)

random_forest_auc = roc_auc_score(
    y_test,
    random_forest_probability
)

print(
    f"Accuracy : {random_forest_accuracy:.4f}"
)

print(
    f"F1 Score : {random_forest_f1:.4f}"
)

print(
    f"ROC-AUC  : {random_forest_auc:.4f}"
)

print()


# =========================================================
# 10. PERBANDINGAN MODEL
# =========================================================

print("=" * 60)
print("10. PERBANDINGAN MODEL")
print("=" * 60)

comparison_df = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        logistic_accuracy,
        decision_tree_accuracy,
        random_forest_accuracy
    ],

    "F1 Score": [
        logistic_f1,
        decision_tree_f1,
        random_forest_f1
    ],

    "ROC-AUC": [
        logistic_auc,
        decision_tree_auc,
        random_forest_auc
    ]
})


# Urutkan berdasarkan ROC-AUC
comparison_df = comparison_df.sort_values(
    by="ROC-AUC",
    ascending=False
).reset_index(drop=True)


print(
    comparison_df.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Menentukan model terbaik
# ---------------------------------------------------------

best_model_name = (
    comparison_df.iloc[0]["Model"]
)

print(
    f"\nBest Model berdasarkan ROC-AUC: "
    f"{best_model_name}"
)


# ---------------------------------------------------------
# Save comparison
# ---------------------------------------------------------

EDA_DIR = BASE_DIR / "eda_results"

EDA_DIR.mkdir(
    exist_ok=True
)

COMPARISON_PATH = (
    EDA_DIR /
    "model_comparison.csv"
)

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False
)

print(
    f"\nModel comparison saved:"
)

print(
    COMPARISON_PATH
)

print()


# =========================================================
# 11. EVALUASI RANDOM FOREST
# =========================================================

print("=" * 60)
print("11. EVALUASI RANDOM FOREST")
print("=" * 60)

print(
    f"Accuracy : {random_forest_accuracy:.4f}"
)

print(
    f"F1 Score : {random_forest_f1:.4f}"
)

print(
    f"ROC-AUC  : {random_forest_auc:.4f}"
)


print("\nClassification Report:")

rf_report = classification_report(
    y_test,
    random_forest_pred
)

print(
    rf_report
)


# =========================================================
# 12. CONFUSION MATRIX
# =========================================================

print("=" * 60)
print("12. CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    random_forest_pred
)

print(cm)


plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Low Engagement",
        "High Engagement"
    ],
    yticklabels=[
        "Low Engagement",
        "High Engagement"
    ]
)

plt.title(
    "Random Forest - Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

CM_PATH = (
    EDA_DIR /
    "confusion_matrix.png"
)

plt.savefig(
    CM_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nConfusion matrix saved:"
)

print(
    CM_PATH
)

print()


# =========================================================
# 13. PREDICTED RESULTS
# =========================================================

print("=" * 60)
print("13. PREDICTED RESULTS")
print("=" * 60)

predicted_results = X_test.copy()

predicted_results["actual"] = (
    y_test.values
)

predicted_results["predicted"] = (
    random_forest_pred
)

predicted_results["probability_high"] = (
    random_forest_probability
)

predicted_results["actual_label"] = (
    predicted_results["actual"].map({
        0: "Low Engagement",
        1: "High Engagement"
    })
)

predicted_results["predicted_label"] = (
    predicted_results["predicted"].map({
        0: "Low Engagement",
        1: "High Engagement"
    })
)

PREDICTED_PATH = (
    EDA_DIR /
    "predicted_results.csv"
)

predicted_results.to_csv(
    PREDICTED_PATH,
    index=False
)

print(
    f"Predicted results saved:"
)

print(
    PREDICTED_PATH
)

print()


# =========================================================
# 14. CHECK REQUIREMENT
# =========================================================

print("=" * 60)
print("14. CHECK REQUIREMENT")
print("=" * 60)

if random_forest_accuracy >= 0.70:
    print("✓ Accuracy >= 0.70")
else:
    print("✗ Accuracy < 0.70")

if random_forest_auc >= 0.70:
    print("✓ ROC-AUC >= 0.70")
else:
    print("✗ ROC-AUC < 0.70")

if random_forest_f1 >= 0.70:
    print("✓ F1 Score >= 0.70")
else:
    print("✗ F1 Score < 0.70")

print()


# =========================================================
# 15. SAVE MODEL
# =========================================================

print("=" * 60)
print("15. SAVE MODEL")
print("=" * 60)

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR /
    "youtube_engagement_pipeline.joblib"
)

# Random Forest digunakan sebagai model utama
# untuk aplikasi Streamlit.

joblib.dump(
    random_forest_pipeline,
    MODEL_PATH
)

print(
    f"Model saved:"
)

print(
    MODEL_PATH
)

print()


# =========================================================
# 16. SAVE METADATA
# =========================================================

print("=" * 60)
print("16. SAVE METADATA")
print("=" * 60)

metadata = {

    "views_threshold": float(
        views_threshold
    ),

    "accuracy": float(
        random_forest_accuracy
    ),

    "f1_score": float(
        random_forest_f1
    ),

    "auc": float(
        random_forest_auc
    ),

    "features": features,

    "model": "Random Forest"
}

METADATA_PATH = (
    MODEL_DIR /
    "metadata.joblib"
)

joblib.dump(
    metadata,
    METADATA_PATH
)

print(
    f"Metadata saved:"
)

print(
    METADATA_PATH
)

print()


# =========================================================
# 17. SELESAI
# =========================================================

print("=" * 60)
print("17. SELESAI")
print("=" * 60)

print("\nModel utama:")
print("Random Forest")

print(
    f"\nAccuracy : "
    f"{random_forest_accuracy:.4f}"
)

print(
    f"F1 Score : "
    f"{random_forest_f1:.4f}"
)

print(
    f"ROC-AUC  : "
    f"{random_forest_auc:.4f}"
)

print("\nFile yang dihasilkan:")

print(
    f"- {COMPARISON_PATH}"
)

print(
    f"- {CM_PATH}"
)

print(
    f"- {PREDICTED_PATH}"
)

print(
    f"- {MODEL_PATH}"
)

print(
    f"- {METADATA_PATH}"
)

print("\nTraining dan evaluasi selesai!")