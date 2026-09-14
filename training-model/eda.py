import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "40000_yt_videos.csv"
OUTPUT_DIR = "eda_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET INFO")
print("=" * 60)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 2. CEK DATA
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

print("Duplicate rows:", df.duplicated().sum())


# ============================================================
# 3. DATA CLEANING
# ============================================================

df["duration_sec"] = pd.to_numeric(
    df["duration_sec"],
    errors="coerce"
)

df["subscriber_count"] = pd.to_numeric(
    df["subscriber_count"],
    errors="coerce"
)

df["views"] = pd.to_numeric(
    df["views"],
    errors="coerce"
)

df["likes"] = pd.to_numeric(
    df["likes"],
    errors="coerce"
)

df["comments"] = pd.to_numeric(
    df["comments"],
    errors="coerce"
)

df["publish_date"] = pd.to_datetime(
    df["publish_date"],
    errors="coerce"
)

# Isi missing duration dengan median
df["duration_sec"] = df["duration_sec"].fillna(
    df["duration_sec"].median()
)

# Isi missing subscriber dengan median
df["subscriber_count"] = df["subscriber_count"].fillna(
    df["subscriber_count"].median()
)

# Hapus baris jika kolom penting masih kosong
df = df.dropna(
    subset=[
        "views",
        "likes",
        "comments",
        "publish_date"
    ]
)

print("\nDataset setelah cleaning:", df.shape)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

df["publish_hour"] = df["publish_date"].dt.hour

df["publish_dayofweek"] = df["publish_date"].dt.dayofweek

df["publish_month"] = df["publish_date"].dt.month


# ============================================================
# 5. MEMBUAT TARGET HIGH ENGAGEMENT
# ============================================================

views_threshold = df["views"].median()

df["high_engagement"] = (
    df["views"] >= views_threshold
).astype(int)

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print("Median views:", views_threshold)

print(
    df["high_engagement"].value_counts()
)


# ============================================================
# 6. DISTRIBUSI KATEGORI
# ============================================================

plt.figure(figsize=(10, 6))

category_counts = df["category_id"].value_counts().sort_index()

category_counts.plot(kind="bar")

plt.title("Distribusi Video Berdasarkan Kategori")
plt.xlabel("Category ID")
plt.ylabel("Jumlah Video")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_distribusi_kategori.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. DISTRIBUSI VIEWS
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["views"],
    bins=50
)

plt.title("Distribusi Views")
plt.xlabel("Views")
plt.ylabel("Jumlah Video")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_distribusi_views.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. DISTRIBUSI LIKES
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["likes"],
    bins=50
)

plt.title("Distribusi Likes")
plt.xlabel("Likes")
plt.ylabel("Jumlah Video")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_distribusi_likes.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. DISTRIBUSI COMMENTS
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["comments"],
    bins=50
)

plt.title("Distribusi Comments")
plt.xlabel("Comments")
plt.ylabel("Jumlah Video")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_distribusi_comments.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. DISTRIBUSI DURASI
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["duration_sec"],
    bins=50
)

plt.title("Distribusi Durasi Video")
plt.xlabel("Durasi (detik)")
plt.ylabel("Jumlah Video")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_distribusi_durasi.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. TOP 10 CHANNEL
# ============================================================

plt.figure(figsize=(10, 6))

top_channels = (
    df["channel_name"]
    .value_counts()
    .head(10)
    .sort_values()
)

top_channels.plot(
    kind="barh"
)

plt.title("Top 10 Channel Berdasarkan Jumlah Video")
plt.xlabel("Jumlah Video")
plt.ylabel("Channel")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_top_10_channel.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. MEDIAN VIEWS PER CATEGORY
# ============================================================

median_views_category = (
    df.groupby("category_id")["views"]
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

median_views_category.plot(
    kind="bar"
)

plt.title("Median Views per Category")
plt.xlabel("Category ID")
plt.ylabel("Median Views")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "07_median_views_kategori.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. MEDIAN LIKES PER CATEGORY
# ============================================================

median_likes_category = (
    df.groupby("category_id")["likes"]
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

median_likes_category.plot(
    kind="bar"
)

plt.title("Median Likes per Category")
plt.xlabel("Category ID")
plt.ylabel("Median Likes")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "08_median_likes_kategori.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. MEDIAN COMMENTS PER CATEGORY
# ============================================================

median_comments_category = (
    df.groupby("category_id")["comments"]
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

median_comments_category.plot(
    kind="bar"
)

plt.title("Median Comments per Category")
plt.xlabel("Category ID")
plt.ylabel("Median Comments")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "09_median_comments_kategori.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. SUBSCRIBER VS VIEWS
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["subscriber_count"],
    df["views"],
    alpha=0.3
)

plt.title("Subscriber Count vs Views")
plt.xlabel("Subscriber Count")
plt.ylabel("Views")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "10_subscriber_vs_views.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. VIEWS VS LIKES
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["views"],
    df["likes"],
    alpha=0.3
)

plt.title("Views vs Likes")
plt.xlabel("Views")
plt.ylabel("Likes")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "11_views_vs_likes.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. CORRELATION MATRIX / HEATMAP
# ============================================================

correlation_columns = [
    "duration_sec",
    "subscriber_count",
    "publish_hour",
    "publish_dayofweek",
    "publish_month",
    "high_engagement"
]

correlation_matrix = df[
    correlation_columns
].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,          # MENAMPILKAN ANGKA
    fmt=".2f",           # 2 ANGKA DI BELAKANG KOMA
    cmap="coolwarm",
    linewidths=0.5,
    square=True
)

plt.title(
    "Correlation Matrix - YouTube Features"
)

plt.xlabel("")
plt.ylabel("")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "12_correlation_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. PANJANG TITLE
# ============================================================

df["title_length"] = (
    df["title"]
    .fillna("")
    .astype(str)
    .str.len()
)

plt.figure(figsize=(10, 6))

plt.hist(
    df["title_length"],
    bins=50
)

plt.title("Distribusi Panjang Judul Video")
plt.xlabel("Panjang Judul (karakter)")
plt.ylabel("Jumlah Video")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "13_panjang_title.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 19. PANJANG DESCRIPTION
# ============================================================

df["description_length"] = (
    df["description"]
    .fillna("")
    .astype(str)
    .str.len()
)

plt.figure(figsize=(10, 6))

plt.hist(
    df["description_length"],
    bins=50
)

plt.title("Distribusi Panjang Description")
plt.xlabel("Panjang Description (karakter)")
plt.ylabel("Jumlah Video")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "14_panjang_description.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 20. STATISTIK DESKRIPTIF
# ============================================================

numeric_columns = [
    "duration_sec",
    "subscriber_count",
    "views",
    "likes",
    "comments",
    "publish_hour",
    "publish_dayofweek",
    "publish_month",
    "high_engagement"
]

classification_report = df[
    numeric_columns
].describe()

classification_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    )
)

print("\n" + "=" * 60)
print("EDA SELESAI")
print("=" * 60)

print(
    f"Semua hasil EDA disimpan di folder: {OUTPUT_DIR}"
)

print("\nFile yang dibuat:")

for file in sorted(os.listdir(OUTPUT_DIR)):
    print("-", file)