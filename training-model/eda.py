# ============================================================
# EDA - YouTube Videos Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. SETUP
# ============================================================

# Lokasi file CSV
BASE_DIR = Path(__file__).resolve().parent

# File dataset
DATA_PATH = BASE_DIR / "40000_yt_videos.csv"

# Folder untuk menyimpan hasil grafik
OUTPUT_DIR = BASE_DIR / "eda_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS - YOUTUBE VIDEOS")
print("=" * 70)

print(f"\nJumlah data : {df.shape[0]:,} baris")
print(f"Jumlah kolom: {df.shape[1]} kolom")


# ============================================================
# 3. CATEGORY MAPPING
# ============================================================

category_mapping = {
    1: "Film & Animation",
    2: "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    19: "Travel & Events",
    20: "Gaming",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism"
}

df["category_name"] = df["category_id"].map(category_mapping)


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing[missing > 0])

print("\nTotal missing values:", df.isnull().sum().sum())


# ============================================================
# 5. DUPLICATE
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

print("Duplicate seluruh baris :", df.duplicated().sum())
print("Duplicate video_id      :", df["video_id"].duplicated().sum())
print("Duplicate title         :", df["title"].duplicated().sum())


# ============================================================
# 6. STATISTIK
# ============================================================

numeric_columns = [
    "views",
    "likes",
    "comments",
    "duration_sec",
    "subscriber_count"
]

print("\n" + "=" * 70)
print("STATISTIK NUMERIK")
print("=" * 70)

print(df[numeric_columns].describe().T)


# ============================================================
# 7. GRAFIK 1
# DISTRIBUSI KATEGORI
# ============================================================

category_counts = df["category_name"].value_counts()

plt.figure(figsize=(12, 7))

category_counts.sort_values().plot(kind="barh")

plt.title(
    "Distribusi Video Berdasarkan Kategori",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Jumlah Video")
plt.ylabel("Kategori")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_distribusi_kategori.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. GRAFIK 2
# DISTRIBUSI VIEWS
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    np.log1p(df["views"]),
    bins=50
)

plt.title(
    "Distribusi Views Video",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("log(Views + 1)")
plt.ylabel("Jumlah Video")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_distribusi_views.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. GRAFIK 3
# DISTRIBUSI LIKES
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    np.log1p(df["likes"]),
    bins=50
)

plt.title(
    "Distribusi Likes Video",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("log(Likes + 1)")
plt.ylabel("Jumlah Video")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_distribusi_likes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. GRAFIK 4
# DISTRIBUSI COMMENTS
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    np.log1p(df["comments"]),
    bins=50
)

plt.title(
    "Distribusi Comments Video",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("log(Comments + 1)")
plt.ylabel("Jumlah Video")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_distribusi_comments.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. GRAFIK 5
# DISTRIBUSI DURASI
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["duration_sec"],
    bins=50
)

plt.title(
    "Distribusi Durasi Video",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Durasi (detik)")
plt.ylabel("Jumlah Video")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_distribusi_durasi.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. GRAFIK 6
# TOP 10 CHANNEL
# ============================================================

top_channels = (
    df["channel_name"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10, 6))

top_channels.sort_values().plot(kind="barh")

plt.title(
    "Top 10 Channel Berdasarkan Jumlah Video",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Jumlah Video")
plt.ylabel("Channel")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_top_10_channel.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. GRAFIK 7
# MEDIAN VIEWS PER KATEGORI
# ============================================================

median_views = (
    df.groupby("category_name")["views"]
    .median()
    .sort_values()
)

plt.figure(figsize=(12, 7))

median_views.plot(kind="barh")

plt.title(
    "Median Views Berdasarkan Kategori",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Median Views")
plt.ylabel("Kategori")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "07_median_views_kategori.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. GRAFIK 8
# MEDIAN LIKES PER KATEGORI
# ============================================================

median_likes = (
    df.groupby("category_name")["likes"]
    .median()
    .sort_values()
)

plt.figure(figsize=(12, 7))

median_likes.plot(kind="barh")

plt.title(
    "Median Likes Berdasarkan Kategori",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Median Likes")
plt.ylabel("Kategori")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "08_median_likes_kategori.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. GRAFIK 9
# MEDIAN COMMENTS PER KATEGORI
# ============================================================

median_comments = (
    df.groupby("category_name")["comments"]
    .median()
    .sort_values()
)

plt.figure(figsize=(12, 7))

median_comments.plot(kind="barh")

plt.title(
    "Median Comments Berdasarkan Kategori",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Median Comments")
plt.ylabel("Kategori")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "09_median_comments_kategori.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. GRAFIK 10
# SUBSCRIBER VS VIEWS
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    np.log1p(df["subscriber_count"]),
    np.log1p(df["views"]),
    alpha=0.3
)

plt.title(
    "Hubungan Subscriber dengan Views",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("log(Subscriber Count + 1)")
plt.ylabel("log(Views + 1)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "10_subscriber_vs_views.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. GRAFIK 11
# VIEWS VS LIKES
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    np.log1p(df["views"]),
    np.log1p(df["likes"]),
    alpha=0.3
)

plt.title(
    "Hubungan Views dengan Likes",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("log(Views + 1)")
plt.ylabel("log(Likes + 1)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "11_views_vs_likes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. GRAFIK 12
# CORRELATION MATRIX
# ============================================================

correlation = df[numeric_columns].corr()

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

print(correlation.round(2))

plt.figure(figsize=(9, 7))

plt.imshow(
    correlation,
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(numeric_columns)),
    numeric_columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(numeric_columns)),
    numeric_columns
)

plt.title(
    "Correlation Matrix",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "12_correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 19. TEXT ANALYSIS
# ============================================================

df["title_length"] = (
    df["title"]
    .fillna("")
    .astype(str)
    .str.len()
)

df["description_length"] = (
    df["description"]
    .fillna("")
    .astype(str)
    .str.len()
)


# ============================================================
# 20. GRAFIK 13
# TITLE LENGTH
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["title_length"],
    bins=50
)

plt.title(
    "Distribusi Panjang Judul Video",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Panjang Judul (karakter)")
plt.ylabel("Jumlah Video")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "13_panjang_title.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 21. GRAFIK 14
# DESCRIPTION LENGTH
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["description_length"],
    bins=50
)

plt.title(
    "Distribusi Panjang Description",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Panjang Description (karakter)")
plt.ylabel("Jumlah Video")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "14_panjang_description.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. DATA QUALITY
# ============================================================

print("\n" + "=" * 70)
print("DATA QUALITY CHECK")
print("=" * 70)

print("Views negatif    :", (df["views"] < 0).sum())
print("Likes negatif    :", (df["likes"] < 0).sum())
print("Comments negatif :", (df["comments"] < 0).sum())
print("Duration negatif :", (df["duration_sec"] < 0).sum())

print("Views = 0        :", (df["views"] == 0).sum())
print("Likes = 0        :", (df["likes"] == 0).sum())
print("Comments = 0      :", (df["comments"] == 0).sum())


# ============================================================
# 23. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY EDA")
print("=" * 70)

print(f"""
Jumlah video       : {len(df):,}
Jumlah kolom       : {df.shape[1]}
Jumlah kategori    : {df["category_name"].nunique()}
Jumlah channel     : {df["channel_id"].nunique():,}

Missing values     : {df.isnull().sum().sum():,}
Duplicate rows     : {df.duplicated().sum():,}

Median views       : {df["views"].median():,.0f}
Median likes       : {df["likes"].median():,.0f}
Median comments    : {df["comments"].median():,.0f}
Median duration    : {df["duration_sec"].median():,.0f} detik
""")

print("=" * 70)
print("SEMUA GRAFIK BERHASIL DISIMPAN")
print("=" * 70)

print(f"\nLokasi grafik:")
print(OUTPUT_DIR)