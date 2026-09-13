import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Rekomendasi Konten",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Rekomendasi Konten")

st.write(
    "Temukan video YouTube yang relevan berdasarkan "
    "topik, kategori, channel, kebaruan, dan popularitas."
)


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "40000_yt_videos.csv"


# =========================================================
# CATEGORY
# =========================================================

CATEGORY_NAMES = {
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


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    # Text
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["channel_name"] = df["channel_name"].fillna("")

    # Numeric
    df["views"] = pd.to_numeric(
        df["views"],
        errors="coerce"
    ).fillna(0)

    df["likes"] = pd.to_numeric(
        df["likes"],
        errors="coerce"
    ).fillna(0)

    # Date
    df["publish_date"] = pd.to_datetime(
        df["publish_date"],
        errors="coerce"
    )

    # Category
    df["category_name"] = (
        df["category_id"]
        .map(CATEGORY_NAMES)
        .fillna("Other")
    )

    # Combined text
    df["combined_text"] = (
        df["title"].astype(str)
        + " "
        + df["description"].astype(str)
        + " "
        + df["category_name"].astype(str)
        + " "
        + df["channel_name"].astype(str)
    )

    # =====================================================
    # POPULARITY SCORE
    # =====================================================

    log_views = np.log1p(df["views"])

    min_views = log_views.min()
    max_views = log_views.max()

    if max_views > min_views:
        df["popularity_score"] = (
            (log_views - min_views)
            / (max_views - min_views)
        )
    else:
        df["popularity_score"] = 0.0

    # =====================================================
    # RECENCY SCORE
    # =====================================================

    latest_date = df["publish_date"].max()

    age_days = (
        latest_date - df["publish_date"]
    ).dt.total_seconds() / (60 * 60 * 24)

    age_days = age_days.fillna(9999).clip(lower=0)

    # Video terbaru -> mendekati 1
    # Video lama -> mendekati 0
    df["recency_score"] = 1 / (
        1 + age_days / 30
    )

    return df


# =========================================================
# TF-IDF
# =========================================================

@st.cache_resource
def create_tfidf(text_data):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(
        text_data
    )

    return vectorizer, matrix


# =========================================================
# RECOMMENDATION
# =========================================================

def recommend_videos(
    query,
    df,
    vectorizer,
    matrix,
    n_results=10,
    category_filter="Semua Kategori"
):

    # =====================================================
    # FILTER KATEGORI DAHULU
    # =====================================================

    if category_filter != "Semua Kategori":

        valid_indices = df.index[
            df["category_name"] == category_filter
        ]

        if len(valid_indices) == 0:
            return pd.DataFrame()

        candidate_matrix = matrix[valid_indices]
        candidate_df = df.loc[
            valid_indices
        ].copy()

    else:

        candidate_matrix = matrix
        candidate_df = df.copy()

    # =====================================================
    # QUERY
    # =====================================================

    query_vector = vectorizer.transform(
        [query]
    )

    # =====================================================
    # SIMILARITY
    # =====================================================

    similarity = cosine_similarity(
        query_vector,
        candidate_matrix
    ).flatten()

    candidate_df["similarity"] = similarity

    # =====================================================
    # FINAL SCORE
    # =====================================================

    candidate_df["recommendation_score"] = (
        candidate_df["similarity"] * 0.60
        + candidate_df["recency_score"] * 0.20
        + candidate_df["popularity_score"] * 0.20
    )

    # =====================================================
    # SORT
    # =====================================================

    results = candidate_df.sort_values(
        "recommendation_score",
        ascending=False
    )

    return results.head(n_results)


# =========================================================
# LOAD
# =========================================================

df = load_data()

vectorizer, tfidf_matrix = create_tfidf(
    tuple(
        df["combined_text"].tolist()
    )
)


# =========================================================
# USER INPUT
# =========================================================

query = st.text_input(
    "🔎 Apa yang ingin kamu tonton?",
    placeholder=(
        "Contoh: artificial intelligence, gaming, music..."
    )
)


col1, col2 = st.columns(2)


with col1:

    category_filter = st.selectbox(
        "🏷️ Kategori",
        ["Semua Kategori"]
        + sorted(
            df["category_name"]
            .dropna()
            .unique()
            .tolist()
        )
    )


with col2:

    n_results = st.selectbox(
        "Jumlah rekomendasi",
        [5, 10, 15, 20],
        index=1
    )


# =========================================================
# BUTTON
# =========================================================

if st.button(
    "🔍 Cari Rekomendasi",
    type="primary",
    use_container_width=True
):

    if not query.strip():

        st.warning(
            "Masukkan kata kunci terlebih dahulu."
        )

    else:

        results = recommend_videos(
            query=query,
            df=df,
            vectorizer=vectorizer,
            matrix=tfidf_matrix,
            n_results=n_results,
            category_filter=category_filter
        )

        if len(results) == 0:

            st.warning(
                "Tidak ditemukan video yang sesuai."
            )

        else:

            st.success(
                f"Ditemukan {len(results)} rekomendasi."
            )

            st.caption(
                "Ranking mempertimbangkan relevansi, "
                "kebaruan video, dan jumlah views."
            )

            # =================================================
            # DISPLAY
            # =================================================

            for rank, (_, row) in enumerate(
                results.iterrows(),
                start=1
            ):

                st.subheader(
                    f"#{rank} — {row['title']}"
                )

                st.write(
                    f"📺 **Channel:** "
                    f"{row['channel_name']}"
                )

                st.write(
                    f"🏷️ **Kategori:** "
                    f"{row['category_name']}"
                )

                if pd.notna(row["publish_date"]):

                    st.write(
                        f"📅 **Dipublikasikan:** "
                        f"{row['publish_date'].strftime('%d %B %Y')}"
                    )

                c1, c2, c3, c4 = st.columns(4)

                with c1:

                    st.metric(
                        "👁️ Views",
                        f"{row['views']:,.0f}"
                    )

                with c2:

                    st.metric(
                        "👍 Likes",
                        f"{row['likes']:,.0f}"
                    )

                with c3:

                    st.metric(
                        "🎯 Relevansi",
                        f"{row['similarity']:.1%}"
                    )

                with c4:

                    st.metric(
                        "⭐ Skor",
                        f"{row['recommendation_score']:.1%}"
                    )

                video_url = (
                    "https://www.youtube.com/watch?v="
                    + str(row["video_id"])
                )

                st.link_button(
                    "▶️ Tonton di YouTube",
                    video_url,
                    use_container_width=True
                )

                st.divider()