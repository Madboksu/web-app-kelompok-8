import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================

st.title("🎯 Rekomendasi Konten")

st.write(
    "Temukan video YouTube yang relevan berdasarkan "
    "judul, deskripsi, kategori, dan channel."
)

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

    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["channel_name"] = df["channel_name"].fillna("")

    df["category_name"] = (
        df["category_id"]
        .map(CATEGORY_NAMES)
        .fillna("Other")
    )

    # Gabungkan informasi untuk recommendation
    df["combined_text"] = (
        df["title"].astype(str)
        + " "
        + df["description"].astype(str)
        + " "
        + df["category_name"].astype(str)
        + " "
        + df["channel_name"].astype(str)
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

    matrix = vectorizer.fit_transform(text_data)

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

    # Ubah query user menjadi TF-IDF
    query_vector = vectorizer.transform(
        [query]
    )

    # Hitung cosine similarity
    similarity = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    # Ambil kandidat terbaik
    top_indices = np.argsort(
        similarity
    )[::-1][:500]

    results = df.iloc[
        top_indices
    ].copy()

    results["similarity"] = similarity[
        top_indices
    ]

    # Filter kategori
    if category_filter != "Semua Kategori":

        results = results[
            results["category_name"]
            == category_filter
        ]

    return results.head(
        n_results
    )


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
        [
            "Semua Kategori"
        ] + list(
            CATEGORY_NAMES.values()
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

            # =============================================
            # DISPLAY RESULT
            # =============================================

            for _, row in results.iterrows():

                st.subheader(
                    row["title"]
                )

                st.write(
                    f"📺 **Channel:** "
                    f"{row['channel_name']}"
                )

                st.write(
                    f"🏷️ **Kategori:** "
                    f"{row['category_name']}"
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Views",
                        f"{row['views']:,.0f}"
                    )

                with c2:

                    st.metric(
                        "Likes",
                        f"{row['likes']:,.0f}"
                    )

                with c3:

                    st.metric(
                        "Similarity",
                        f"{row['similarity']:.1%}"
                    )

                video_url = (
                    "https://www.youtube.com/watch?v="
                    + str(row["video_id"])
                )

                st.link_button(
                    "▶️ Tonton di YouTube",
                    video_url
                )

                st.divider()