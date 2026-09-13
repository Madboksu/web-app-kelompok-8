import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Personalisasi Konten",
    page_icon="🎯",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🎯 AI Personalisasi Konten")

st.markdown(
    """
    Temukan video YouTube yang sesuai dengan preferensi kamu
    berdasarkan **kategori, durasi, engagement, kebaruan video,
    dan popularitas**.
    """
)


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "40000_yt_videos.csv"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "youtube_engagement_pipeline.joblib"
)

METADATA_PATH = (
    BASE_DIR
    / "models"
    / "metadata.joblib"
)


# =========================================================
# CATEGORY MAPPING
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

    # -----------------------------------------------------
    # Pastikan kolom penting tersedia
    # -----------------------------------------------------

    required_columns = [
        "video_id",
        "title",
        "category_id",
        "views",
        "likes",
        "comments",
        "duration_sec",
        "publish_date",
        "channel_name",
        "subscriber_count"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Kolom berikut tidak ditemukan dalam dataset: "
            + ", ".join(missing_columns)
        )


    # -----------------------------------------------------
    # TEXT
    # -----------------------------------------------------

    df["title"] = df["title"].fillna("Tanpa Judul")

    df["channel_name"] = (
        df["channel_name"]
        .fillna("Unknown Channel")
    )


    # -----------------------------------------------------
    # NUMERIC
    # -----------------------------------------------------

    numeric_columns = [
        "category_id",
        "views",
        "likes",
        "comments",
        "duration_sec",
        "subscriber_count"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


    df["views"] = df["views"].fillna(0)

    df["likes"] = df["likes"].fillna(0)

    df["comments"] = df["comments"].fillna(0)

    df["duration_sec"] = (
        df["duration_sec"]
        .fillna(df["duration_sec"].median())
    )

    df["subscriber_count"] = (
        df["subscriber_count"]
        .fillna(df["subscriber_count"].median())
    )


    # -----------------------------------------------------
    # DATE
    # -----------------------------------------------------

    df["publish_date"] = pd.to_datetime(
        df["publish_date"],
        errors="coerce"
    )


    # -----------------------------------------------------
    # CATEGORY NAME
    # -----------------------------------------------------

    df["category_name"] = (
        df["category_id"]
        .map(CATEGORY_NAMES)
        .fillna("Other")
    )


    # =====================================================
    # RECENCY SCORE
    # =====================================================

    latest_date = df["publish_date"].max()

    age_days = (
        latest_date - df["publish_date"]
    ).dt.total_seconds() / (60 * 60 * 24)

    age_days = (
        age_days
        .fillna(9999)
        .clip(lower=0)
    )


    # Semakin baru -> semakin mendekati 1
    #
    # 0 hari   -> 1.00
    # 30 hari  -> 0.50
    # 60 hari  -> 0.33
    # 365 hari -> sekitar 0.08
    #

    df["recency_score"] = 1 / (
        1 + age_days / 30
    )


    # =====================================================
    # POPULARITY SCORE
    # =====================================================

    # Menggunakan log agar video dengan miliaran views
    # tidak terlalu mendominasi video lainnya.

    log_views = np.log1p(
        df["views"]
    )


    min_views = log_views.min()
    max_views = log_views.max()


    if max_views > min_views:

        df["popularity_score"] = (
            (log_views - min_views)
            / (max_views - min_views)
        )

    else:

        df["popularity_score"] = 0.0


    return df


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        return None

    try:

        model = joblib.load(
            MODEL_PATH
        )

        return model

    except Exception:

        return None


# =========================================================
# LOAD METADATA
# =========================================================

@st.cache_resource
def load_metadata():

    if not METADATA_PATH.exists():

        return None

    try:

        metadata = joblib.load(
            METADATA_PATH
        )

        return metadata

    except Exception:

        return None


# =========================================================
# INITIALIZE
# =========================================================

try:

    df = load_data()

except Exception as e:

    st.error(
        f"❌ Gagal membaca dataset: {e}"
    )

    st.stop()


model = load_model()

metadata = load_metadata()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Pengaturan")

    st.subheader("🤖 Informasi Model")

    if model is not None:

        st.success(
            "Model berhasil dimuat"
        )

    else:

        st.warning(
            "Model belum tersedia"
        )


    # -----------------------------------------------------
    # METADATA MODEL
    # -----------------------------------------------------

    if metadata is not None:

        accuracy = metadata.get(
            "accuracy"
        )

        roc_auc = metadata.get(
            "roc_auc"
        )

        if accuracy is not None:

            st.metric(
                "Accuracy",
                f"{accuracy:.2%}"
            )

        if roc_auc is not None:

            st.metric(
                "ROC-AUC",
                f"{roc_auc:.2%}"
            )


    st.divider()


    st.subheader("📊 Sistem Ranking")

    st.markdown(
        """
        **50% Engagement**

        Prediksi probabilitas video
        memiliki engagement tinggi.

        **25% Kebaruan**

        Video yang lebih baru mendapatkan
        skor lebih tinggi.

        **25% Popularitas**

        Video dengan views lebih tinggi
        mendapatkan skor lebih tinggi.
        """
    )


    st.divider()


    st.subheader("ℹ️ Cara Kerja")

    st.markdown(
        """
        1. Pilih kategori video.
        2. Tentukan durasi maksimum.
        3. Sistem memprediksi engagement.
        4. Sistem menghitung skor kebaruan.
        5. Sistem menghitung skor popularitas.
        6. Semua skor digabungkan.
        7. Video dengan skor tertinggi
           ditampilkan terlebih dahulu.
        """
    )


# =========================================================
# USER PREFERENCES
# =========================================================

st.subheader("🎯 Atur Preferensi Kamu")


col1, col2, col3 = st.columns(3)


# =========================================================
# CATEGORY
# =========================================================

with col1:

    category_options = [
        "Semua Kategori"
    ] + sorted(
        df["category_name"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_category = st.selectbox(
        "🏷️ Kategori",
        category_options
    )


# =========================================================
# MAX DURATION
# =========================================================

with col2:

    max_duration_minutes = st.slider(
        "⏱️ Durasi Maksimum",
        min_value=1,
        max_value=60,
        value=15,
        step=1
    )


# =========================================================
# NUMBER RESULTS
# =========================================================

with col3:

    n_results = st.selectbox(
        "📋 Jumlah Rekomendasi",
        [5, 10, 15, 20],
        index=1
    )


# =========================================================
# BUTTON
# =========================================================

generate_button = st.button(
    "🚀 Tampilkan Rekomendasi",
    type="primary",
    use_container_width=True
)


# =========================================================
# GENERATE RECOMMENDATION
# =========================================================

if generate_button:

    # =====================================================
    # FILTER CATEGORY
    # =====================================================

    filtered_df = df.copy()


    if selected_category != "Semua Kategori":

        filtered_df = filtered_df[
            filtered_df["category_name"]
            == selected_category
        ]


    # =====================================================
    # FILTER DURATION
    # =====================================================

    max_duration_sec = (
        max_duration_minutes * 60
    )

    filtered_df = filtered_df[
        filtered_df["duration_sec"]
        <= max_duration_sec
    ]


    # =====================================================
    # CHECK DATA
    # =====================================================

    if len(filtered_df) == 0:

        st.warning(
            "⚠️ Tidak ada video yang sesuai "
            "dengan filter yang dipilih."
        )

        st.stop()


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    if model is None:

        st.error(
            """
            ❌ Model tidak ditemukan.

            Pastikan file berikut tersedia:

            `app/models/youtube_engagement_pipeline.joblib`
            """
        )

        st.stop()


    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    prediction_data = filtered_df.copy()


    prediction_data[
        "publish_hour"
    ] = prediction_data[
        "publish_date"
    ].dt.hour


    prediction_data[
        "publish_dayofweek"
    ] = prediction_data[
        "publish_date"
    ].dt.dayofweek


    prediction_data[
        "publish_month"
    ] = prediction_data[
        "publish_date"
    ].dt.month


    # =====================================================
    # MODEL FEATURES
    # =====================================================

    feature_columns = [
        "category_id",
        "duration_sec",
        "subscriber_count",
        "publish_hour",
        "publish_dayofweek",
        "publish_month"
    ]


    X = prediction_data[
        feature_columns
    ]


    # =====================================================
    # PREDICT
    # =====================================================

    try:

        probabilities = model.predict_proba(
            X
        )

        # Probability class 1
        engagement_probability = probabilities[
            :,
            1
        ]

        prediction_data[
            "engagement_probability"
        ] = engagement_probability


    except Exception as e:

        st.error(
            f"❌ Gagal melakukan prediksi: {e}"
        )

        st.stop()


    # =====================================================
    # ENGAGEMENT SCORE
    # =====================================================

    prediction_data[
        "engagement_score"
    ] = prediction_data[
        "engagement_probability"
    ]


    # =====================================================
    # DURATION SCORE
    # =====================================================

    # Video yang lebih pendek mendapatkan skor lebih tinggi.
    # Tetapi durasi tetap hanya menjadi faktor kecil.

    duration_ratio = (
        prediction_data["duration_sec"]
        / max_duration_sec
    )

    prediction_data[
        "duration_score"
    ] = (
        1 - duration_ratio
    ).clip(
        lower=0,
        upper=1
    )


    # =====================================================
    # RECENCY SCORE
    # =====================================================

    prediction_data[
        "recency_score"
    ] = prediction_data[
        "recency_score"
    ].clip(
        lower=0,
        upper=1
    )


    # =====================================================
    # POPULARITY SCORE
    # =====================================================

    prediction_data[
        "popularity_score"
    ] = prediction_data[
        "popularity_score"
    ].clip(
        lower=0,
        upper=1
    )


    # =====================================================
    # FINAL PERSONALIZATION SCORE
    # =====================================================

    #
    # Engagement : 50%
    # Recency    : 25%
    # Popularity : 25%
    #

    prediction_data[
        "personalization_score"
    ] = (
        prediction_data[
            "engagement_score"
        ] * 0.50

        +

        prediction_data[
            "recency_score"
        ] * 0.25

        +

        prediction_data[
            "popularity_score"
        ] * 0.25
    )


    # =====================================================
    # SORT
    # =====================================================

    results = prediction_data.sort_values(
        "personalization_score",
        ascending=False
    ).head(
        n_results
    )


    # =====================================================
    # RESULT SUMMARY
    # =====================================================

    st.divider()

    st.subheader(
        "✨ Rekomendasi Untuk Kamu"
    )

    st.success(
        f"""
        Ditemukan {len(filtered_df):,} video
        yang sesuai dengan filter.
        Menampilkan {len(results)} video terbaik.
        """
    )


    st.caption(
        "Ranking menggunakan kombinasi "
        "engagement, kebaruan, dan popularitas video."
    )


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    for rank, (_, row) in enumerate(
        results.iterrows(),
        start=1
    ):

        # -------------------------------------------------
        # TITLE
        # -------------------------------------------------

        st.markdown(
            f"## #{rank} — {row['title']}"
        )


        # -------------------------------------------------
        # CHANNEL & CATEGORY
        # -------------------------------------------------

        st.write(
            f"📺 **Channel:** "
            f"{row['channel_name']}"
        )

        st.write(
            f"🏷️ **Kategori:** "
            f"{row['category_name']}"
        )


        # -------------------------------------------------
        # DATE
        # -------------------------------------------------

        if pd.notna(
            row["publish_date"]
        ):

            st.write(
                f"📅 **Dipublikasikan:** "
                f"{row['publish_date'].strftime('%d %B %Y')}"
            )


        # -------------------------------------------------
        # DURATION
        # -------------------------------------------------

        duration_seconds = float(
            row["duration_sec"]
        )

        duration_minutes = (
            int(duration_seconds)
            // 60
        )

        duration_remaining_seconds = (
            int(duration_seconds)
            % 60
        )

        if duration_minutes > 0:

            duration_text = (
                f"{duration_minutes} menit "
                f"{duration_remaining_seconds} detik"
            )

        else:

            duration_text = (
                f"{duration_remaining_seconds} detik"
            )


        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "👁️ Views",
                f"{row['views']:,.0f}"
            )


        with col2:

            st.metric(
                "👍 Likes",
                f"{row['likes']:,.0f}"
            )


        with col3:

            st.metric(
                "⏱️ Durasi",
                duration_text
            )


        with col4:

            st.metric(
                "🎯 Engagement",
                f"{row['engagement_probability']:.1%}"
            )


        # -------------------------------------------------
        # SCORE
        # -------------------------------------------------

        score_col1, score_col2, score_col3 = st.columns(3)


        with score_col1:

            st.metric(
                "🔥 Engagement Score",
                f"{row['engagement_score']:.1%}"
            )


        with score_col2:

            st.metric(
                "🆕 Recency Score",
                f"{row['recency_score']:.1%}"
            )


        with score_col3:

            st.metric(
                "👁️ Popularity Score",
                f"{row['popularity_score']:.1%}"
            )


        # -------------------------------------------------
        # FINAL SCORE
        # -------------------------------------------------

        st.progress(
            float(
                row["personalization_score"]
            )
        )

        st.write(
            f"⭐ **Personalization Score:** "
            f"{row['personalization_score']:.1%}"
        )


        # -------------------------------------------------
        # ENGAGEMENT STATUS
        # -------------------------------------------------

        if row[
            "engagement_probability"
        ] >= 0.70:

            st.success(
                "🔥 Potensi engagement tinggi"
            )

        elif row[
            "engagement_probability"
        ] >= 0.50:

            st.info(
                "👍 Potensi engagement sedang"
            )

        else:

            st.warning(
                "📉 Potensi engagement relatif rendah"
            )


        # -------------------------------------------------
        # YOUTUBE LINK
        # -------------------------------------------------

        video_url = (
            "https://www.youtube.com/watch?v="
            + str(row["video_id"])
        )


        st.link_button(
            "▶️ Tonton di YouTube",
            video_url,
            use_container_width=True
        )


        # -------------------------------------------------
        # EXPLANATION
        # -------------------------------------------------

        with st.expander(
            "🔎 Lihat alasan rekomendasi"
        ):

            st.write(
                f"""
                **Mengapa video ini direkomendasikan?**

                - 🎯 Engagement Score:
                  **{row['engagement_score']:.1%}**
                - 🆕 Recency Score:
                  **{row['recency_score']:.1%}**
                - 👁️ Popularity Score:
                  **{row['popularity_score']:.1%}**

                Video kemudian mendapatkan
                **Personalization Score**
                sebesar **{row['personalization_score']:.1%}**.
                """
            )


        st.divider()


# =========================================================
# FOOTER / INFORMATION
# =========================================================

with st.expander(
    "ℹ️ Tentang Sistem Personalisasi"
):

    st.markdown(
        """
        ### Cara kerja

        Sistem menggunakan model machine learning
        untuk memprediksi kemungkinan sebuah video
        mendapatkan engagement tinggi.

        Fitur yang digunakan model adalah:

        - Category ID
        - Durasi video
        - Jumlah subscriber channel
        - Jam publikasi
        - Hari publikasi
        - Bulan publikasi

        Setelah probabilitas engagement diperoleh,
        sistem melakukan **ranking tambahan** berdasarkan:

        **50% Engagement + 25% Kebaruan + 25% Popularitas**

        Skor popularitas menggunakan transformasi log
        terhadap jumlah views sehingga perbedaan ekstrem
        antara video dengan ribuan dan miliaran views
        tidak terlalu mendominasi ranking.

        Sementara itu, skor kebaruan memberikan nilai
        lebih tinggi kepada video yang lebih dekat dengan
        tanggal terbaru dalam dataset.
        """
    )