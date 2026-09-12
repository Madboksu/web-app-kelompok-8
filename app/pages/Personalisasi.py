import streamlit as st
import joblib
import pandas as pd

from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Personalisasi Konten",
    page_icon="🎯",
    layout="wide"
)


# =========================================================
# PATH
# =========================================================

# Personalisasi.py berada di:
# repository/app/pages/Personalisasi.py
#
# parents[0] = pages
# parents[1] = app
# parents[2] = repository

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "app" / "40000_yt_videos.csv"

MODEL_PATH = (
    BASE_DIR
    / "app"
    / "models"
    / "youtube_engagement_pipeline.joblib"
)

METADATA_PATH = (
    BASE_DIR
    / "app"
    / "models"
    / "metadata.joblib"
)


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

    # -----------------------------------------------------
    # Convert numeric columns
    # -----------------------------------------------------

    numeric_columns = [
        "category_id",
        "duration_sec",
        "subscriber_count",
        "views",
        "likes",
        "comments"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


    # -----------------------------------------------------
    # Convert publish date
    # -----------------------------------------------------

    if "publish_date" in df.columns:

        df["publish_date"] = pd.to_datetime(
            df["publish_date"],
            errors="coerce"
        )

        # Fitur waktu publikasi
        df["publish_hour"] = (
            df["publish_date"].dt.hour
        )

        df["publish_dayofweek"] = (
            df["publish_date"].dt.dayofweek
        )

        df["publish_month"] = (
            df["publish_date"].dt.month
        )

    else:

        df["publish_hour"] = 0
        df["publish_dayofweek"] = 0
        df["publish_month"] = 0


    # -----------------------------------------------------
    # Category name
    # -----------------------------------------------------

    df["category_name"] = (
        df["category_id"]
        .map(CATEGORY_NAMES)
        .fillna("Other")
    )


    return df


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        return None

    try:

        return joblib.load(
            MODEL_PATH
        )

    except Exception as e:

        st.error(
            f"Model gagal dimuat: {e}"
        )

        return None


# =========================================================
# LOAD METADATA
# =========================================================

@st.cache_resource
def load_metadata():

    if not METADATA_PATH.exists():

        return None

    try:

        return joblib.load(
            METADATA_PATH
        )

    except Exception:

        return None


# =========================================================
# LOAD
# =========================================================

try:

    df = load_data()

except Exception as e:

    st.error(
        f"Dataset gagal dimuat: {e}"
    )

    st.stop()


model = load_model()

metadata = load_metadata()


# =========================================================
# HEADER
# =========================================================

st.title("🎯 AI Personalisasi Konten")

st.write(
    """
    Temukan konten YouTube yang lebih sesuai dengan
    preferensi kamu. Sistem akan menyaring video berdasarkan
    kategori dan durasi, kemudian menggunakan model Machine
    Learning untuk memperkirakan potensi engagement.
    """
)


# =========================================================
# CHECK MODEL
# =========================================================

if model is None:

    st.error(
        """
        Model Personalisasi tidak ditemukan.

        Pastikan file berikut tersedia:

        `app/models/youtube_engagement_pipeline.joblib`
        """
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🤖 Informasi Model")

    st.write(
        "**Model:** Random Forest"
    )

    st.write(
        "**Target:** High Engagement"
    )

    if metadata is not None:

        accuracy = metadata.get(
            "accuracy",
            None
        )

        auc = metadata.get(
            "auc",
            None
        )

        if accuracy is not None:

            st.metric(
                "Accuracy",
                f"{accuracy:.2%}"
            )

        if auc is not None:

            st.metric(
                "ROC-AUC",
                f"{auc:.2%}"
            )

    st.divider()

    st.header("💡 Cara Kerja")

    st.caption(
        """
        1. Pilih kategori.
        
        2. Tentukan durasi maksimal.
        
        3. Sistem menyaring video.
        
        4. Model memprediksi potensi engagement.
        
        5. Video diurutkan berdasarkan probabilitas.
        """
    )


# =========================================================
# USER PREFERENCES
# =========================================================

st.subheader("⚙️ Atur Preferensi Kamu")


col1, col2 = st.columns(2)


# =========================================================
# CATEGORY SELECTBOX
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

    category_filter = st.selectbox(
        "🏷️ Kategori Konten",
        category_options
    )


# =========================================================
# DURATION SLIDER
# =========================================================

with col2:

    max_duration = st.slider(
        "⏱️ Durasi Maksimal Video",
        min_value=1,
        max_value=60,
        value=15,
        step=1
    )

    st.caption(
        f"Video dengan durasi maksimal {max_duration} menit"
    )


# =========================================================
# NUMBER OF RESULTS
# =========================================================

n_results = st.selectbox(
    "📋 Jumlah konten yang ditampilkan",
    [5, 10, 15, 20],
    index=1
)


# =========================================================
# ENGAGEMENT PRIORITY
# =========================================================

engagement_priority = st.slider(
    "🔥 Prioritas Engagement",
    min_value=0,
    max_value=100,
    value=70,
    step=10
)

st.caption(
    f"""
    Prioritas engagement: **{engagement_priority}%**
    
    Semakin tinggi nilainya, semakin sistem
    memprioritaskan video dengan probabilitas
    engagement tinggi.
    """
)


# =========================================================
# BUTTON
# =========================================================

personalize_button = st.button(
    "✨ Personalisasikan Konten",
    type="primary",
    use_container_width=True
)


# =========================================================
# PERSONALIZATION
# =========================================================

if personalize_button:

    # -----------------------------------------------------
    # COPY DATA
    # -----------------------------------------------------

    results = df.copy()


    # -----------------------------------------------------
    # FILTER CATEGORY
    # -----------------------------------------------------

    if category_filter != "Semua Kategori":

        results = results[
            results["category_name"]
            == category_filter
        ]


    # -----------------------------------------------------
    # FILTER DURATION
    # -----------------------------------------------------

    max_duration_sec = (
        max_duration * 60
    )

    results = results[
        results["duration_sec"]
        <= max_duration_sec
    ]


    # -----------------------------------------------------
    # FEATURES
    # -----------------------------------------------------

    model_features = [
        "category_id",
        "duration_sec",
        "subscriber_count",
        "publish_hour",
        "publish_dayofweek",
        "publish_month"
    ]


    # -----------------------------------------------------
    # CHECK FEATURES
    # -----------------------------------------------------

    missing_features = [
        feature
        for feature in model_features
        if feature not in results.columns
    ]

    if missing_features:

        st.error(
            "Fitur model tidak ditemukan: "
            + ", ".join(missing_features)
        )

        st.stop()


    # -----------------------------------------------------
    # REMOVE MISSING VALUES
    # -----------------------------------------------------

    results = results.dropna(
        subset=model_features
    )


    # -----------------------------------------------------
    # CHECK RESULT
    # -----------------------------------------------------

    if len(results) == 0:

        st.warning(
            """
            Tidak ditemukan video yang sesuai
            dengan preferensi kamu.

            Coba pilih kategori lain atau
            naikkan durasi maksimal.
            """
        )

        st.stop()


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    X = results[
        model_features
    ]


    try:

        predictions = model.predict(X)

        probabilities = (
            model.predict_proba(X)[:, 1]
        )

    except Exception as e:

        st.error(
            f"""
            Terjadi masalah saat melakukan prediksi.

            Kemungkinan fitur yang digunakan saat
            training berbeda dengan fitur aplikasi.

            Detail error:
            {e}
            """
        )

        st.stop()


    # -----------------------------------------------------
    # SAVE PREDICTION
    # -----------------------------------------------------

    results["prediction"] = predictions

    results["engagement_probability"] = (
        probabilities
    )


    # -----------------------------------------------------
    # NORMALIZE DURATION SCORE
    # -----------------------------------------------------

    duration_score = (
        1
        -
        (
            results["duration_sec"]
            /
            max_duration_sec
        )
    )

    duration_score = duration_score.clip(
        lower=0,
        upper=1
    )


    # -----------------------------------------------------
    # ENGAGEMENT SCORE
    # -----------------------------------------------------

    engagement_score = (
        results["engagement_probability"]
    )


    # -----------------------------------------------------
    # WEIGHT
    # -----------------------------------------------------

    engagement_weight = (
        engagement_priority / 100
    )

    duration_weight = (
        1 - engagement_weight
    )


    # -----------------------------------------------------
    # PERSONALIZATION SCORE
    # -----------------------------------------------------

    results["personalization_score"] = (
        engagement_score
        *
        engagement_weight
        +
        duration_score
        *
        duration_weight
    )


    # -----------------------------------------------------
    # SORT
    # -----------------------------------------------------

    results = results.sort_values(
        "personalization_score",
        ascending=False
    )


    # -----------------------------------------------------
    # LIMIT RESULTS
    # -----------------------------------------------------

    results = results.head(
        n_results
    )


    # =====================================================
    # RESULT SUMMARY
    # =====================================================

    st.success(
        f"""
        Berhasil menemukan {len(results)}
        konten yang sesuai dengan preferensi kamu.
        """
    )


    # =====================================================
    # SUMMARY METRICS
    # =====================================================

    avg_probability = (
        results["engagement_probability"]
        .mean()
    )

    avg_duration = (
        results["duration_sec"]
        .mean()
        / 60
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "🎬 Konten",
            len(results)
        )


    with col2:

        st.metric(
            "🔥 Rata-rata Engagement",
            f"{avg_probability:.1%}"
        )


    with col3:

        st.metric(
            "⏱️ Rata-rata Durasi",
            f"{avg_duration:.1f} menit"
        )


    # =====================================================
    # RESULT TITLE
    # =====================================================

    st.subheader(
        "🎯 Hasil Personalisasi"
    )


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    for rank, (_, row) in enumerate(
        results.iterrows(),
        start=1
    ):

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
        # METRICS
        # -------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            duration_minutes = (
                row["duration_sec"] / 60
            )

            st.metric(
                "⏱️ Durasi",
                f"{duration_minutes:.1f} menit"
            )


        with col2:

            st.metric(
                "👁️ Views",
                f"{row['views']:,.0f}"
            )


        with col3:

            if "likes" in row:

                st.metric(
                    "👍 Likes",
                    f"{row['likes']:,.0f}"
                )


        with col4:

            probability = (
                row["engagement_probability"]
            )

            st.metric(
                "🔥 Engagement",
                f"{probability:.1%}"
            )


        # -------------------------------------------------
        # PROBABILITY BAR
        # -------------------------------------------------

        st.write(
            "Probabilitas High Engagement"
        )

        st.progress(
            float(
                row["engagement_probability"]
            )
        )


        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        if row["prediction"] == 1:

            st.success(
                "🔥 Model memprediksi potensi "
                "engagement tinggi."
            )

        else:

            st.info(
                "📊 Model memprediksi potensi "
                "engagement relatif lebih rendah."
            )


        # -------------------------------------------------
        # PERSONALIZATION SCORE
        # -------------------------------------------------

        st.caption(
            f"""
            Personalization Score:
            **{row['personalization_score']:.3f}**
            """
        )


        # -------------------------------------------------
        # YOUTUBE LINK
        # -------------------------------------------------

        if "video_id" in row:

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


# =========================================================
# HOW IT WORKS
# =========================================================

with st.expander(
    "ℹ️ Bagaimana Personalisasi bekerja?"
):

    st.markdown(
        """
        ### 1. Pengguna menentukan preferensi

        Pengguna memilih kategori konten dan
        durasi maksimal video.

        ### 2. Sistem melakukan filtering

        Dataset YouTube kemudian disaring berdasarkan
        preferensi tersebut.

        ### 3. Model Machine Learning melakukan prediksi

        Setiap video yang lolos filter dianalisis
        menggunakan model **Random Forest**.

        Model mempertimbangkan karakteristik video
        seperti:

        - Kategori video
        - Durasi video
        - Jumlah subscriber channel
        - Jam publikasi
        - Hari publikasi
        - Bulan publikasi

        ### 4. Sistem menghitung probabilitas

        Model menghasilkan probabilitas bahwa sebuah
        video termasuk kategori **High Engagement**.

        ### 5. Sistem melakukan ranking

        Video kemudian diurutkan berdasarkan
        **Personalization Score**.

        Score mempertimbangkan:

        - Potensi engagement
        - Kesesuaian durasi dengan preferensi pengguna

        ### 6. Hasil personalisasi

        Pengguna mendapatkan daftar video yang
        paling sesuai dengan preferensi dan
        memiliki potensi engagement tinggi.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🎯 AI Content Personalization | "
    "Powered by Machine Learning"
)
