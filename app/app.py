import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Media & Konten",
    page_icon="📱",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "40000_yt_videos.csv"

CLASSIFICATION_MODEL_PATH = (
    BASE_DIR / "models" / "youtube_engagement_pipeline.joblib"
)

CLASSIFICATION_METADATA_PATH = (
    BASE_DIR / "models" / "metadata.joblib"
)

MODERATION_MODEL_PATH = (
    BASE_DIR / "models_moderation" / "moderation_pipeline.joblib"
)

MODERATION_METADATA_PATH = (
    BASE_DIR / "models_moderation" / "moderation_metadata.joblib"
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
# LOAD YOUTUBE DATA
# =========================================================

@st.cache_data
def load_youtube_data():

    df = pd.read_csv(DATA_PATH)

    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["channel_name"] = df["channel_name"].fillna("")

    df["category_name"] = (
        df["category_id"]
        .map(CATEGORY_NAMES)
        .fillna("Other")
    )

    df["publish_date"] = pd.to_datetime(
        df["publish_date"],
        errors="coerce"
    )

    df["publish_hour"] = (
        df["publish_date"]
        .dt.hour
        .fillna(12)
        .astype(int)
    )

    df["publish_dayofweek"] = (
        df["publish_date"]
        .dt.dayofweek
        .fillna(0)
        .astype(int)
    )

    return df


# =========================================================
# RECOMMENDATION MODEL
# =========================================================

@st.cache_resource
def create_recommendation_model(text_data):

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
# LOAD CLASSIFICATION MODEL
# =========================================================

@st.cache_resource
def load_classification_model():

    if not CLASSIFICATION_MODEL_PATH.exists():
        return None, None

    model = joblib.load(
        CLASSIFICATION_MODEL_PATH
    )

    metadata = joblib.load(
        CLASSIFICATION_METADATA_PATH
    )

    return model, metadata


# =========================================================
# LOAD MODERATION MODEL
# =========================================================

@st.cache_resource
def load_moderation_model():

    if not MODERATION_MODEL_PATH.exists():
        return None, None

    model = joblib.load(
        MODERATION_MODEL_PATH
    )

    metadata = joblib.load(
        MODERATION_METADATA_PATH
    )

    return model, metadata


# =========================================================
# LOAD EVERYTHING
# =========================================================

df = load_youtube_data()

# Text untuk recommendation
df["combined_text"] = (
    df["title"].astype(str)
    + " "
    + df["description"].astype(str)
    + " "
    + df["category_name"].astype(str)
    + " "
    + df["channel_name"].astype(str)
)

vectorizer, tfidf_matrix = create_recommendation_model(
    tuple(df["combined_text"].tolist())
)

classification_model, classification_metadata = (
    load_classification_model()
)

moderation_model, moderation_metadata = (
    load_moderation_model()
)


# =========================================================
# RECOMMENDATION FUNCTION
# =========================================================

def recommend_videos(
    query,
    n_results=10,
    category_filter="Semua Kategori"
):

    query_vector = vectorizer.transform(
        [query]
    )

    similarity = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    # Ambil kandidat lebih banyak
    top_indices = np.argsort(
        similarity
    )[::-1][:500]

    results = df.iloc[
        top_indices
    ].copy()

    results["similarity"] = similarity[
        top_indices
    ]

    if category_filter != "Semua Kategori":

        results = results[
            results["category_name"]
            == category_filter
        ]

    return results.head(
        n_results
    )


# =========================================================
# PERSONALIZED FEED
# =========================================================

def create_personalized_feed(
    categories,
    min_subscribers,
    sort_preference,
    n_results
):

    feed = df.copy()

    # Filter subscriber
    feed = feed[
        feed["subscriber_count"]
        >= min_subscribers
    ]

    # Preference score
    feed["preference_score"] = np.where(
        feed["category_name"].isin(categories),
        1.0,
        0.0
    )

    # Normalisasi views
    views_log = np.log1p(
        feed["views"]
    )

    max_views = views_log.max()

    if max_views > 0:

        feed["popularity_score"] = (
            views_log / max_views
        )

    else:

        feed["popularity_score"] = 0


    # Engagement score
    engagement_raw = (
        np.log1p(feed["likes"])
        + np.log1p(feed["comments"])
    )

    max_engagement = engagement_raw.max()

    if max_engagement > 0:

        feed["engagement_score"] = (
            engagement_raw
            / max_engagement
        )

    else:

        feed["engagement_score"] = 0


    # Score utama
    feed["feed_score"] = (
        feed["preference_score"] * 0.50
        + feed["popularity_score"] * 0.25
        + feed["engagement_score"] * 0.25
    )


    if sort_preference == "Paling relevan":

        feed = feed.sort_values(
            "feed_score",
            ascending=False
        )

    elif sort_preference == "Paling populer":

        feed = feed.sort_values(
            "views",
            ascending=False
        )

    else:

        feed = feed.sort_values(
            "publish_date",
            ascending=False
        )

    return feed.head(
        n_results
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📱 Media & Konten")

menu = st.sidebar.radio(
    "Pilih fitur",
    [
        "🎯 Rekomendasi Konten",
        "🛡️ Moderasi Konten Otomatis",
        "👤 Personalisasi Feed",
        "🎬 Generasi Subtitle Otomatis"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "AI Media & Konten"
)

st.sidebar.caption(
    f"Dataset: {len(df):,} video YouTube"
)


# =========================================================
# 1. RECOMMENDATION
# =========================================================

if menu == "🎯 Rekomendasi Konten":

    st.title("🎯 Rekomendasi Konten")

    st.write(
        "Cari video YouTube berdasarkan kemiripan "
        "topik, judul, deskripsi, kategori, dan channel."
    )

    st.divider()

    query = st.text_input(
        "🔎 Apa yang ingin kamu tonton?",
        placeholder=(
            "Contoh: artificial intelligence, gaming, music..."
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        category_filter = st.selectbox(
            "Kategori",
            [
                "Semua Kategori"
            ] + list(CATEGORY_NAMES.values())
        )

    with col2:

        n_results = st.selectbox(
            "Jumlah rekomendasi",
            [5, 10, 15, 20],
            index=1
        )


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
                query,
                n_results,
                category_filter
            )

            if len(results) == 0:

                st.warning(
                    "Tidak ditemukan video."
                )

            else:

                st.success(
                    f"{len(results)} video ditemukan."
                )

                for _, row in results.iterrows():

                    with st.container():

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


# =========================================================
# 2. MODERATION
# =========================================================

elif menu == "🛡️ Moderasi Konten Otomatis":

    st.title("🛡️ Moderasi Konten Otomatis")

    st.write(
        "Masukkan komentar atau teks untuk memprediksi "
        "kategori konten menggunakan model machine learning."
    )

    if moderation_model is None:

        st.error(
            "Model moderasi tidak ditemukan."
        )

        st.code(
            "modelsModeration/moderation_pipeline.joblib"
        )

    else:

        text = st.text_area(
            "💬 Masukkan teks",
            height=180,
            placeholder="Tulis komentar yang ingin diperiksa..."
        )

        if st.button(
            "🛡️ Periksa Konten",
            type="primary",
            use_container_width=True
        ):

            if not text.strip():

                st.warning(
                    "Masukkan teks terlebih dahulu."
                )

            else:

                prediction = moderation_model.predict(
                    [text]
                )[0]

                probabilities = (
                    moderation_model
                    .predict_proba([text])[0]
                )

                classes = (
                    moderation_model
                    .classes_
                )

                best_probability = (
                    max(probabilities)
                )


                # Hasil utama
                st.subheader(
                    "Hasil Moderasi"
                )

                st.error(
                    f"Kategori: **{prediction}**"
                )

                st.metric(
                    "Confidence",
                    f"{best_probability:.2%}"
                )


                # Semua probabilitas
                st.subheader(
                    "Probabilitas Setiap Kategori"
                )

                probability_df = pd.DataFrame({
                    "Kategori": classes,
                    "Probabilitas": probabilities
                })

                probability_df = (
                    probability_df
                    .sort_values(
                        "Probabilitas",
                        ascending=False
                    )
                    .set_index("Kategori")
                )

                st.bar_chart(
                    probability_df
                )


# =========================================================
# 3. PERSONALIZED FEED
# =========================================================

elif menu == "👤 Personalisasi Feed":

    st.title("👤 Personalisasi Feed")

    st.write(
        "Atur preferensi untuk mendapatkan feed "
        "video yang lebih sesuai dengan minat pengguna."
    )

    st.divider()

    selected_categories = st.multiselect(
        "❤️ Pilih kategori yang kamu sukai",
        list(CATEGORY_NAMES.values()),
        default=[
            "Gaming",
            "Science & Technology"
        ]
    )

    min_subscribers = st.number_input(
        "Minimum subscriber channel",
        min_value=0,
        max_value=500_000_000,
        value=100_000,
        step=10_000
    )

    sort_preference = st.selectbox(
        "Prioritas feed",
        [
            "Paling relevan",
            "Paling populer",
            "Terbaru"
        ]
    )

    n_feed = st.selectbox(
        "Jumlah video",
        [5, 10, 15, 20],
        index=1
    )


    if st.button(
        "✨ Buat Feed Saya",
        type="primary",
        use_container_width=True
    ):

        if len(selected_categories) == 0:

            st.warning(
                "Pilih minimal satu kategori."
            )

        else:

            feed = create_personalized_feed(
                selected_categories,
                min_subscribers,
                sort_preference,
                n_feed
            )

            if len(feed) == 0:

                st.warning(
                    "Tidak ada video yang memenuhi filter."
                )

            else:

                st.success(
                    "Feed berhasil dipersonalisasi."
                )

                for _, row in feed.iterrows():

                    st.subheader(
                        row["title"]
                    )

                    st.write(
                        f"📺 {row['channel_name']} "
                        f"• {row['category_name']}"
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:

                        st.metric(
                            "Views",
                            f"{row['views']:,.0f}"
                        )

                    with c2:

                        st.metric(
                            "Subscribers",
                            f"{row['subscriber_count']:,.0f}"
                        )

                    with c3:

                        st.metric(
                            "Feed Score",
                            f"{row['feed_score']:.2f}"
                        )

                    video_url = (
                        "https://www.youtube.com/watch?v="
                        + str(row["video_id"])
                    )

                    st.link_button(
                        "▶️ Tonton",
                        video_url
                    )

                    st.divider()


# =========================================================
# 4. SUBTITLE
# =========================================================

elif menu == "🎬 Generasi Subtitle Otomatis":

    st.title("🎬 Generasi Subtitle Otomatis")

    st.write(
        "Upload file audio/video dan sistem akan "
        "mengubah speech menjadi subtitle."
    )

    st.info(
        "Format output: SRT"
    )

    uploaded_file = st.file_uploader(
        "📁 Upload audio/video",
        type=[
            "mp3",
            "wav",
            "m4a",
            "mp4",
            "mov",
            "mkv",
            "webm"
        ]
    )

    whisper_model_name = st.selectbox(
        "Whisper Model",
        [
            "tiny",
            "base",
            "small"
        ],
        index=1
    )


    if uploaded_file is not None:

        st.success(
            f"File dipilih: {uploaded_file.name}"
        )


        if st.button(
            "🎬 Generate Subtitle",
            type="primary",
            use_container_width=True
        ):

            try:

                import whisper

                with st.spinner(
                    "Sedang memproses audio/video..."
                ):

                    # Simpan sementara
                    temp_dir = (
                        BASE_DIR / "temp"
                    )

                    temp_dir.mkdir(
                        exist_ok=True
                    )

                    input_path = (
                        temp_dir
                        / uploaded_file.name
                    )

                    with open(
                        input_path,
                        "wb"
                    ) as f:

                        f.write(
                            uploaded_file.getbuffer()
                        )


                    # Load Whisper
                    whisper_model = whisper.load_model(
                        whisper_model_name
                    )


                    # Transcription
                    result = whisper_model.transcribe(
                        str(input_path)
                    )


                    # =================================================
                    # CREATE SRT
                    # =================================================

                    def format_timestamp(seconds):

                        hours = int(
                            seconds // 3600
                        )

                        minutes = int(
                            (seconds % 3600) // 60
                        )

                        secs = int(
                            seconds % 60
                        )

                        milliseconds = int(
                            (seconds - int(seconds))
                            * 1000
                        )

                        return (
                            f"{hours:02d}:"
                            f"{minutes:02d}:"
                            f"{secs:02d},"
                            f"{milliseconds:03d}"
                        )


                    srt_content = ""

                    for i, segment in enumerate(
                        result["segments"],
                        start=1
                    ):

                        start = format_timestamp(
                            segment["start"]
                        )

                        end = format_timestamp(
                            segment["end"]
                        )

                        text_segment = (
                            segment["text"]
                            .strip()
                        )

                        srt_content += (
                            f"{i}\n"
                            f"{start} --> {end}\n"
                            f"{text_segment}\n\n"
                        )


                    st.success(
                        "Subtitle berhasil dibuat!"
                    )

                    st.download_button(
                        label="⬇️ Download Subtitle (.srt)",
                        data=srt_content,
                        file_name=(
                            Path(
                                uploaded_file.name
                            ).stem
                            + ".srt"
                        ),
                        mime="text/plain",
                        use_container_width=True
                    )


                    st.subheader(
                        "Preview Subtitle"
                    )

                    st.text_area(
                        "SRT",
                        srt_content,
                        height=400
                    )


            except ImportError:

                st.error(
                    "Library Whisper belum terinstall."
                )

                st.code(
                    "pip install openai-whisper"
                )


            except Exception as e:

                st.error(
                    "Terjadi error saat membuat subtitle."
                )

                st.exception(e)


# =========================================================
# CLASSIFICATION INFO
# =========================================================

st.sidebar.divider()

if classification_model is not None:

    st.sidebar.subheader(
        "📊 Model Engagement"
    )

    st.sidebar.metric(
        "Accuracy",
        f"{classification_metadata['accuracy']:.3f}"
    )

    st.sidebar.metric(
        "ROC-AUC",
        f"{classification_metadata['auc']:.3f}"
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    try:

        preprocessor = (
            classification_model
            .named_steps["preprocessor"]
        )

        rf_model = (
            classification_model
            .named_steps["classifier"]
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        importances = (
            rf_model
            .feature_importances_
        )


        importance = {
            "category_id": 0.0,
            "duration_sec": 0.0,
            "subscriber_count": 0.0,
            "publish_hour": 0.0,
            "publish_dayofweek": 0.0,
            "publish_month": 0.0
        }


        for name, value in zip(
            feature_names,
            importances
        ):

            clean_name = (
                name
                .replace("categorical__", "")
                .replace("numeric__", "")
                .replace("cat__", "")
                .replace("num__", "")
            )

            if clean_name.startswith(
                "category_id_"
            ):

                importance[
                    "category_id"
                ] += float(value)

            elif clean_name.startswith(
                "publish_hour_"
            ):

                importance[
                    "publish_hour"
                ] += float(value)

            elif clean_name.startswith(
                "publish_dayofweek_"
            ):

                importance[
                    "publish_dayofweek"
                ] += float(value)

            elif clean_name.startswith(
                "publish_month_"
            ):

                importance[
                    "publish_month"
                ] += float(value)

            elif clean_name == "duration_sec":

                importance[
                    "duration_sec"
                ] += float(value)

            elif clean_name == "subscriber_count":

                importance[
                    "subscriber_count"
                ] += float(value)


        importance_df = pd.DataFrame({
            "Feature": list(
                importance.keys()
            ),
            "Importance": list(
                importance.values()
            )
        })

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
            .set_index("Feature")
        )


        st.sidebar.subheader(
            "⭐ Feature Importance"
        )

        st.sidebar.bar_chart(
            importance_df
        )


        st.sidebar.caption(
            "Subscriber count biasanya penting karena "
            "merepresentasikan ukuran audience potensial "
            "sebuah channel. Semakin besar basis subscriber, "
            "semakin besar peluang video memperoleh views. "
            "Kategori dan waktu publikasi membantu model "
            "menangkap pola minat dan perilaku penonton."
        )

    except Exception:

        st.sidebar.info(
            "Feature importance tidak dapat ditampilkan."
        )