import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Analisis Performa YouTube",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# PATH MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "youtube_engagement_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.joblib"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    return joblib.load(METADATA_PATH)


try:
    model = load_model()
    metadata = load_metadata()

except Exception as e:
    st.error("Model tidak dapat dimuat.")
    st.exception(e)
    st.stop()


# =========================================================
# MAPPING DATA
# =========================================================

CATEGORY_MAPPING = {
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

DAY_MAPPING = {
    0: "Senin",
    1: "Selasa",
    2: "Rabu",
    3: "Kamis",
    4: "Jumat",
    5: "Sabtu",
    6: "Minggu"
}

MONTH_MAPPING = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember"
}


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🤖 Tentang AI")

    st.write(
        """
        Sistem ini menggunakan **Random Forest** untuk memprediksi
        tingkat engagement video berdasarkan karakteristik video
        dan channel.
        """
    )

    st.divider()

    st.subheader("📌 Fitur Model")

    st.write("• Kategori video")
    st.write("• Durasi video")
    st.write("• Jumlah subscriber")
    st.write("• Jam upload")
    st.write("• Hari upload")
    st.write("• Bulan upload")

    st.divider()

    st.caption(
        "Prediksi bukan jaminan performa aktual video."
    )


# =========================================================
# TITLE
# =========================================================

st.title("📈 Analisis Performa Sebelum Upload")

st.write(
    """
    Masukkan karakteristik video dan channel kamu. AI akan menganalisis
    pola dari dataset YouTube dan memprediksi apakah video berpotensi
    memperoleh **High Engagement** atau **Low Engagement**.
    """
)

st.divider()


# =========================================================
# INPUT DATA
# =========================================================

st.header("🎥 Masukkan Informasi Video")

col1, col2 = st.columns(2)


with col1:

    category_name = st.selectbox(
        "Kategori Video",
        options=list(CATEGORY_MAPPING.values())
    )

    duration_sec = st.number_input(
        "Durasi Video (detik)",
        min_value=1,
        max_value=86400,
        value=600,
        step=1,
        help="Masukkan durasi video dalam satuan detik."
    )

    subscriber_count = st.number_input(
        "Jumlah Subscriber",
        min_value=0,
        value=10000,
        step=100,
        help="Jumlah subscriber channel saat video akan diupload."
    )


with col2:

    publish_hour = st.slider(
        "Jam Upload",
        min_value=0,
        max_value=23,
        value=18,
        format="%02d:00"
    )

    selected_day = st.selectbox(
        "Hari Upload",
        options=list(DAY_MAPPING.values())
    )

    selected_month = st.selectbox(
        "Bulan Upload",
        options=list(MONTH_MAPPING.values())
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

analyze_button = st.button(
    "🚀 Analisis Performa Video",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if analyze_button:

    # -----------------------------------------------------
    # Convert user input to model values
    # -----------------------------------------------------

    category_id = next(
        key for key, value in CATEGORY_MAPPING.items()
        if value == category_name
    )

    publish_dayofweek = next(
        key for key, value in DAY_MAPPING.items()
        if value == selected_day
    )

    publish_month = next(
        key for key, value in MONTH_MAPPING.items()
        if value == selected_month
    )


    # -----------------------------------------------------
    # Create input dataframe
    # -----------------------------------------------------

    input_data = pd.DataFrame({
        "category_id": [category_id],
        "duration_sec": [duration_sec],
        "subscriber_count": [subscriber_count],
        "publish_hour": [publish_hour],
        "publish_dayofweek": [publish_dayofweek],
        "publish_month": [publish_month]
    })


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    classes = model.classes_

    probability_dict = dict(
        zip(classes, probabilities)
    )

    low_probability = probability_dict.get(0, 0)
    high_probability = probability_dict.get(1, 0)


    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    st.header("📈 Hasil Analisis")


    if prediction == 1:

        st.success(
            "🟢 Video diprediksi memiliki **HIGH ENGAGEMENT**"
        )

        st.write(
            "Model memprediksi karakteristik video ini lebih dekat "
            "dengan pola video berengagement tinggi."
        )

    else:

        st.warning(
            "🟡 Video diprediksi memiliki **LOW ENGAGEMENT**"
        )

        st.write(
            "Model memprediksi karakteristik video ini lebih dekat "
            "dengan pola video berengagement rendah."
        )


    # =====================================================
    # PROBABILITY
    # =====================================================

    st.subheader("🎯 Probabilitas Prediksi")

    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:
        st.metric(
            "Low Engagement",
            f"{low_probability:.2%}"
        )

    with prob_col2:
        st.metric(
            "High Engagement",
            f"{high_probability:.2%}"
        )


    # =====================================================
    # PROBABILITY CHART
    # =====================================================

    st.subheader("📊 Tingkat Keyakinan Model")

    probability_df = pd.DataFrame({
        "Engagement": [
            "Low Engagement",
            "High Engagement"
        ],
        "Probability": [
            low_probability,
            high_probability
        ]
    })

    fig_probability = px.bar(
        probability_df,
        x="Engagement",
        y="Probability",
        text="Probability"
    )

    fig_probability.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside"
    )

    fig_probability.update_layout(
        width=800,
        height=400,
        margin=dict(
            l=50,
            r=50,
            t=30,
            b=50
        ),
        yaxis=dict(
            range=[0, 1],
            tickformat=".0%",
            dtick=0.1,
            fixedrange=True
        ),
        xaxis=dict(
            fixedrange=True
        )
    )

    st.plotly_chart(
        fig_probability,
        use_container_width=False,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "responsive": False
        }
    )


    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    st.subheader("📝 Ringkasan Input")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.write(
            f"**Kategori:** {category_name}"
        )

        st.write(
            f"**Durasi:** {duration_sec:,} detik"
        )

        st.write(
            f"**Subscriber:** {subscriber_count:,}"
        )

    with summary_col2:

        st.write(
            f"**Jam Upload:** {publish_hour:02d}:00"
        )

        st.write(
            f"**Hari Upload:** {selected_day}"
        )

        st.write(
            f"**Bulan Upload:** {selected_month}"
        )


    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.divider()

    st.header("📊 Feature Importance")

    st.write(
        """
        Feature importance menunjukkan seberapa besar kontribusi
        masing-masing karakteristik terhadap keputusan Random Forest
        dalam membedakan video dengan engagement tinggi dan rendah.
        """
    )


    # -----------------------------------------------------
    # Get pipeline components
    # -----------------------------------------------------

    preprocessor = model.named_steps["preprocessor"]

    classifier = model.named_steps["classifier"]


    # -----------------------------------------------------
    # Get transformed feature names
    # -----------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = classifier.feature_importances_


    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })


    # -----------------------------------------------------
    # Group One-Hot encoded features
    # -----------------------------------------------------

    def group_feature(feature_name):

        if feature_name.startswith(
            "categorical__category_id"
        ):
            return "Kategori Video"

        if feature_name.startswith(
            "categorical__publish_hour"
        ):
            return "Jam Upload"

        if feature_name.startswith(
            "categorical__publish_dayofweek"
        ):
            return "Hari Upload"

        if feature_name.startswith(
            "categorical__publish_month"
        ):
            return "Bulan Upload"

        if feature_name.startswith(
            "numeric__duration_sec"
        ):
            return "Durasi Video"

        if feature_name.startswith(
            "numeric__subscriber_count"
        ):
            return "Jumlah Subscriber"

        return feature_name


    importance_df["feature_group"] = (
        importance_df["feature"]
        .apply(group_feature)
    )


    # -----------------------------------------------------
    # Aggregate importance
    # -----------------------------------------------------

    grouped_importance = (
        importance_df
        .groupby(
            "feature_group",
            as_index=False
        )["importance"]
        .sum()
        .sort_values(
            "importance",
            ascending=True
        )
    )


    # =====================================================
    # FEATURE IMPORTANCE CHART
    # =====================================================

    fig_importance = px.bar(
        grouped_importance,
        x="importance",
        y="feature_group",
        orientation="h",
        text="importance"
    )

    fig_importance.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside"
    )

    fig_importance.update_layout(
        width=900,
        height=450,
        margin=dict(
            l=50,
            r=100,
            t=30,
            b=50
        ),
        xaxis=dict(
            range=[0, 0.7],
            tickformat=".0%",
            dtick=0.1,
            fixedrange=True
        ),
        yaxis=dict(
            fixedrange=True
        )
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=False,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "responsive": False
        }
    )


    # =====================================================
    # BUSINESS INSIGHT
    # =====================================================

    most_important_feature = (
        grouped_importance
        .sort_values(
            "importance",
            ascending=False
        )
        .iloc[0]
    )

    feature_name = (
        most_important_feature["feature_group"]
    )

    feature_value = (
        most_important_feature["importance"]
    )


    st.subheader("💡 Business Insight")

    st.write(
        f"""
        Berdasarkan model Random Forest, **{feature_name}** merupakan
        karakteristik yang paling berpengaruh dalam membedakan pola
        engagement video pada dataset, dengan kontribusi relatif
        sebesar **{feature_value:.2%}**.

        Hal ini menunjukkan bahwa karakteristik tersebut perlu
        diperhatikan oleh creator ketika merencanakan video sebelum
        melakukan upload. Namun, feature importance menunjukkan
        pengaruh terhadap keputusan model dan **bukan berarti fitur
        tersebut secara langsung menyebabkan engagement menjadi tinggi**.
        """
    )


    # =====================================================
    # FINAL NOTE
    # =====================================================

    st.info(
        """
        ⚠️ Prediksi AI merupakan estimasi berdasarkan pola pada dataset
        dan tidak menjamin performa aktual video setelah diupload.
        """
    )