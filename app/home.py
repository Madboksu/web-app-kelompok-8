import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Analisis Performa YouTube",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# HOME PAGE
# =========================================================

def home_page():

    # =====================================================
    # HEADER
    # =====================================================

    st.title("📊 AI Analisis & Prediksi Performa Video YouTube")

    st.subheader(
        "Prediksi potensi engagement video sebelum di-upload"
    )

    st.write(
        """
        Aplikasi ini menggunakan Machine Learning untuk membantu
        content creator memperkirakan apakah video yang akan dibuat
        berpotensi mendapatkan **High Engagement** atau **Low Engagement**.
        """
    )

    st.divider()


    # =====================================================
    # INTRO
    # =====================================================

    st.header("🚀 Mulai Analisis")

    st.info(
        """
        Pilih **Analisis Performa** pada menu di sebelah kiri,
        kemudian masukkan karakteristik video yang ingin dianalisis.
        """
    )


    # =====================================================
    # FITUR UTAMA
    # =====================================================

    st.header("✨ Fitur Utama")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🤖 Prediksi Performa")

        st.write(
            """
            Model AI memprediksi apakah video termasuk
            **High Engagement** atau **Low Engagement**
            berdasarkan karakteristik video.
            """
        )


    with col2:

        st.subheader("📈 Probability & Feature Importance")

        st.write(
            """
            Aplikasi menampilkan probabilitas hasil prediksi
            serta fitur yang paling berpengaruh terhadap
            keputusan model.
            """
        )


    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    st.divider()

    st.header("🧠 Model Machine Learning")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model Utama",
            "Random Forest"
        )

    with col2:

        st.metric(
            "Accuracy",
            "73.32%"
        )

    with col3:

        st.metric(
            "ROC-AUC",
            "81.31%"
        )


    # =====================================================
    # CARA KERJA
    # =====================================================

    st.divider()

    st.header("🔍 Cara Kerja")

    step1, step2, step3 = st.columns(3)

    with step1:

        st.subheader("1️⃣ Input")

        st.write(
            """
            Masukkan kategori, durasi video,
            jumlah subscriber, dan waktu publikasi.
            """
        )


    with step2:

        st.subheader("2️⃣ Analisis AI")

        st.write(
            """
            Model Random Forest menganalisis karakteristik
            video berdasarkan pola dari dataset.
            """
        )


    with step3:

        st.subheader("3️⃣ Hasil")

        st.write(
            """
            Dapatkan prediksi engagement,
            probabilitas, dan insight fitur yang berpengaruh.
            """
        )


    # =====================================================
    # FOOTER
    # =====================================================

    st.divider()

    st.caption(
        "AI Analisis & Prediksi Performa Video YouTube "
        "— Machine Learning Project"
    )


# =========================================================
# CUSTOM NAVIGATION
# =========================================================

pg = st.navigation(
    [
        st.Page(
            home_page,
            title="Home",
            icon="🏠"
        ),

        st.Page(
            "pages/Analisis_Performa.py",
            title="Analisis Performa",
            icon="📊"
        )
    ]
)

pg.run()