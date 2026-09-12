import streamlit as st
import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# SETUP
# ============================================================

# Struktur folder:
# app/
# ├── app.py
# ├── models_moderation/
# │   ├── moderation_pipeline.joblib
# │   └── moderation_metadata.joblib
# └── pages/
#     └── Moderasi.py   <- file ini

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models_moderation"

MODEL_PATH = MODEL_DIR / "moderation_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "moderation_metadata.joblib"

st.set_page_config(page_title="Moderasi Komentar", page_icon="🛡️")


# ============================================================
# LOAD MODEL & METADATA
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_metadata():
    if not METADATA_PATH.exists():
        return None
    return joblib.load(METADATA_PATH)


model = load_model()
metadata = load_metadata()


# ============================================================
# PAGE
# ============================================================

st.title("🛡️ AI Content Moderation")

st.write(
    """
    Masukkan komentar untuk memeriksa kategori toxicity-nya.
    Model akan mengklasifikasikan komentar ke salah satu label
    berikut beserta tingkat keyakinannya.
    """
)

if model is None:
    st.error(
        f"Model tidak ditemukan di `{MODEL_PATH}`.\n\n"
        "Jalankan `python trainmoderation.py` di folder `training-model/`, "
        "lalu copy hasil `moderation_pipeline.joblib` "
        "(dan `moderation_metadata.joblib`) ke folder `app/models_moderation/`."
    )
    st.stop()


# ============================================================
# INFO MODEL (opsional, dari metadata)
# ============================================================

if metadata is not None:
    with st.sidebar:
        st.header("🛡️ Info Model")
        st.write(f"**Model:** {metadata.get('model', '-')}")
        st.write(f"**Vectorizer:** {metadata.get('vectorizer', '-')}")
        st.write(f"**Accuracy:** {metadata.get('accuracy', 0):.2%}")
        st.write(f"**ROC-AUC:** {metadata.get('auc', 0):.2%}")
        st.divider()
        st.caption("AI Content Moderation")
else:
    with st.sidebar:
        st.header("🛡️ Moderation Model")
        st.write("**TF-IDF + Logistic Regression**")
        st.caption("AI Content Moderation")


# ============================================================
# INPUT
# ============================================================

comment = st.text_area(
    "Masukkan komentar:",
    height=180,
    placeholder="Tulis komentar di sini...",
)

if st.button("🔍 Cek Komentar", use_container_width=True):

    if comment.strip() == "":
        st.warning("Silakan masukkan komentar terlebih dahulu.")
    else:
        prediction = model.predict([comment])[0]
        probabilities = model.predict_proba([comment])[0]
        classes = model.classes_

        st.subheader("Hasil Moderasi")

        # Label utama yang dianggap "aman"
        safe_labels = {"non_toxic", "not_toxic", "clean", "neutral", "none"}

        if str(prediction).strip().lower() in safe_labels:
            st.success(f"✅ Komentar terindikasi: **{prediction}**")
        else:
            st.error(f"⚠️ Komentar terindikasi: **{prediction}**")

        st.write("**Detail probabilitas per kategori:**")

        prob_df = pd.DataFrame({
            "Kategori": classes,
            "Probabilitas": probabilities,
        }).sort_values("Probabilitas", ascending=False)

        for _, row in prob_df.iterrows():
            st.write(f"{row['Kategori']}: {row['Probabilitas']:.2%}")
            st.progress(float(row["Probabilitas"]))


# ============================================================
# CATATAN
# ============================================================

st.divider()
st.caption(
    "Model mengklasifikasikan komentar ke salah satu kategori "
    "toxicity yang dipelajari dari data training."
)
