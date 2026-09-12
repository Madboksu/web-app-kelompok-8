import streamlit as st
import tempfile
import os

from pathlib import Path
from faster_whisper import WhisperModel


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Subtitle Generator",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# LOAD WHISPER MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

    return model


# ============================================================
# HELPER - FORMAT WAKTU SRT
# ============================================================

def format_timestamp(seconds):

    hours = int(seconds // 3600)

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(seconds % 60)

    milliseconds = int(
        (seconds - int(seconds)) * 1000
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


# ============================================================
# CREATE SRT
# ============================================================

def create_srt(segments):

    srt_text = ""

    for i, segment in enumerate(
        segments,
        start=1
    ):

        start = format_timestamp(
            segment.start
        )

        end = format_timestamp(
            segment.end
        )

        text = segment.text.strip()

        srt_text += (
            f"{i}\n"
            f"{start} --> {end}\n"
            f"{text}\n\n"
        )

    return srt_text


# ============================================================
# CREATE TXT
# ============================================================

def create_txt(segments):

    text = ""

    for segment in segments:

        text += segment.text.strip() + " "

    return text.strip()


# ============================================================
# HEADER
# ============================================================

st.title("🎬 AI Subtitle Generator")

st.write(
    """
    Upload video atau audio dan biarkan AI membuat subtitle
    secara otomatis menggunakan **Whisper Speech Recognition**.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Subtitle Settings")

language_option = st.sidebar.selectbox(
    "Bahasa Audio",
    [
        "Auto Detect",
        "Indonesian",
        "English"
    ]
)

model_size = st.sidebar.selectbox(
    "AI Model",
    [
        "base",
        "small"
    ]
)

st.sidebar.info(
    """
    **Model:** Faster-Whisper

    Model digunakan untuk mengubah
    suara dalam video/audio menjadi teks
    secara otomatis.
    """
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Video / Audio",
    type=[
        "mp4",
        "mkv",
        "mov",
        "avi",
        "mp3",
        "wav",
        "m4a",
        "webm"
    ]
)


# ============================================================
# PROCESS
# ============================================================

if uploaded_file is not None:

    st.success(
        f"File berhasil diupload: "
        f"**{uploaded_file.name}**"
    )

    # Informasi file
    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Nama File",
            uploaded_file.name
        )

    with col2:

        file_size_mb = (
            uploaded_file.size
            / (1024 * 1024)
        )

        st.metric(
            "Ukuran File",
            f"{file_size_mb:.2f} MB"
        )


    # Preview video
    if uploaded_file.type.startswith("video"):

        st.video(
            uploaded_file
        )


    st.divider()


    # ========================================================
    # BUTTON
    # ========================================================

    if st.button(
        "🎯 Generate Subtitle",
        use_container_width=True
    ):

        # Simpan file sementara
        suffix = Path(
            uploaded_file.name
        ).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.read()
            )

            temp_path = temp_file.name


        try:

            # ==================================================
            # LOAD MODEL
            # ==================================================

            with st.spinner(
                "Memuat AI Whisper..."
            ):

                model = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type="int8"
                )


            # ==================================================
            # LANGUAGE
            # ==================================================

            if language_option == "Auto Detect":

                language = None

            elif language_option == "Indonesian":

                language = "id"

            else:

                language = "en"


            # ==================================================
            # TRANSCRIBE
            # ==================================================

            with st.spinner(
                "🎙️ AI sedang mentranskripsi audio..."
            ):

                segments, info = model.transcribe(
                    temp_path,
                    language=language,
                    beam_size=5
                )

                segments = list(segments)


            # ==================================================
            # RESULT
            # ==================================================

            st.success(
                "✅ Subtitle berhasil dibuat!"
            )


            # ==================================================
            # DETECTED LANGUAGE
            # ==================================================

            st.info(
                f"Bahasa terdeteksi: "
                f"**{info.language}**"
            )


            # ==================================================
            # CREATE FILES
            # ==================================================

            srt_text = create_srt(
                segments
            )

            txt_text = create_txt(
                segments
            )


            # ==================================================
            # PREVIEW
            # ==================================================

            st.subheader(
                "📝 Hasil Transkripsi"
            )

            st.text_area(
                "Transcript",
                txt_text,
                height=300
            )


            # ==================================================
            # SUBTITLE TIMELINE
            # ==================================================

            st.subheader(
                "⏱️ Subtitle Timeline"
            )

            for i, segment in enumerate(
                segments,
                start=1
            ):

                start = format_timestamp(
                    segment.start
                )

                end = format_timestamp(
                    segment.end
                )

                st.markdown(
                    f"""
                    **{i}. {start} → {end}**

                    {segment.text.strip()}
                    """
                )

                st.divider()


            # ==================================================
            # DOWNLOAD
            # ==================================================

            st.subheader(
                "⬇️ Download Subtitle"
            )

            col1, col2 = st.columns(2)


            with col1:

                st.download_button(
                    label="📄 Download TXT",
                    data=txt_text,
                    file_name="subtitle.txt",
                    mime="text/plain",
                    use_container_width=True
                )


            with col2:

                st.download_button(
                    label="🎬 Download SRT",
                    data=srt_text,
                    file_name="subtitle.srt",
                    mime="text/plain",
                    use_container_width=True
                )


        except Exception as e:

            st.error(
                "Terjadi error saat membuat subtitle."
            )

            st.code(
                str(e)
            )


        finally:

            # Hapus temporary file
            if os.path.exists(temp_path):

                os.remove(temp_path)


# ============================================================
# INFORMATION
# ============================================================

else:

    st.info(
        """
        👆 Upload file video atau audio untuk memulai.

        **Format yang didukung:**
        MP4, MKV, MOV, AVI, MP3, WAV, M4A, dan WEBM.
        """
    )


# ============================================================
# HOW IT WORKS
# ============================================================

with st.expander(
    "ℹ️ Bagaimana AI Subtitle bekerja?"
):

    st.markdown(
        """
        ### 1. Upload

        Pengguna mengupload video atau audio.

        ### 2. Audio Processing

        Sistem mengambil audio dari file yang diberikan.

        ### 3. Speech Recognition

        **Faster-Whisper** digunakan untuk mengenali
        suara dan mengubahnya menjadi teks.

        ### 4. Timestamp

        Setiap bagian teks diberikan timestamp
        berdasarkan waktu audio.

        ### 5. Subtitle

        Sistem menggabungkan teks dan timestamp
        menjadi format **SRT**.

        ### 6. Download

        Pengguna dapat mengunduh hasil subtitle
        dalam format `.srt` atau transcript `.txt`.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "🎬 AI Subtitle Generator | "
    "Powered by Faster-Whisper"
)
