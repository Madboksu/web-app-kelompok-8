import streamlit as st
import whisper
import tempfile
import os
from pathlib import Path


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Subtitle Generator",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# LOAD WHISPER MODEL
# =========================================================

@st.cache_resource
def load_model(model_size):
    """
    Load Whisper model dan cache agar
    tidak di-download/load ulang setiap rerun.
    """
    return whisper.load_model(model_size)


# =========================================================
# FORMAT TIMESTAMP
# =========================================================

def format_timestamp(seconds):
    """
    Mengubah detik menjadi format timestamp SRT:

    HH:MM:SS,mmm
    """

    milliseconds = int(round(seconds * 1000))

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    secs = milliseconds // 1_000
    milliseconds %= 1_000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d},"
        f"{milliseconds:03d}"
    )


# =========================================================
# CREATE SRT
# =========================================================

def create_srt(segments):
    """
    Membuat file subtitle format SRT
    berdasarkan segment Whisper.
    """

    srt_parts = []

    for index, segment in enumerate(segments, start=1):

        start = format_timestamp(
            segment["start"]
        )

        end = format_timestamp(
            segment["end"]
        )

        text = segment["text"].strip()

        if not text:
            continue

        srt_parts.append(
            f"{index}\n"
            f"{start} --> {end}\n"
            f"{text}\n"
        )

    return "\n".join(srt_parts)


# =========================================================
# CREATE TXT
# =========================================================

def create_txt(segments):
    """
    Menggabungkan seluruh segment menjadi
    transcript TXT.
    """

    texts = []

    for segment in segments:

        text = segment["text"].strip()

        if text:
            texts.append(text)

    return " ".join(texts)


# =========================================================
# HEADER
# =========================================================

st.title("🎬 AI Subtitle Generator")

st.write(
    """
    Upload video atau audio dan biarkan AI membuat subtitle
    secara otomatis menggunakan **OpenAI Whisper Speech Recognition**.
    """
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Subtitle Settings")

    language_option = st.selectbox(
        "Bahasa Audio",
        [
            "Auto Detect",
            "Indonesian",
            "English"
        ]
    )

    model_size = st.selectbox(
        "AI Model",
        [
            "base",
            "small"
        ],
        index=0
    )

    st.divider()

    st.info(
        """
        **Model:** OpenAI Whisper

        Whisper digunakan untuk mengenali suara
        dari video/audio dan mengubahnya menjadi teks.

        Model **base** direkomendasikan untuk deployment
        karena lebih ringan dibandingkan model small.
        """
    )


# =========================================================
# UPLOAD FILE
# =========================================================

uploaded_file = st.file_uploader(
    "📁 Upload Video / Audio",
    type=[
        "mp4",
        "mkv",
        "mov",
        "avi",
        "mp3",
        "wav",
        "m4a",
        "webm"
    ],
    help="Format yang didukung: MP4, MKV, MOV, AVI, MP3, WAV, M4A, WEBM."
)


# =========================================================
# INFORMATION IF NO FILE
# =========================================================

if uploaded_file is None:

    st.info(
        """
        👆 Upload file video atau audio untuk memulai.

        **Format yang didukung:**

        MP4, MKV, MOV, AVI, MP3, WAV, M4A, dan WEBM.
        """
    )


# =========================================================
# PROCESS UPLOADED FILE
# =========================================================

else:

    # =====================================================
    # FILE INFORMATION
    # =====================================================

    st.success(
        f"File berhasil diupload: **{uploaded_file.name}**"
    )

    file_size_mb = (
        uploaded_file.size / (1024 * 1024)
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "📁 Nama File",
            uploaded_file.name
        )

    with col2:

        st.metric(
            "💾 Ukuran File",
            f"{file_size_mb:.2f} MB"
        )


    # =====================================================
    # FILE SIZE WARNING
    # =====================================================

    if file_size_mb > 200:

        st.warning(
            """
            ⚠️ File cukup besar. Proses transkripsi dapat
            membutuhkan waktu lebih lama dan resource lebih besar.
            """
        )


    # =====================================================
    # VIDEO PREVIEW
    # =====================================================

    if uploaded_file.type.startswith("video"):

        st.subheader("🎥 Preview Video")

        st.video(uploaded_file)


    st.divider()


    # =====================================================
    # GENERATE BUTTON
    # =====================================================

    generate_button = st.button(
        "🎯 Generate Subtitle",
        use_container_width=True,
        type="primary"
    )


    # =====================================================
    # GENERATE SUBTITLE
    # =====================================================

    if generate_button:

        suffix = Path(
            uploaded_file.name
        ).suffix

        temp_path = None

        try:

            # =================================================
            # SAVE TEMPORARY FILE
            # =================================================

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name


            # =================================================
            # LANGUAGE
            # =================================================

            if language_option == "Auto Detect":

                language = None

            elif language_option == "Indonesian":

                language = "id"

            else:

                language = "en"


            # =================================================
            # LOAD WHISPER
            # =================================================

            with st.spinner(
                "🤖 Memuat model Whisper..."
            ):

                model = load_model(
                    model_size
                )


            # =================================================
            # TRANSCRIPTION
            # =================================================

            with st.spinner(
                "🎙️ AI sedang membuat subtitle..."
            ):

                result = model.transcribe(
                    temp_path,
                    language=language,
                    verbose=False
                )


            # =================================================
            # GET SEGMENTS
            # =================================================

            segments = result.get(
                "segments",
                []
            )


            if not segments:

                st.warning(
                    """
                    Tidak ditemukan suara atau teks
                    yang dapat ditranskripsikan dari file.
                    """
                )

                st.stop()


            # =================================================
            # SUCCESS
            # =================================================

            st.success(
                "✅ Subtitle berhasil dibuat!"
            )


            # =================================================
            # DETECTED LANGUAGE
            # =================================================

            detected_language = result.get(
                "language",
                "unknown"
            )

            st.info(
                f"🌐 Bahasa terdeteksi: "
                f"**{detected_language}**"
            )


            # =================================================
            # CREATE OUTPUT
            # =================================================

            srt_text = create_srt(
                segments
            )

            txt_text = create_txt(
                segments
            )


            # =================================================
            # TRANSCRIPT
            # =================================================

            st.subheader(
                "📝 Hasil Transkripsi"
            )

            st.text_area(
                "Transcript",
                txt_text,
                height=300
            )


            # =================================================
            # TIMELINE
            # =================================================

            st.subheader(
                "⏱️ Subtitle Timeline"
            )

            for index, segment in enumerate(
                segments,
                start=1
            ):

                start = format_timestamp(
                    segment["start"]
                )

                end = format_timestamp(
                    segment["end"]
                )

                text = segment["text"].strip()

                if not text:
                    continue

                st.markdown(
                    f"""
                    **{index}. {start} → {end}**

                    {text}
                    """
                )

                st.divider()


            # =================================================
            # DOWNLOAD
            # =================================================

            st.subheader(
                "⬇️ Download Subtitle"
            )

            col1, col2 = st.columns(2)


            # =================================================
            # DOWNLOAD TXT
            # =================================================

            with col1:

                st.download_button(
                    label="📄 Download TXT",
                    data=txt_text,
                    file_name="subtitle.txt",
                    mime="text/plain",
                    use_container_width=True
                )


            # =================================================
            # DOWNLOAD SRT
            # =================================================

            with col2:

                st.download_button(
                    label="🎬 Download SRT",
                    data=srt_text,
                    file_name="subtitle.srt",
                    mime="application/x-subrip",
                    use_container_width=True
                )


        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except Exception as e:

            st.error(
                """
                ❌ Terjadi error saat membuat subtitle.
                """
            )

            st.code(
                str(e)
            )


        # =====================================================
        # CLEAN TEMPORARY FILE
        # =====================================================

        finally:

            if (
                temp_path is not None
                and os.path.exists(temp_path)
            ):

                try:

                    os.remove(
                        temp_path
                    )

                except Exception:

                    pass


# =========================================================
# HOW IT WORKS
# =========================================================

with st.expander(
    "ℹ️ Bagaimana AI Subtitle bekerja?"
):

    st.markdown(
        """
        ### 1. Upload

        Pengguna mengupload video atau audio.

        ### 2. Speech Recognition

        **OpenAI Whisper** mengenali suara dari
        video/audio dan mengubahnya menjadi teks.

        ### 3. Timestamp

        Setiap bagian teks diberikan timestamp
        berdasarkan waktu audio.

        ### 4. Subtitle

        Sistem menggabungkan teks dan timestamp
        menjadi format **SRT**.

        ### 5. Download

        Pengguna dapat mengunduh hasil subtitle
        dalam format `.srt` atau transcript `.txt`.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🎬 AI Subtitle Generator | "
    "Powered by OpenAI Whisper"
)