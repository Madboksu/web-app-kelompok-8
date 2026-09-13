import streamlit as st

st.set_page_config(
    page_title="Media & Konten",
    page_icon="📱",
    layout="wide"
)

# =========================
# HEADER
# =========================

st.title("📱 Media & Konten")
st.subheader("Platform AI untuk membantu mengelola dan menemukan konten digital")

st.write(
    "Gunakan berbagai fitur AI untuk menemukan konten yang relevan, "
    "mengecek komentar, mendapatkan rekomendasi personal, "
    "dan membuat subtitle secara otomatis."
)

st.divider()

# =========================
# PETUNJUK
# =========================

st.markdown("### 🚀 Mulai dari sini")

st.info(
    "Pilih salah satu fitur pada menu di sebelah kiri untuk mulai menggunakan aplikasi."
)

# =========================
# FITUR
# =========================

st.markdown("### ✨ Fitur yang tersedia")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🎯 Rekomendasi Konten")
    st.write(
        "Temukan video YouTube yang relevan berdasarkan "
        "topik atau kata kunci yang kamu masukkan."
    )

    st.markdown("#### 🛡️ Moderasi Komentar")
    st.write(
        "Analisis komentar untuk mendeteksi berbagai jenis "
        "konten yang berpotensi berbahaya."
    )

with col2:
    st.markdown("#### 👤 Personalisasi Feed")
    st.write(
        "Dapatkan rekomendasi konten berdasarkan preferensi "
        "kategori dan karakteristik video."
    )

    st.markdown("#### 🎬 AI Subtitle")
    st.write(
        "Buat subtitle secara otomatis dari file video atau audio "
        "menggunakan AI."
    )

st.divider()

# =========================
# CARA MENGGUNAKAN
# =========================

st.markdown("### 📖 Cara menggunakan")

st.markdown("""
1. **Pilih fitur** yang ingin digunakan dari sidebar.
2. **Masukkan data** sesuai petunjuk pada halaman tersebut.
3. **Jalankan fitur AI** menggunakan tombol yang tersedia.
4. **Lihat hasil** prediksi atau rekomendasi yang diberikan.
""")

st.success(
    "💡 Tips: Kalau baru pertama kali menggunakan aplikasi, "
    "kamu bisa mulai dari **🎯 Rekomendasi Konten**."
)