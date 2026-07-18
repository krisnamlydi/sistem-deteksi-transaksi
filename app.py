import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# 1. Konfigurasi Halaman & Tema
st.set_page_config(page_title="Dashboard Operasional Finansial", page_icon="📊", layout="wide")

# Injeksi CSS Khusus untuk Tema dan Sidebar
st.markdown("""
    <style>
    /* Latar belakang konten utama */
    .stApp { background-color: #f4fbf4; }
    h1, h2, h3, h4 { color: #1b5e20; }
    p, div, span, label, th, td { color: #000000 !important; }
    .stButton>button { background-color: #2e7d32; color: white; border-radius: 5px; font-weight: bold; }
    
    /* Mengubah latar Sidebar menjadi Biru Muda yang profesional */
    [data-testid="stSidebar"] {
        background-color: #e3f2fd !important; 
    }
    
    /* Membuat seluruh teks di dalam Sidebar menjadi Tebal (Bold) */
    [data-testid="stSidebar"] * {
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Header Utama
st.title("📊 Sistem Deteksi Dini Pembatalan Transaksi")
st.markdown("**Dashboard Pendukung Keputusan Operasional Finansial**")
st.markdown("---")

# 3. Pengaturan Sidebar untuk Navigasi Data
with st.sidebar:
    st.header("⚙️ Konfigurasi Sistem")
    st.write("Unggah dataset untuk memulai siklus pemrosesan.")
    uploaded_file = st.file_uploader("Upload data_transaksi_dummy.csv", type="csv")
    
    st.markdown("---")
    st.info("Sistem ini menggunakan algoritma **Random Forest** untuk mendeteksi anomali pada arus kas.")

# 4. Logika Utama dengan Tabs (Antarmuka Profesional)
if uploaded_file is not None:
    # Membaca data
    df = pd.read_csv(uploaded_file)
    X = df[['nilai_belanja', 'jumlah_cicilan', 'jumlah_item', 'frekuensi_belanja']]
    y = df['status_batal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Membuat Tabs
    tab1, tab2, tab3 = st.tabs(["📂 Eksplorasi Data", "🧠 Pelatihan & Evaluasi", "🚀 Simulasi Keputusan"])
    
    with tab1:
        st.subheader("Siklus 1 - 3: Ingestion & Preprocessing")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("**Pratinjau Data Transaksi:**")
            st.table(df.head(7))
        with col2:
            st.write("**Pengecekan Missing Values:**")
            st.table(pd.DataFrame(df.isnull().sum(), columns=["Jumlah Kosong"]))
            
    with tab2:
        st.subheader("Siklus 4 - 6: Pemodelan Machine Learning")
        if st.button("Jalankan Pelatihan Model (Run Fit)"):
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, 'model_rf_transaksi.joblib')
            
            y_pred = model.predict(X_test)
            akurasi = accuracy_score(y_test, y_pred)
            
            st.success("✅ Model berhasil dilatih dan disimpan ke dalam Model Registry!")
            
            col_metric, col_chart = st.columns([1, 2])
            with col_metric:
                st.metric(label="Tingkat Akurasi Model", value=f"{akurasi * 100:.2f}%")
                st.write("Performa model siap dianalisis lebih lanjut untuk pelaporan evaluasi.")
                
            with col_chart:
                fig, ax = plt.subplots(figsize=(5,4))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
                ax.set_title('Matriks Kebingungan (Confusion Matrix)')
                ax.set_xlabel('Prediksi Sistem')
                ax.set_ylabel('Data Aktual')
                st.pyplot(fig)
                
    with tab3:
        st.subheader("Siklus 7: Inference Engine & Kebijakan Bisnis")
        if os.path.exists('model_rf_transaksi.joblib'):
            loaded_model = joblib.load('model_rf_transaksi.joblib')
            
            with st.form("form_simulasi"):
                st.write("Masukkan metrik transaksi pelanggan baru:")
                c1, c2, c3, c4 = st.columns(4)
                with c1: val_belanja = st.number_input("Nilai Belanja (Rp)", value=500000)
                with c2: val_cicilan = st.selectbox("Cicilan (Bulan)", [1, 3, 6, 12])
                with c3: val_item = st.number_input("Jumlah Item", min_value=1, value=2)
                with c4: val_frekuensi = st.number_input("Frekuensi Belanja", min_value=1, value=5)
                
                submit_btn = st.form_submit_button("Analisis Potensi Batal")
                
                if submit_btn:
                    data_baru = pd.DataFrame([[val_belanja, val_cicilan, val_item, val_frekuensi]], 
                                             columns=['nilai_belanja', 'jumlah_cicilan', 'jumlah_item', 'frekuensi_belanja'])
                    prediksi = loaded_model.predict(data_baru)
                    
                    st.markdown("### Hasil Keputusan Sistem:")
                    if prediksi[0] == 1:
                        st.error("🚨 **STATUS: BERPOTENSI BATAL**")
                        st.warning("**Rekomendasi Tindakan:** Segera alokasikan intervensi finansial (misal: penawaran potongan harga ongkos kirim) untuk mengamankan retensi pelanggan pada transaksi ini.")
                    else:
                        st.success("✅ **STATUS: AMAN (SUKSES)**")
                        st.info("**Rekomendasi Tindakan:** Lanjutkan ke proses operasional standar tanpa perlu eskalasi tambahan.")
        else:
            st.warning("Silakan latih model di tab 'Pelatihan & Evaluasi' terlebih dahulu.")

else:
    st.info("Menunggu data dimasukkan melalui panel sebelah kiri...")