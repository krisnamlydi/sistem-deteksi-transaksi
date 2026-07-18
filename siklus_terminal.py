import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

print("="*50)
print("SISTEM DETEKSI DINI PEMBATALAN TRANSAKSI")
print("="*50, "\n")

# SIKLUS 1 - 3: Ingestion & Preprocessing
print("--- [SIKLUS 1-3] Ingestion & Preprocessing ---")
try:
    df = pd.read_csv('data_transaksi_dummy.csv')
    print("Berhasil memuat dataset data_transaksi_dummy.csv")
    print("\n5 Data Teratas:")
    print(df.head())
    
    # Feature Engineering (Pemisahan Fitur dan Target)
    X = df[['nilai_belanja', 'jumlah_cicilan', 'jumlah_item', 'frekuensi_belanja']]
    y = df['status_batal']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nData berhasil dibagi: {len(X_train)} data latih, {len(X_test)} data uji.\n")
    
except FileNotFoundError:
    print("ERROR: File data_transaksi_dummy.csv tidak ditemukan di folder ini!")
    exit()

# SIKLUS 4: Pemodelan (Modeling)
print("--- [SIKLUS 4] Pemodelan Machine Learning ---")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'model_rf_transaksi.joblib')
print("Model Random Forest berhasil dilatih dan disimpan sebagai 'model_rf_transaksi.joblib'.\n")

# SIKLUS 5: Evaluasi
print("--- [SIKLUS 5] Evaluasi Kinerja Model ---")
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
print(f"Tingkat Akurasi Model: {akurasi * 100:.2f}%\n")

# SIKLUS 6: Visualisasi
print("--- [SIKLUS 6] Visualisasi Data ---")
print("Membuka jendela grafik Confusion Matrix... (Tutup jendela grafik untuk melanjutkan ke Siklus 7)\n")
fig, ax = plt.subplots(figsize=(5,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
ax.set_title('Matriks Kebingungan (Confusion Matrix)')
ax.set_xlabel('Prediksi Sistem')
ax.set_ylabel('Data Aktual')
plt.show() # Ini akan memunculkan pop-up window

# SIKLUS 7: Data Decision (Simulasi Bisnis)
print("--- [SIKLUS 7] Simulasi Keputusan Bisnis ---")
print("Menguji model dengan data pelanggan fiktif...")

# Contoh input: Belanja 600rb, Cicilan 3 bulan, 2 Item, Frekuensi 1
data_baru = pd.DataFrame([[600000, 3, 2, 1]], 
                         columns=['nilai_belanja', 'jumlah_cicilan', 'jumlah_item', 'frekuensi_belanja'])

print("\nData Input:")
print(data_baru.to_string(index=False))

prediksi = model.predict(data_baru)

print("\nHasil Keputusan Sistem:")
if prediksi[0] == 1:
    print("🚨 STATUS: BERPOTENSI BATAL")
    print("Rekomendasi Tindakan: Berikan diskon/voucher untuk mengamankan retensi pelanggan.")
else:
    print("✅ STATUS: AMAN (SUKSES)")
    print("Rekomendasi Tindakan: Lanjutkan ke proses operasional standar.")

print("\n", "="*50)
print("EKSEKUSI PROGRAM SELESAI")
print("="*50)