import pandas as pd
import numpy as np

# Menentukan jumlah baris data transaksi
jumlah_data = 1000

# Membuat data acak (sintetis)
data = {
    'nilai_belanja': np.random.randint(50000, 5000000, size=jumlah_data),
    'jumlah_cicilan': np.random.choice([1, 3, 6, 12], size=jumlah_data),
    'jumlah_item': np.random.randint(1, 10, size=jumlah_data),
    'frekuensi_belanja': np.random.randint(1, 50, size=jumlah_data),
    # Mensimulasikan status batal (0 = Selesai, 1 = Batal)
    # Asumsi: transaksi dibatalkan sekitar 20% dari total data
    'status_batal': np.random.choice([0, 1], p=[0.8, 0.2], size=jumlah_data)
}

# Mengubah menjadi DataFrame
df = pd.DataFrame(data)

# Menyimpan ke dalam file CSV
df.to_csv('data_transaksi_dummy.csv', index=False)
print("File data_transaksi_dummy.csv berhasil dibuat!")
