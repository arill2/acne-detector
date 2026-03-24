# 🔬 Advanced Skin Acne Scanner v2.0

> Analisis kulit wajah secara real-time menggunakan computer vision — mendeteksi jerawat, mengklasifikasi zona wajah, dan memberikan rekomendasi perawatan kulit.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_API-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## ✨ Fitur Utama

- **Multi-method Detection** — Menggabungkan 4 metode deteksi sekaligus:
  - Deteksi kemerahan berbasis HSV (redness detection)
  - Analisis tekstur menggunakan Laplacian
  - Deteksi bintik gelap (dark spots / hiperpigmentasi)
  - Analisis anomali warna via LAB color space (a-channel)
- **Zone Analysis** — Analisis per zona wajah: Dahi, Pipi Kiri/Kanan, Hidung, Dagu, Rahang
- **Temporal Smoothing** — Hasil deteksi dihaluskan dengan averaging 15 frame agar tidak "goyang"
- **Skin Health Score** — Skor kesehatan kulit 0–100 dengan indikator visual
- **Rekomendasi Skincare** — Saran perawatan otomatis sesuai tingkat keparahan dan zona bermasalah
- **UI Real-time Premium** — Scan line animasi, pulsing markers, panel info lengkap
- **Face Mesh Overlay** — Toggle visualisasi 468 titik landmark wajah (tekan `M`)
- **Screenshot** — Simpan hasil scan kapan saja (tekan `S`)

---

## 📸 Demo

```
Tekan M untuk toggle face mesh
Tekan S untuk simpan screenshot
Tekan Q untuk keluar
```

Tampilan aplikasi menampilkan:
- Bounding box wajah dengan targeting frame
- Marker jerawat berwarna (kuning = ringan, oranye = sedang, merah = parah)
- Panel kanan: skor kesehatan, zone analysis bar, rekomendasi skincare

---

## 🛠️ Instalasi

### 1. Clone repository

```bash
git clone https://github.com/username/skin-acne-scanner.git
cd skin-acne-scanner
```

### 2. Buat virtual environment (opsional tapi disarankan)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi

```bash
python acne_scanner.py
```

> Model MediaPipe Face Landmarker (~29MB) akan otomatis diunduh saat pertama kali dijalankan.

---

## 📦 Dependencies

| Package | Versi | Kegunaan |
|---|---|---|
| `opencv-python` | ≥ 4.8 | Pemrosesan gambar & UI |
| `mediapipe` | ≥ 0.10 | Face landmark detection |
| `numpy` | ≥ 1.24 | Operasi numerik & array |

---

## ⚙️ Konfigurasi

Semua parameter dapat diatur di class `Config` dalam `acne_scanner.py`:

```python
class Config:
    CAMERA_INDEX = 0          # Index kamera (0 = default)
    CAMERA_WIDTH = 1280       # Resolusi horizontal
    CAMERA_HEIGHT = 720       # Resolusi vertikal

    # Threshold deteksi kemerahan (HSV)
    RED_SAT_MIN = 40
    RED_VAL_MIN = 60

    # Ukuran blob jerawat (pixel)
    ACNE_MIN_AREA = 25
    ACNE_MAX_AREA = 2500

    # Batas tingkat keparahan
    MILD_THRESHOLD = 8
    MODERATE_THRESHOLD = 20
    SEVERE_THRESHOLD = 40

    # Temporal smoothing
    HISTORY_SIZE = 15
```

---

## 🧠 Cara Kerja

```
Frame Kamera
     │
     ▼
MediaPipe Face Landmarker
     │  468 landmark wajah
     ▼
Skin Mask Generation
     │  Wajah diisolasi, mata/mulut/alis dieksklusi
     ▼
┌────┴─────────────────────────────────┐
│  Redness  │ Texture │ Dark  │ LAB   │
│  (HSV)    │  (Lap.) │ Spots │ Anom. │
└────┬─────────────────────────────────┘
     │  Kombinasi weighted mask
     ▼
Blob Detection + Circularity Filter
     │
     ▼
Zone Classification (7 zona wajah)
     │
     ▼
Temporal Smoothing → Health Score → UI
```

---

## 📊 Tingkat Keparahan

| Level | Jumlah Titik | Skor Kesehatan | Warna |
|---|---|---|---|
| Bersih | 0 | 100 | 🟢 |
| Ringan | 1–8 | 70–99 | 🟡 |
| Sedang | 9–20 | 40–69 | 🟠 |
| Parah | 21–40 | 15–39 | 🔴 |
| Sangat Parah | > 40 | < 15 | 🔴 |

---

## ⚠️ Keterbatasan

- Akurasi dipengaruhi kondisi pencahayaan — disarankan menggunakan cahaya yang merata
- Threshold HSV dioptimalkan untuk kulit terang-medium; mungkin perlu penyesuaian untuk tone kulit gelap
- Bukan pengganti diagnosis medis profesional — hasil analisis hanya bersifat indikatif

---

## 🤝 Kontribusi

Pull request dan issue sangat disambut! Beberapa area yang bisa dikembangkan:

- [ ] Adaptasi threshold otomatis berdasarkan tone kulit
- [ ] Export laporan PDF hasil scan
- [ ] Mode foto (tanpa kamera live)
- [ ] Model ML khusus deteksi jerawat

---

## 📄 Lisensi

Didistribusikan di bawah lisensi MIT. Lihat `LICENSE` untuk informasi lebih lanjut.

---

> **Disclaimer:** Aplikasi ini dibuat untuk tujuan edukasi dan eksplorasi computer vision. Bukan pengganti konsultasi dengan dokter kulit (dermatolog).
