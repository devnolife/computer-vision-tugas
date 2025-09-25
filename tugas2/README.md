# Tutorial 7.1: Pemrosesan Citra Digital

## Deskripsi
Tutorial ini membahas operasi dasar dalam pemrosesan citra digital menggunakan Python, OpenCV, dan Matplotlib. Program ini mencakup:

- **Pemotongan Citra (Cropping)**: Interaktif dan manual
- **Pengubahan Ukuran (Resizing)**: Dengan berbagai metode interpolasi
- **Pembalikan Citra (Flipping)**: Horizontal dan vertikal  
- **Rotasi Citra (Rotation)**: Dengan berbagai sudut dan interpolasi

## Struktur File
```
tugas2/
├── 7.1.py                          # Program utama
├── penjelasan_tutorial_7_1.md       # Penjelasan lengkap tutorial
└── README.md                        # File ini
```

## Persyaratan
### Library yang Dibutuhkan:
```bash
pip install opencv-python numpy matplotlib scikit-image pillow
```

### File Citra:
Program akan otomatis menggunakan file dari folder `../assets/`:
- `cameraman2.tif` (untuk tutorial utama)
- `moon.tif` (untuk rotasi)
- Alternatif: `lena.png`, `tire.tif`, `lindsay.tif`, `pout.tif`

## Cara Menjalankan

1. **Pastikan berada di direktori tugas2:**
   ```bash
   cd tugas2
   ```

2. **Jalankan program:**
   ```bash
   python 7.1.py
   ```

3. **Program akan menampilkan:**
   - Grafik matplotlib dengan berbagai hasil pemrosesan
   - Output teks dalam bahasa Indonesia
   - Jawaban untuk semua pertanyaan tutorial

## Fitur Program

### 1. Pemotongan Citra Interaktif
- Menggunakan `matplotlib.widgets.RectangleSelector`
- User dapat drag dan select area yang diinginkan
- Koordinat otomatis ditampilkan

### 2. Pemotongan Manual
- Menggunakan koordinat yang sudah ditentukan
- Contoh koordinat: `x1=186, y1=105, x2=211, y2=159`

### 3. Pembesaran dengan Interpolasi
- **Nearest-neighbor**: Hasil pixelated, cepat
- **Bilinear**: Hasil halus, kecepatan sedang
- **Bicubic**: Hasil terbaik, lambat

### 4. Pengecilan Citra
- **Subsampling**: Mengambil setiap pixel ke-n
- **Interpolasi**: Hasil lebih halus, faktor bebas

### 5. Pembalikan Citra
- **Atas-bawah**: `cv2.flip(image, 0)`
- **Kiri-kanan**: `cv2.flip(image, 1)`

### 6. Rotasi Citra
- Sudut bebas (positif = counter-clockwise)
- Berbagai metode interpolasi
- Opsi crop untuk mempertahankan ukuran

## Output Program

Program akan menampilkan:

### Grafik yang Dihasilkan:
1. Citra asli
2. Citra yang dipotong manual
3. Perbandingan 4 hasil pembesaran (asli + 3 interpolasi)
4. Perbandingan pengecilan (asli vs subsampling)
5. Perbandingan 3 metode pengecilan dengan interpolasi
6. Perbandingan pembalikan (asli, atas-bawah, kiri-kanan)
7. Perbandingan 4 hasil rotasi (asli + 3 variasi)

### Teks Output:
- Penjelasan setiap langkah dalam bahasa Indonesia
- Jawaban untuk 8 pertanyaan tutorial
- Informasi ukuran citra sebelum dan sesudah operasi

## Pertanyaan dan Jawaban

### Pertanyaan 2: Bandingkan ketiga hasil pembesaran citra
- **Tetangga Terdekat**: Terlihat kotak-kotak, tepi tajam, tidak ada nilai pixel baru
- **Bilinear**: Lebih halus, sedikit blur
- **Bikubik**: Hasil paling halus, kualitas terbaik

### Pertanyaan 3: Bagaimana cara mengecilkan citra?
Menggunakan indexing `I[::2, ::2]` untuk mengambil setiap pixel ke-2

### Pertanyaan 4: Keterbatasan subsampling
- Hanya faktor integer
- Dapat menyebabkan aliasing
- Kehilangan informasi permanen

### Pertanyaan 5: Mengapa ukuran berubah setelah rotasi?
Rotasi memerlukan canvas lebih besar untuk menampung seluruh citra

### Pertanyaan 6: Rotasi searah jarum jam
Gunakan sudut negatif: `angle = -35`

### Pertanyaan 7: Perbedaan interpolasi
Bilinear menghasilkan tepi lebih halus dari nearest-neighbor

### Pertanyaan 8: Fungsi crop setting
Mempertahankan ukuran asli dengan memotong output

## Troubleshooting

### Error: File tidak ditemukan
- Pastikan folder `../assets/` ada
- Pastikan ada file citra (`.tif`, `.png`) di folder assets
- Program akan otomatis membuat sample jika tidak ada file

### Error: Library tidak ditemukan
```bash
pip install opencv-python numpy matplotlib scikit-image pillow
```

### Error: Matplotlib tidak menampilkan grafik
- Pastikan menggunakan environment dengan GUI support
- Coba tambahkan `plt.ion()` di awal program

## Catatan Teknis

- Program menggunakan OpenCV untuk operasi citra dasar
- Matplotlib untuk visualisasi dan interaksi
- Scikit-image untuk rotasi advanced
- Semua output dalam bahasa Indonesia
- Path menggunakan relative path `../assets/` 

## Pengembangan Lebih Lanjut

Program ini dapat dikembangkan untuk:
- Menambah jenis transformasi geometri lain
- Implementasi filter dan enhancement
- Batch processing multiple images
- GUI yang lebih user-friendly
- Export hasil ke berbagai format
