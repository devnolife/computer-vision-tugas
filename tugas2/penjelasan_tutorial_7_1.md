# Tutorial 7.1: Pemrosesan Citra - Cropping, Resizing, Flipping, dan Rotasi

## Penjelasan Umum

Tutorial ini membahas operasi-operasi dasar dalam pemrosesan citra digital yang meliputi:
1. **Pemotongan citra (Cropping)**
2. **Pengubahan ukuran citra (Resizing)**
3. **Pembalikan citra (Flipping)**
4. **Rotasi citra (Rotation)**

## Langkah-langkah Tutorial

### 1. Pemotongan Citra Interaktif (Step 1-3)

**Tujuan:** Memahami cara memotong bagian tertentu dari citra secara interaktif.

**Penjelasan:**
- Menggunakan `RectangleSelector` dari matplotlib untuk memilih area yang akan dipotong
- Koordinat disimpan dalam format MATLAB (kolom, baris)
- User dapat drag dan select area yang diinginkan

**Hasil yang Diharapkan:**
- Tampil jendela dengan citra asli
- User dapat memilih area dengan mouse
- Koordinat area yang dipilih akan ditampilkan

### 2. Pemotongan Manual dengan Koordinat (Step 4-7)

**Tujuan:** Memotong citra menggunakan koordinat yang sudah ditentukan.

**Penjelasan:**
- Menggunakan koordinat yang sudah direkam sebelumnya
- Format: `[xmin, ymin, width, height]`
- Python menggunakan indexing `[y1:y2, x1:x2]`

**Contoh Koordinat:**
```python
x1, y1, x2, y2 = 186, 105, 211, 159
```

### 3. Pembesaran Citra dengan Interpolasi (Step 8-10)

**Tujuan:** Membandingkan hasil pembesaran citra dengan metode interpolasi yang berbeda.

**Metode Interpolasi:**

1. **Nearest-neighbor (Tetangga Terdekat)**
   - Paling sederhana dan cepat
   - Hasil terlihat pixelated (kotak-kotak)
   - Tidak ada nilai pixel baru yang dibuat
   - Cocok untuk citra biner atau label

2. **Bilinear (Dua Arah Linear)**
   - Menggunakan 4 pixel tetangga terdekat
   - Hasil lebih halus dari nearest-neighbor
   - Sedikit blur pada tepi
   - Komputasi sedang

3. **Bicubic (Kubik Dua Arah)**
   - Menggunakan 16 pixel tetangga
   - Hasil paling halus dan berkualitas tinggi
   - Komputasi paling berat
   - Terbaik untuk foto dan citra natural

**Perbandingan:**
- **Nearest-neighbor:** Tajam tapi pixelated
- **Bilinear:** Lebih halus, sedikit blur
- **Bicubic:** Paling halus, kualitas terbaik

### 4. Pengecilan dengan Subsampling (Step 11-12)

**Tujuan:** Memahami cara mengecilkan citra dengan mengambil setiap pixel ke-n.

**Penjelasan:**
- Menggunakan indexing `I[::2, ::2]` untuk mengambil setiap pixel ke-2
- Hanya bisa mengecilkan dengan faktor integer
- Dapat menyebabkan aliasing
- Informasi yang hilang tidak dapat dipulihkan

**Keterbatasan:**
- Hanya faktor integer (2x, 3x, 4x, dst.)
- Potensi aliasing artifacts
- Kehilangan informasi permanen

### 5. Pengecilan dengan Interpolasi (Step 13)

**Tujuan:** Membandingkan hasil pengecilan dengan metode interpolasi berbeda.

**Perbedaan dengan Subsampling:**
- Dapat menggunakan faktor non-integer (0.5, 0.75, dll.)
- Hasil lebih halus dan terkontrol
- Mengurangi aliasing artifacts

### 6. Pembalikan Citra (Step 14-16)

**Tujuan:** Memahami operasi flipping pada citra.

**Jenis Flipping:**

1. **Upside-down (Atas-bawah)**
   - `cv2.flip(image, 0)`
   - Membalik citra secara vertikal
   - Seperti melihat pantulan di air

2. **Left-right (Kiri-kanan)**
   - `cv2.flip(image, 1)`
   - Membalik citra secara horizontal
   - Seperti melihat di cermin

### 7. Rotasi Citra (Step 17-20)

**Tujuan:** Memahami operasi rotasi dan efeknya pada ukuran citra.

**Parameter Rotasi:**
- **Sudut:** Positif = counter-clockwise, Negatif = clockwise
- **Interpolasi:** Nearest, bilinear, atau bicubic
- **Crop output:** Mempertahankan ukuran asli atau expand canvas

**Efek Rotasi:**
- Ukuran citra berubah karena perlu canvas yang lebih besar
- Sudut 35° counter-clockwise = sudut -35° clockwise
- Interpolasi bilinear menghasilkan tepi yang lebih halus
- Crop setting mempertahankan ukuran asli dengan memotong hasil

## Pertanyaan dan Jawaban

### Question 2: Bandingkan ketiga hasil pembesaran citra
**Jawaban:**
- **Nearest-neighbor:** Pixelated, tepi tajam, tidak ada nilai pixel baru
- **Bilinear:** Lebih halus dari nearest-neighbor, sedikit blur
- **Bicubic:** Hasil paling halus, kualitas terbaik tapi komputasi tertinggi

### Question 3: Bagaimana cara mengecilkan citra?
**Jawaban:** 
Menggunakan indexing `I[::2, ::2]` untuk mengambil setiap pixel ke-2 dalam kedua dimensi.

### Question 4: Keterbatasan teknik subsampling
**Jawaban:**
- Hanya bisa resize dengan faktor integer
- Dapat menyebabkan aliasing artifacts
- Kehilangan informasi yang tidak dapat dipulihkan

### Question 5: Mengapa ukuran citra berubah setelah rotasi?
**Jawaban:**
Rotasi memerlukan canvas yang lebih besar untuk menampung seluruh citra yang dirotasi.

### Question 6: Cara rotasi 35° searah jarum jam
**Jawaban:**
Gunakan sudut negatif: `angle = -35`

### Question 7: Perbedaan interpolasi bilinear vs nearest-neighbor
**Jawaban:**
Interpolasi bilinear menghasilkan tepi yang lebih halus dibandingkan nearest-neighbor.

### Question 8: Fungsi crop setting pada rotasi
**Jawaban:**
Crop setting mempertahankan ukuran citra asli dengan memotong output hasil rotasi.

## Kesimpulan

Tutorial ini memberikan pemahaman fundamental tentang operasi geometri pada citra digital:

1. **Cropping:** Memilih region of interest (ROI) dari citra
2. **Resizing:** Mengubah ukuran dengan berbagai metode interpolasi
3. **Flipping:** Membalik citra secara horizontal atau vertikal
4. **Rotation:** Memutar citra dengan sudut tertentu

Setiap operasi memiliki kelebihan dan keterbatasan yang perlu dipahami untuk aplikasi yang tepat dalam pemrosesan citra.
