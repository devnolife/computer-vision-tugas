# LAPORAN PRAKTIKUM
## OPERASI GEOMETRIK PADA CITRA DIGITAL
### Tutorial 7.1: Image Cropping, Resizing, Flipping, dan Rotation
### Tutorial 7.2: Spatial Transformations dan Image Registration

**Nama:** Andi Agung Dwi Arya B  
**NIM:**   D082251054
**Kelas:** B  
**Tanggal:** 25 September 2025  

---

## TUTORIAL 7.1: IMAGE CROPPING, RESIZING, FLIPPING, DAN ROTATION

### TUJUAN
Tujuan dari tutorial ini adalah mempelajari cara memotong (crop), mengubah ukuran (resize), membalik (flip), dan merotasi citra digital menggunakan Python.

### OBJEKTIF
- Mempelajari cara memotong citra menggunakan fungsi crop
- Mempelajari cara mengubah ukuran citra dengan berbagai metode interpolasi
- Mempelajari cara membalik citra secara vertikal dan horizontal
- Mempelajari cara merotasi citra dengan berbagai parameter
- Mengeksplorasi metode interpolasi untuk resizing dan rotasi

---

## LANGKAH KERJA DAN PERTANYAAN

### Bagian 1: Cropping Citra

#### Langkah 1-3: Interactive Cropping
**Kode yang digunakan:**
```python
# [Sisipkan snippet kode untuk interactive cropping]
```

**Screenshot:**
```
[GAMBAR 1: Gambar asli cameraman.tif]
[GAMBAR 2: Proses pemilihan area crop secara interaktif]
[GAMBAR 3: Hasil cropping yang menampilkan building tertinggi]
```

#### Pertanyaan 1: 
Angka-angka apa yang Anda catat untuk koordinat pojok kiri atas dan kanan bawah, dan apa artinya? Perhatikan konvensi yang digunakan pada status bar informasi pixel.

**Jawaban:**
Koordinat yang dicatat: _________________

Penjelasan arti koordinat: _________________

---

### Bagian 2: Resizing Citra - Enlargement

#### Langkah 8-10: Pembesaran Citra dengan Berbagai Interpolasi
**Kode yang digunakan:**
```python
# [Sisipkan snippet kode untuk enlargement dengan factor 3]
```

**Screenshot:**
```
[GAMBAR 4: Perbandingan hasil enlargement]
- Gambar asli
- Bicubic interpolation (factor 3)
- Nearest-neighbor interpolation (factor 3) 
- Bilinear interpolation (factor 3)
```

#### Pertanyaan 2:
Bandingkan secara visual ketiga citra yang telah diperbesar. Bagaimana perbedaannya?

**Jawaban:**
Perbandingan metode interpolasi:
- Nearest-neighbor: _________________
- Bilinear: _________________
- Bicubic: _________________

Perbedaan visual yang diamati: _________________

---

### Bagian 3: Resizing Citra - Shrinking

#### Langkah 11-12: Pengecilan dengan Subsampling
**Kode yang digunakan:**
```python
# [Sisipkan snippet kode untuk subsampling]
```

**Screenshot:**
```
[GAMBAR 5: Perbandingan gambar asli dengan hasil subsampling]
```

#### Pertanyaan 3:
Bagaimana cara kita melakukan scaling pada gambar dengan metode ini?

**Jawaban:**
Metode scaling yang digunakan: _________________

#### Pertanyaan 4:
Apa keterbatasan dari teknik ini?

**Jawaban:**
Keterbatasan metode subsampling:
- _________________
- _________________
- _________________

#### Langkah 13: Pengecilan dengan Fungsi Resize
**Kode yang digunakan:**
```python
# [Sisipkan snippet kode untuk shrinking dengan interpolasi]
```

**Screenshot:**
```
[GAMBAR 6: Perbandingan hasil shrinking dengan berbagai interpolasi (factor 0.5)]
- Nearest-neighbor
- Bilinear  
- Bicubic
```

**Penjelasan:**
Perbedaan hasil antara ketiga metode interpolasi pada pengecilan: _________________

---

### Bagian 4: Flipping Citra

#### Langkah 14-16: Membalik Citra
**Kode yang digunakan:**
```python
# [Sisipkan snippet kode untuk flipping]
```

**Screenshot:**
```
[GAMBAR 7: Perbandingan flipping]
- Gambar asli
- Flipped upside-down (flipud)
- Flipped left-right (fliplr)
```

**Penjelasan:**
Fungsi yang digunakan untuk flipping dan efeknya: _________________

---

### Bagian 5: Rotasi Citra

#### Langkah 17-20: Rotasi Citra
**Kode yang digunakan:**
```python
# [Sisipkan snippet kode untuk rotasi dengan berbagai parameter]
```

**Screenshot:**
```
[GAMBAR 8: Hasil rotasi eight.tif]
- Gambar asli eight.tif
- Rotasi 35° (bicubic)
- Rotasi 35° (bilinear) 
- Rotasi 35° dengan crop
```

#### Pertanyaan 5:
Periksa ukuran (jumlah baris dan kolom) dari hasil rotasi dan bandingkan dengan gambar asli. Mengapa berbeda?

**Jawaban:**
Ukuran gambar asli: _________________
Ukuran gambar hasil rotasi: _________________
Alasan perbedaan ukuran: _________________

#### Pertanyaan 6:
Langkah sebelumnya merotasi gambar berlawanan arah jarum jam. Bagaimana cara merotasi gambar 35° searah jarum jam?

**Jawaban:**
Cara rotasi searah jarum jam: _________________

#### Pertanyaan 7:
Bagaimana interpolasi bilinear mempengaruhi output rotasi? Petunjuk: Perbedaan terlihat di sekitar tepi gambar yang dirotasi dan di sekitar koin.

**Jawaban:**
Efek interpolasi bilinear pada rotasi: _________________

#### Pertanyaan 8:
Bagaimana pengaturan crop mengubah ukuran output kita?

**Jawaban:**
Efek crop setting pada ukuran output: _________________

---

## TUTORIAL 7.2: SPATIAL TRANSFORMATIONS DAN IMAGE REGISTRATION

### TUJUAN
Mengeksplorasi fungsi transformasi spasial dan mendemonstrasikan contoh sederhana pemilihan control points serta penggunaannya dalam konteks image registration.

---

## LANGKAH KERJA DAN PERTANYAAN

### Bagian 1: Transformasi Affine - Scaling

#### Langkah 1-4: Transformasi Scaling
**Kode yang digunakan:**
```python
# [Sisipkan snippet kode untuk affine scaling transformation]
```

**Screenshot:**
```
[GAMBAR 9: Perbandingan scaling methods]
- Gambar asli
- Menggunakan affine transformation (sx=2, sy=2)
- Menggunakan image resizing
```

#### Pertanyaan 1:
Bandingkan kedua citra hasil (transformasi affine vs image resizing). Periksa ukuran, rentang gray-level, dan kualitas visual. Bagaimana perbedaannya? Mengapa?

**Jawaban:**
Perbandingan hasil transformasi:
- Ukuran hasil affine: _________________
- Ukuran hasil resize: _________________
- Perbedaan kualitas visual: _________________
- Alasan perbedaan: _________________

---

### Bagian 2: Transformasi Affine - Rotation

#### Langkah 5-7: Transformasi Rotasi
**Kode yang digunakan:**
```python
# [Sisipkan snippet kode untuk affine rotation transformation]
```

**Screenshot:**
```
[GAMBAR 10: Perbandingan rotation methods]
- Gambar asli
- Menggunakan affine transformation (35°)
- Menggunakan image rotation
```

#### Pertanyaan 2:
Bandingkan kedua citra hasil (transformasi affine vs image rotation). Periksa ukuran, rentang gray-level, dan kualitas visual. Bagaimana perbedaannya? Mengapa?

**Jawaban:**
Perbandingan hasil rotasi:
- Kualitas visual affine vs rotation: _________________
- Perbedaan dalam boundary handling: _________________
- Perbedaan interpolasi: _________________

---

### Bagian 3: Transformasi Affine - Translation

#### Langkah 8-10: Transformasi Translasi
**Kode yang digunakan:**
```python
# [Sisipkan snippet kode untuk translation transformation]
```

**Screenshot:**
```
[GAMBAR 11: Hasil translation]
- Gambar asli
- Gambar hasil translasi (dx=50, dy=100) dengan fill value abu-abu
```

#### Pertanyaan 3:
Bandingkan kedua gambar (asli dan hasil translasi). Periksa ukuran, rentang gray-level, dan kualitas visual. Bagaimana perbedaannya? Mengapa?

**Jawaban:**
Perbandingan hasil translasi:
- Ukuran gambar asli: _________________
- Ukuran gambar hasil translasi: _________________
- Efek fill value: _________________
- Alasan perubahan ukuran: _________________

---

### Bagian 4: Transformasi Affine - Shearing

#### Langkah 11-13: Transformasi Shearing
**Kode yang digunakan:**
```python
# [Sisipkan snippet kode untuk shearing transformation]
```

**Screenshot:**
```
[GAMBAR 12: Hasil shearing]
- Gambar asli
- Gambar hasil shearing (shx=2, shy=1.5)
```

**Penjelasan hasil shearing:** _________________

---

## BAGIAN IMAGE REGISTRATION

### Langkah 14: Load Base dan Unregistered Images
**Screenshot:**
```
[GAMBAR 13: Input images untuk registration]
- Base image (klcc_a.png)
- Unregistered image (klcc_b.png)
```

### Langkah 15: Control Point Selection
**Kode yang digunakan:**
```python
# [Sisipkan snippet kode untuk control point selection]
```

**Screenshot:**
```
[GAMBAR 14: Control Point Selection Interface]
- Tampilan interactive control point selection
- Titik-titik yang dipilih pada kedua gambar
```

**Koordinat control points yang dipilih:**
```
Input points: _________________
Base points: _________________
```

### Langkah 17: Fine-tune Control Points
**Kode yang digunakan:**
```python
# [Sisipkan snippet kode untuk fine-tuning control points]
```

#### Pertanyaan 4:
Bandingkan nilai input_points_adj dengan input_points. Apakah Anda melihat perubahan? Mengapa (tidak)?

**Jawaban:**
Perbandingan points sebelum dan sesudah fine-tuning:
- Points asli: _________________
- Points setelah fine-tuning: _________________
- Perubahan yang diamati: _________________
- Alasan perubahan: _________________

### Langkah 18-20: Estimasi dan Aplikasi Transformasi
**Kode yang digunakan:**
```python
# [Sisipkan snippet kode untuk transformation estimation dan application]
```

### Langkah 21: Display Registered Image
**Screenshot:**
```
[GAMBAR 15: Hasil Image Registration]
- Base image
- Unregistered image  
- Registered image
- Overlay visualization (Red: Base, Green: Registered)
```

#### Pertanyaan 5:
Apakah Anda puas dengan hasilnya? Jika harus mengulang lagi, apa yang akan Anda lakukan berbeda?

**Jawaban:**
Evaluasi hasil registration:
- Kualitas registration: _________________
- Mean squared difference sebelum: _________________
- Mean squared difference sesudah: _________________
- Tingkat improvement: _________________
- Saran perbaikan: _________________

---

## ANALISIS DAN KESIMPULAN

### Perbandingan Metode Interpolasi
Berdasarkan hasil praktikum, jelaskan kelebihan dan kekurangan masing-masing metode interpolasi:

**Nearest-neighbor:**
- Kelebihan: _________________
- Kekurangan: _________________

**Bilinear:**
- Kelebihan: _________________
- Kekurangan: _________________

**Bicubic:**
- Kelebihan: _________________
- Kekurangan: _________________

### Transformasi Affine vs Fungsi Built-in
Jelaskan perbedaan fundamental antara menggunakan transformasi affine manual vs fungsi built-in untuk operasi geometrik:

_________________

### Image Registration
Jelaskan faktor-faktor yang mempengaruhi kualitas image registration:

1. _________________
2. _________________
3. _________________
4. _________________

### Aplikasi Praktis
Sebutkan dan jelaskan aplikasi praktis dari setiap operasi geometrik yang dipelajari:

**Cropping:** _________________

**Resizing:** _________________

**Flipping:** _________________

**Rotation:** _________________

**Image Registration:** _________________

---

## KESIMPULAN UMUM

Tuliskan kesimpulan umum dari praktikum ini, mencakup:
1. Pemahaman tentang operasi geometrik dasar pada citra
2. Pentingnya pemilihan metode interpolasi yang tepat
3. Aplikasi transformasi affine dalam image processing
4. Proses dan tantangan dalam image registration

**Kesimpulan:**
_________________

---

## DAFTAR PUSTAKA

1. _________________
2. _________________
3. _________________

---

## LAMPIRAN

### Lampiran A: Source Code Lengkap
```python
# [Lampirkan source code lengkap yang digunakan]
```

### Lampiran B: Error Messages dan Solusi
| Error | Solusi |
|-------|--------|
| _____ | ______ |
| _____ | ______ |

---

**Catatan:** 
- Pastikan semua screenshot disertakan dengan kualitas yang baik
- Berikan penjelasan yang detail untuk setiap gambar
- Jawaban harus mencerminkan pemahaman konsep, bukan hanya hasil eksperimen
- Sertakan analisis perbandingan antar metode dengan objektif
