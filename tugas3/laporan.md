# LAPORAN PRAKTIKUM
## HISTOGRAM PROCESSING
### Tutorial 9.1: Image Histograms
### Tutorial 9.2: Histogram Equalization and Specification
### Tutorial 9.3: Other Histogram Modification Techniques

**Nama:** _________________  
**NIM:** _________________  
**Kelas:** _________________  
**Tanggal:** _________________  

---

## TUTORIAL 9.1: IMAGE HISTOGRAMS

### TUJUAN
Tujuan dari tutorial ini adalah menggunakan Python untuk menghitung dan menampilkan histogram citra.

### OBJEKTIF
- Mempelajari cara menghitung histogram citra
- Mempelajari berbagai teknik plotting untuk melihat dan menganalisis data histogram
- Memahami normalisasi histogram
- Mengeksplorasi berbagai cara visualisasi histogram

---

## LANGKAH KERJA DAN PERTANYAAN

### Bagian 1: Menampilkan Histogram Dasar

#### Langkah 1-2: Menampilkan Citra dan Histogram dengan Berbagai Jumlah Bin

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk membuat histogram dengan 256, 64, dan 32 bins]
processor = HistogramProcessor('circuit.tif')

# Plot dengan 256 bins
# Plot dengan 64 bins  
# Plot dengan 32 bins
```

**Screenshot:**
```
[GAMBAR 1: Perbandingan histogram dengan berbagai jumlah bins]
- Gambar asli circuit.tif
- Histogram dengan 256 bins
- Histogram dengan 64 bins
- Histogram dengan 32 bins
```

#### Pertanyaan 1:
Jelaskan perubahan drastis pada nilai sumbu Y ketika histogram ditampilkan dengan jumlah bin yang lebih sedikit.

**Jawaban:**
Perubahan nilai sumbu Y: _________________

Penjelasan: _________________

Alasan matematis: _________________

---

### Bagian 2: Mendapatkan dan Menormalisasi Data Histogram

#### Langkah 3-4: Mendapatkan Nilai Histogram dan Normalisasi

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk mendapatkan nilai histogram]
hist_32, _ = processor.compute_histogram(bins=32)

# [Sisipkan kode untuk normalisasi]
hist_norm = processor.normalize_histogram(hist_32)
```

#### Pertanyaan 2:
Apa yang dilakukan fungsi `numel()` dalam MATLAB atau ekuivalennya di Python?

**Jawaban:**
Fungsi yang digunakan di Python: _________________

Fungsi yang dilakukan: _________________

#### Pertanyaan 3:
Tulis pernyataan satu baris yang akan memverifikasi bahwa jumlah nilai yang dinormalisasi adalah 1.

**Kode verifikasi:**
```python
# [Sisipkan kode untuk verifikasi]
```

**Hasil verifikasi:** _________________

---

### Bagian 3: Visualisasi Histogram dengan Bar Chart

#### Langkah 6-9: Menampilkan Bar Chart dan Kustomisasi

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk membuat bar chart]
# Bar chart standar
# Bar chart yang dinormalisasi
# Kustomisasi warna, axis limits, tick marks
```

**Screenshot:**
```
[GAMBAR 2: Bar Chart Visualization]
- Bar chart standar (warna merah)
- Bar chart yang dinormalisasi (warna hijau)
```

**Penjelasan kustomisasi yang dilakukan:** _________________

#### Pertanyaan 4:
Bagaimana cara mengubah lebar bar dalam bar chart?

**Jawaban:**
Parameter yang digunakan: _________________

Contoh kode: _________________

---

### Bagian 4: Stem Chart

#### Langkah 11: Menampilkan Stem Chart

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk stem chart]
# Stem chart standar
# Stem chart yang dinormalisasi
```

**Screenshot:**
```
[GAMBAR 3: Stem Chart Visualization]
- Stem chart standar
- Stem chart yang dinormalisasi
```

#### Pertanyaan 5:
Eksplorasi properti stem chart. Bagaimana cara membuat garis menjadi putus-putus (dotted) daripada solid?

**Jawaban:**
Parameter yang digunakan: _________________

Contoh kode: _________________

#### Pertanyaan 6:
Sesuaikan batas sumbu (axis limits) dan tick marks agar mencerminkan data yang ditampilkan dalam stem plot.

**Jawaban:**
Kode untuk menyesuaikan axis: _________________

Range yang dipilih: _________________

---

### Bagian 5: Plot Graph

#### Langkah 12: Menampilkan Plot Graph

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk plot graph]
# Plot graph standar
# Plot graph yang dinormalisasi
```

**Screenshot:**
```
[GAMBAR 4: Plot Graph Visualization]
- Plot graph standar
- Plot graph yang dinormalisasi
```

#### Pertanyaan 7:
Eksplorasi properti plot graph. Dalam kode di atas, titik-titik untuk setiap bin hilang secara visual dalam garis grafik. Bagaimana cara membuat titik-titik lebih tebal sehingga lebih terlihat?

**Jawaban:**
Parameter untuk membuat titik lebih terlihat: _________________

Contoh kode: _________________

---

## TUTORIAL 9.2: HISTOGRAM EQUALIZATION DAN SPECIFICATION

### TUJUAN
Tujuan dari tutorial ini adalah mempelajari cara menggunakan histogram equalization (global dan lokal) dan histogram specification (matching).

### OBJEKTIF
- Mengeksplorasi proses histogram equalization
- Mempelajari cara melakukan histogram specification (matching)
- Mempelajari cara melakukan local histogram equalization dengan adaptive methods
- Memahami kapan histogram equalization bekerja dengan baik dan kapan tidak

---

## LANGKAH KERJA DAN PERTANYAAN

### Bagian 1: Histogram Equalization pada Gambar Pout

#### Langkah 1-3: Histogram Equalization Dasar

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk histogram equalization pada pout.tif]
processor_pout = HistogramProcessor('pout.tif')
pout_eq = processor_pout.histogram_equalization()
```

**Screenshot:**
```
[GAMBAR 5: Histogram Equalization pada pout.tif]
- Gambar asli
- Histogram asli
- Gambar setelah equalization
- Histogram setelah equalization
```

#### Pertanyaan 1:
Mengapa harus menyertakan parameter kedua (256) dalam fungsi histogram equalization?

**Jawaban:**
Alasan parameter 256: _________________

Fungsi parameter ini: _________________

#### Pertanyaan 2:
Apa efek histogram equalization pada gambar dengan kontras rendah?

**Jawaban:**
Efek pada kontras: _________________

Efek pada distribusi intensitas: _________________

Perubahan visual yang terlihat: _________________

---

### Bagian 2: Histogram Equalization pada Gambar Tire

#### Langkah 5: Histogram Equalization pada Tire.tif

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk histogram equalization pada tire.tif]
processor_tire = HistogramProcessor('tire.tif')
tire_eq = processor_tire.histogram_equalization()
```

**Screenshot:**
```
[GAMBAR 6: Histogram Equalization pada tire.tif]
- Gambar asli
- Histogram asli
- Gambar setelah equalization
- Histogram setelah equalization
```

#### Pertanyaan 3:
Berdasarkan histogram asli gambar tire, apa yang dapat dikatakan tentang brightness keseluruhan gambar?

**Jawaban:**
Karakteristik brightness: _________________

Distribusi nilai pixel: _________________

#### Pertanyaan 4:
Bagaimana histogram equalization mempengaruhi brightness keseluruhan gambar dalam kasus ini?

**Jawaban:**
Perubahan brightness: _________________

Perubahan kontras: _________________

Kualitas gambar hasil: _________________

---

### Bagian 3: Histogram Equalization pada Gambar Eight

#### Langkah 7: Histogram Equalization pada Eight.tif

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk histogram equalization pada eight.tif]
processor_eight = HistogramProcessor('eight.tif')
eight_eq = processor_eight.histogram_equalization()
```

**Screenshot:**
```
[GAMBAR 7: Histogram Equalization pada eight.tif]
- Gambar asli
- Histogram asli (bimodal)
- Gambar setelah equalization
- Histogram setelah equalization
```

#### Pertanyaan 5:
Mengapa terjadi penurunan kualitas gambar yang sangat signifikan setelah histogram equalization?

**Jawaban:**
Alasan penurunan kualitas: _________________

Hubungan dengan distribusi histogram bimodal: _________________

Kesimpulan tentang limitasi histogram equalization: _________________

---

### Bagian 4: Transformation Function (CDF)

#### Langkah 8-9: Menampilkan Fungsi Transformasi

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk menghitung dan plot CDF]
hist_eight, _ = processor_eight.compute_histogram()
cdf = np.cumsum(hist_eight)
cdf_normalized = cdf / cdf[-1]
```

**Screenshot:**
```
[GAMBAR 8: Normalized CDF - Transformation Function]
- Plot CDF yang dinormalisasi
```

#### Pertanyaan 6:
Apa yang dilakukan fungsi `cumsum()` pada langkah sebelumnya?

**Jawaban:**
Fungsi cumsum: _________________

Penggunaannya dalam histogram equalization: _________________

Hubungan dengan transformation function: _________________

---

### Bagian 5: Histogram Specification (Matching)

#### Langkah 11-13: Histogram Matching

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk histogram matching]
# Membuat desired histogram shapes (uniform dan linear)
uniform_hist = np.ones(256) * 0.5
linear_hist = np.linspace(0, 1, 256)
matched_image = processor_pout.histogram_matching(linear_hist * 1000)
```

**Screenshot:**
```
[GAMBAR 9: Histogram Specification/Matching]
Grid 3x3 menampilkan:
- Baris 1: Original image, histogram, (kosong)
- Baris 2: Equalized image, histogram, desired shape (uniform)
- Baris 3: Matched image, histogram, desired shape (linear)
```

**Analisis hasil histogram matching:** _________________

#### Pertanyaan 7:
Apa fungsi checkbox "Continuous Update" pada Interactive Histogram Matching demo?

**Jawaban:**
Fungsi continuous update: _________________

#### Pertanyaan 8:
Bagaimana metode interpolasi yang berbeda mengubah bentuk kurva histogram yang diinginkan?

**Jawaban:**
Perbedaan metode interpolasi: _________________

#### Pertanyaan 9:
Bagaimana cara memuat demo dengan gambar yang berbeda?

**Jawaban:**
Cara memuat gambar berbeda: _________________

---

### Bagian 6: Local (Adaptive) Histogram Equalization

#### Langkah 17: CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk adaptive histogram equalization]
processor_coins = HistogramProcessor('coins.png')
coins_eq = processor_coins.histogram_equalization()
coins_adaptive = processor_coins.adaptive_histogram_equalization(clip_limit=0.1)
```

**Screenshot:**
```
[GAMBAR 10: Adaptive Histogram Equalization]
- Gambar asli coins.png
- Histogram asli (bimodal)
- Global histogram equalization
- Histogram global eq
- Local histogram equalization (CLAHE)
- Histogram local eq
```

#### Pertanyaan 10:
Apa fungsi parameter `ClipLimit` dalam fungsi adaptive histogram equalization?

**Jawaban:**
Fungsi ClipLimit: _________________

Efek pada hasil: _________________

Nilai optimal: _________________

#### Pertanyaan 11:
Berapa ukuran tile default ketika menggunakan adaptive histogram equalization?

**Jawaban:**
Ukuran tile default: _________________

Pengaruh ukuran tile: _________________

---

## TUTORIAL 9.3: TEKNIK MODIFIKASI HISTOGRAM LAINNYA

### TUJUAN
Tujuan dari tutorial ini adalah mempelajari cara melakukan operasi modifikasi histogram umum lainnya.

### OBJEKTIF
- Mempelajari cara menyesuaikan brightness gambar dengan histogram sliding
- Mempelajari cara menggunakan fungsi untuk penyesuaian intensitas
- Mengeksplorasi penyesuaian kontras melalui histogram stretching
- Mempelajari cara menyesuaikan kontras dengan histogram shrinking

---

## LANGKAH KERJA DAN PERTANYAAN

### Bagian 1: Histogram Sliding (Brightness Adjustment)

#### Langkah 1-3: Menambahkan Konstanta pada Pixel

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk histogram sliding]
processor = HistogramProcessor('pout.tif')

# Menambah 0.1
img_bright1 = processor.histogram_sliding(0.1)

# Menambah 0.5
img_bright2 = processor.histogram_sliding(0.5)
```

**Screenshot:**
```
[GAMBAR 11: Histogram Sliding]
Grid 3x2:
- Original image dan histogram
- Image + 0.1 dan histogram
- Image + 0.5 dan histogram
```

#### Pertanyaan 1:
Bagaimana histogram berubah setelah penyesuaian?

**Jawaban:**
Perubahan pada distribusi histogram: _________________

Pergeseran nilai intensitas: _________________

#### Pertanyaan 2:
Apa yang dikandung oleh variabel `bad_values`?

**Jawaban:**
Isi variabel bad_values: _________________

Tujuan variabel ini: _________________

#### Pertanyaan 3:
Mengapa plot ketiga menunjukkan jumlah pixel yang berlebihan dengan nilai 1?

**Jawaban:**
Alasan saturasi pada nilai 1: _________________

Hubungan dengan clipping: _________________

Implikasi untuk kualitas gambar: _________________

---

### Bagian 2: Histogram Stretching

#### Langkah 5-6: Histogram Stretching dengan Fungsi imadjust

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk histogram stretching]
img_stretched = processor.imadjust()

# Dengan parameter default
img_stretched2 = processor.imadjust()

# Bandingkan hasil
diff = cv2.absdiff(img_stretched, img_stretched2)
```

**Screenshot:**
```
[GAMBAR 12: Histogram Stretching]
- Original image dan histogram
- Stretched image dan histogram
- Stretched image (default params) dan histogram
- Difference image
```

#### Pertanyaan 4:
Bagaimana histogram berubah setelah penyesuaian?

**Jawaban:**
Perubahan distribusi histogram: _________________

Perubahan range intensitas: _________________

Peningkatan kontras: _________________

#### Pertanyaan 5:
Apa tujuan menggunakan fungsi `stretchlim()`?

**Jawaban:**
Fungsi stretchlim: _________________

Parameter yang dihitung: _________________

Tujuan dalam histogram stretching: _________________

#### Pertanyaan 6:
Bagaimana tampilan difference image?

**Jawaban:**
Tampilan difference image: _________________

Nilai min: ___________ Nilai max: ___________

#### Pertanyaan 7:
Apa tujuan memeriksa nilai maksimum dan minimum dari difference image?

**Jawaban:**
Tujuan pemeriksaan: _________________

Kesimpulan yang dapat diambil: _________________

---

### Bagian 3: Histogram Shrinking

#### Langkah 8-9: Histogram Shrinking dan Transformation Function

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk histogram shrinking]
processor_west = HistogramProcessor('westconcord.png')
img_shrunk = processor_west.imadjust(low_out=0.25, high_out=0.75)

# Plot transformation function
X = processor_west.original_image.flatten()
Y = img_shrunk.flatten()
plt.scatter(X, Y, alpha=0.1, s=1)
```

**Screenshot:**
```
[GAMBAR 13: Histogram Shrinking]
- Original image dan histogram
- Shrunk image dan histogram
- Transformation function plot
```

#### Pertanyaan 8:
Apa yang dilakukan dua pernyataan pertama dalam kode (reshape/flatten)?

**Jawaban:**
Fungsi reshape/flatten: _________________

Tujuan dalam konteks ini: _________________

#### Pertanyaan 9:
Apa fungsi `xlabel()` dan `ylabel()`?

**Jawaban:**
Fungsi xlabel dan ylabel: _________________

Pentingnya dalam visualisasi: _________________

---

### Bagian 4: Histogram Shrinking dengan Gamma

#### Langkah 11: Menggunakan Nilai Gamma

**Kode yang digunakan:**
```python
# [Sisipkan kode untuk histogram shrinking dengan gamma=2]
img_shrunk_gamma = processor_west.imadjust(
    low_out=0.25, 
    high_out=0.75, 
    gamma=2.0
)
```

**Screenshot:**
```
[GAMBAR 14: Histogram Shrinking dengan Gamma]
- Original image dan histogram
- Adjusted image (gamma=2) dan histogram
- Transformation function dengan gamma=2
```

#### Pertanyaan 10:
Plot transformation function menampilkan gap dari 0 hingga 12 (pada sumbu X) di mana tidak ada titik. Mengapa demikian?

**Jawaban:**
Alasan gap pada plot: _________________

Hubungan dengan histogram asli: _________________

Interpretasi: _________________

---

## ANALISIS DAN KESIMPULAN

### Perbandingan Teknik Histogram Processing

#### 1. Histogram Equalization
**Kelebihan:**
- _________________
- _________________

**Kekurangan:**
- _________________
- _________________

**Cocok untuk:**
- _________________

#### 2. Adaptive Histogram Equalization (CLAHE)
**Kelebihan:**
- _________________
- _________________

**Kekurangan:**
- _________________
- _________________

**Cocok untuk:**
- _________________

#### 3. Histogram Sliding
**Kelebihan:**
- _________________
- _________________

**Kekurangan:**
- _________________
- _________________

**Cocok untuk:**
- _________________

#### 4. Histogram Stretching
**Kelebihan:**
- _________________
- _________________

**Kekurangan:**
- _________________
- _________________

**Cocok untuk:**
- _________________

#### 5. Histogram Shrinking
**Kelebihan:**
- _________________
- _________________

**Kekurangan:**
- _________________
- _________________

**Cocok untuk:**
- _________________

---

## KESIMPULAN UMUM

### Pemahaman Konsep Histogram

Jelaskan pemahaman Anda tentang:

1. **Apa itu histogram dan kegunaannya:**
_________________

2. **Hubungan histogram dengan kualitas gambar:**
_________________

3. **Kapan histogram equalization efektif dan tidak efektif:**
_________________

4. **Perbedaan global vs local histogram equalization:**
_________________

5. **Pentingnya transformation function:**
_________________

### Aplikasi Praktis

Sebutkan dan jelaskan aplikasi praktis dari setiap teknik:

**Histogram Equalization:**
_________________

**Adaptive Histogram Equalization:**
_________________

**Histogram Matching:**
_________________

**Histogram Sliding:**
_________________

**Histogram Stretching/Shrinking:**
_________________

### Perbandingan Python vs MATLAB

Jelaskan perbedaan implementasi yang Anda temukan:

| Aspek | MATLAB | Python | Keterangan |
|-------|---------|---------|------------|
| Fungsi histogram | imhist() | _______ | __________ |
| Histogram equalization | histeq() | _______ | __________ |
| Adaptive equalization | adapthisteq() | _______ | __________ |
| Image adjustment | imadjust() | _______ | __________ |
| Visualisasi | _______ | _______ | __________ |

---

## KESIMPULAN AKHIR

Tuliskan kesimpulan akhir yang mencakup:
1. Pemahaman tentang histogram dan peranannya dalam image processing
2. Kelebihan dan kekurangan setiap teknik modifikasi histogram
3. Pemilihan teknik yang tepat untuk berbagai jenis masalah
4. Pentingnya memahami karakteristik gambar sebelum memilih teknik

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
# [Lampirkan source code lengkap semua tutorial]
```

### Lampiran B: Perbandingan Hasil

| Gambar | Original | Equalized | Adaptive | Stretched | Shrunk |
|--------|----------|-----------|----------|-----------|---------|
| pout.tif | _______ | _______ | _______ | _______ | _______ |
| tire.tif | _______ | _______ | _______ | _______ | _______ |
| eight.tif | _______ | _______ | _______ | _______ | _______ |
| coins.png | _______ | _______ | _______ | _______ | _______ |

### Lampiran C: Analisis Kuantitatif

Sertakan pengukuran kuantitatif seperti:
- Mean intensity sebelum dan sesudah
- Standard deviation
- Contrast ratio
- Histogram entropy

---

**Catatan Pengerjaan:**
- Pastikan semua screenshot memiliki kualitas yang baik dan dapat dibaca dengan jelas
- Berikan penjelasan detail untuk setiap gambar yang disertakan
- Jawaban harus menunjukkan pemahaman konsep, bukan hanya hasil eksperimen
- Sertakan analisis perbandingan antar metode dengan objektif
- Pastikan semua code snippets yang disertakan dapat dijalankan
- Verifikasi bahwa semua pertanyaan telah dijawab dengan lengkap
