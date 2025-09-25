# Tutorial 7.2: Transformasi Spasial dan Registrasi Citra

## Penjelasan Umum

Tutorial ini membahas operasi transformasi spasial lanjutan dan registrasi citra yang meliputi:
1. **Transformasi Affine** (Scaling, Rotation, Translation, Shearing)
2. **Pemilihan Control Points** secara interaktif
3. **Fine-tuning Control Points** menggunakan korelasi
4. **Estimasi Parameter Transformasi**
5. **Registrasi Citra** untuk menyelaraskan dua citra

## Konsep Dasar

### Transformasi Affine
Transformasi affine adalah transformasi geometri yang mempertahankan:
- Garis lurus tetap lurus
- Rasio jarak pada garis yang sama
- Paralelisme antar garis

**Matriks Transformasi Affine (3x3):**
```
[a  b  tx]   [x]   [ax + by + tx]
[c  d  ty] × [y] = [cx + dy + ty]
[0  0   1]   [1]   [     1      ]
```

## Langkah-langkah Tutorial

### Bagian I: Transformasi Affine

#### 1. Penskalaan (Scaling) - Langkah 1-4

**Tujuan:** Mengubah ukuran citra dengan faktor skala tertentu.

**Matriks Penskalaan:**
```
[sx  0  0]
[ 0 sy  0]
[ 0  0  1]
```

**Parameter:**
- `sx`: Faktor skala horizontal
- `sy`: Faktor skala vertikal

**Perbandingan dengan `imresize`:**
- **Transformasi Affine**: Menggunakan matriks transformasi geometri
- **Image Resizing**: Menggunakan interpolasi langsung
- **Perbedaan**: Affine dapat menambahkan padding dan memiliki kontrol lebih pada interpolasi

**Pertanyaan 1:** Bandingkan hasil `scaled_affine` dan `scaled_resize`
- **Ukuran**: Keduanya menghasilkan ukuran yang sama
- **Kualitas**: Affine transformation memberikan kontrol lebih pada boundary handling
- **Kecepatan**: `imresize` biasanya lebih cepat untuk scaling sederhana

#### 2. Rotasi (Rotation) - Langkah 5-7

**Tujuan:** Memutar citra dengan sudut tertentu.

**Matriks Rotasi:**
```
[cos(θ) -sin(θ) 0]
[sin(θ)  cos(θ) 0]
[  0       0    1]
```

**Parameter:**
- `θ`: Sudut rotasi dalam derajat
- Positif = counter-clockwise
- Negatif = clockwise

**Perhitungan Output Size:**
Untuk menampung seluruh citra yang dirotasi:
```python
new_width = |h×sin(θ)| + |w×cos(θ)|
new_height = |h×cos(θ)| + |w×sin(θ)|
```

**Pertanyaan 2:** Bandingkan `rotated_affine` dan `rotated_opencv`
- Keduanya menggunakan prinsip yang sama
- Perbedaan pada implementasi boundary handling dan pusat rotasi

#### 3. Translasi (Translation) - Langkah 8-10

**Tujuan:** Memindahkan posisi citra dengan offset tertentu.

**Matriks Translasi:**
```
[1  0  dx]
[0  1  dy]
[0  0   1]
```

**Parameter:**
- `dx`: Pergeseran horizontal
- `dy`: Pergeseran vertikal

**Output Size Adjustment:**
```python
output_width = original_width + dx
output_height = original_height + dy
```

**Fill Value:**
- Area kosong diisi dengan nilai tertentu (biasanya 0 atau 128)

**Pertanyaan 3:** Bandingkan citra asli dan yang ditranslasi
- **Ukuran**: Citra hasil lebih besar untuk menampung offset
- **Content**: Konten sama, hanya bergeser posisi
- **Background**: Area kosong diisi dengan nilai fill

#### 4. Shearing - Langkah 11-13

**Tujuan:** Memiringkan citra dengan menggeser koordinat secara proporsional.

**Matriks Shearing:**
```
[1   shy  0]
[shx  1   0]
[0    0   1]
```

**Parameter:**
- `shx`: Shearing horizontal
- `shy`: Shearing vertikal

**Efek Shearing:**
- Mengubah bentuk persegi menjadi parallelogram
- Mempertahankan area citra
- Dapat menyebabkan koordinat negatif (perlu adjustment)

### Bagian II: Registrasi Citra

#### 1. Persiapan Citra - Langkah 14

**Tujuan:** Menyiapkan dua citra untuk proses registrasi.

**Requirements:**
- **Base Image**: Citra referensi (target)
- **Unregistered Image**: Citra yang akan diselaraskan

**Fallback Strategy:**
1. Coba gunakan citra dari assets (`cameraman.tif`, `moon.tif`, dll.)
2. Jika tidak ada, buat sample images
3. Buat versi yang ter-transformasi dari citra asli

#### 2. Pemilihan Control Points - Langkah 15

**Tujuan:** Memilih titik-titik kontrol yang berkorespondensi di kedua citra.

**Interactive Selection:**
- Menggunakan matplotlib untuk interface
- Klik pada fitur yang mudah dikenali di kedua citra
- Minimal 3 titik untuk affine, 4 untuk projective

**Tips Pemilihan:**
- Pilih corner atau edge yang jelas
- Distribusi merata di seluruh citra
- Hindari area yang blur atau tidak jelas

#### 3. Fine-tuning Control Points - Langkah 17

**Tujuan:** Memperbaiki akurasi posisi control points menggunakan template matching.

**Algoritma (setara `cpcorr` MATLAB):**
1. Ekstrak template di sekitar titik pada citra pertama
2. Cari posisi terbaik menggunakan correlation matching
3. Update posisi titik dengan hasil yang lebih akurat

**Parameter:**
- `window_size`: Ukuran template (default: 11x11)
- `search_size`: Area pencarian (default: 2x window_size)

**Pertanyaan 4:** Bandingkan `input_points_adj` dengan `input_points`
- Penyesuaian kecil berdasarkan correlation matching
- Biasanya terjadi pergeseran sub-pixel
- Meningkatkan akurasi registrasi

#### 4. Estimasi Transformasi - Langkah 18-20

**Jenis Transformasi:**

1. **Similarity Transform**:
   - 4 parameter: translasi (2), rotasi (1), skala (1)
   - Mempertahankan bentuk, mengizinkan uniform scaling
   - Minimal 2 control points

2. **Affine Transform**:
   - 6 parameter: translasi (2), rotasi (1), skala (2), shear (1)
   - Mempertahankan garis paralel
   - Minimal 3 control points

3. **Projective Transform**:
   - 8 parameter: transformasi perspektif penuh
   - Dapat menangani distorsi perspektif
   - Minimal 4 control points

#### 5. Aplikasi Registrasi - Langkah 21

**Proses:**
1. Aplikasikan transformasi yang diestimasi
2. Warp unregistered image ke koordinat base image
3. Evaluasi kualitas registrasi

**Metrik Evaluasi:**
- **Mean Squared Error (MSE)**: Rata-rata kuadrat perbedaan pixel
- **Visual Overlay**: Overlay red-green untuk melihat alignment
- **Edge Alignment**: Evaluasi alignment pada fitur edge

## Pertanyaan dan Jawaban

### Pertanyaan 1: Perbedaan scaled_affine dan scaled_resize
**Jawaban:**
- **Ukuran output**: Sama jika diatur dengan benar
- **Boundary handling**: Affine dapat menambah padding
- **Interpolation control**: Affine memberikan kontrol lebih detail
- **Performa**: imresize biasanya lebih cepat untuk scaling sederhana

### Pertanyaan 2: Perbedaan rotated_affine dan rotated_opencv
**Jawaban:**
- Keduanya menggunakan prinsip rotasi yang sama
- Perbedaan pada implementasi pusat rotasi dan boundary handling
- OpenCV optimized untuk performa, affine lebih fleksibel

### Pertanyaan 3: Perbandingan citra asli dan translasi
**Jawaban:**
- **Posisi konten**: Bergeser sesuai offset (dx, dy)
- **Ukuran canvas**: Lebih besar untuk menampung translasi
- **Area kosong**: Diisi dengan fill value (biasanya gray)

### Pertanyaan 4: Efek fine-tuning control points
**Jawaban:**
- Penyesuaian posisi sub-pixel berdasarkan correlation
- Meningkatkan akurasi registrasi secara signifikan
- Terutama berguna untuk citra dengan noise atau blur ringan

### Pertanyaan 5: Kepuasan dengan hasil registrasi
**Faktor yang mempengaruhi:**
- **Akurasi control points**: Semakin akurat, semakin baik
- **Jumlah control points**: Lebih banyak = lebih stabil (hingga batas tertentu)
- **Jenis transformasi**: Harus sesuai dengan distorsi yang ada
- **Kualitas citra**: Citra sharp lebih mudah diregistrasi
- **Distribusi points**: Sebaran merata memberikan hasil lebih baik

## Aplikasi Praktis

### Medical Imaging
- Alignment multi-temporal scans
- Registration cross-modality (MRI, CT, PET)
- Template matching untuk diagnosis

### Remote Sensing
- Multi-temporal change detection
- Image mosaicing
- Geometric correction

### Computer Vision
- Stereo vision calibration
- Panorama stitching
- Object tracking dan recognition

### Quality Control
- Industrial inspection
- Deformation analysis
- Template matching untuk defect detection

## Tips dan Best Practices

### Control Point Selection
1. **Pilih fitur yang stabil**: Corner, intersection, distinctive patterns
2. **Distribusi merata**: Jangan terpusat di satu area
3. **Avoid repetitive patterns**: Pilih fitur yang unik
4. **Check consistency**: Verifikasi korespondensi yang benar

### Transformation Model Selection
1. **Rigid**: Hanya rotasi dan translasi
2. **Similarity**: + uniform scaling
3. **Affine**: + shearing dan non-uniform scaling  
4. **Projective**: + perspective distortion

### Quality Assessment
1. **Visual inspection**: Overlay dan difference images
2. **Quantitative metrics**: MSE, RMSE, correlation
3. **Edge alignment**: Check pada fitur linear
4. **Residual analysis**: Analisis error pada control points

## Kesimpulan

Tutorial 7.2 memberikan pemahaman komprehensif tentang:

1. **Transformasi Affine**: Scaling, rotation, translation, shearing
2. **Interactive Tools**: Control point selection dengan matplotlib
3. **Fine-tuning Algorithms**: Template matching untuk akurasi
4. **Registration Pipeline**: End-to-end workflow registrasi citra
5. **Quality Assessment**: Metrik dan visualisasi evaluasi

Kombinasi ini membentuk foundation yang kuat untuk aplikasi computer vision yang lebih advanced seperti image stitching, multi-temporal analysis, dan medical image registration.
