import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

# Step 1: Load Image
# Ganti 'coloredChips.png' dengan path gambar Anda
img = cv2.imread('coloredChips.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Tampilkan gambar
plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Step 2: Konversi ke Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(10, 8))
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# Step 3: Deteksi Lingkaran dengan HoughCircles
# Parameter:
# - minDist: jarak minimum antara pusat lingkaran yang terdeteksi
# - param1: threshold untuk edge detection (Canny)
# - param2: threshold untuk akumulasi (semakin kecil = lebih banyak deteksi)
# - minRadius & maxRadius: range radius yang dicari (20-25 pixels)

def detect_circles(image, min_radius=20, max_radius=25, param1=50, param2=30, min_dist=30):
    """
    Deteksi lingkaran dalam gambar
    
    Parameters:
    - image: gambar input (grayscale)
    - min_radius: radius minimum
    - max_radius: radius maksimum
    - param1: threshold untuk Canny edge detector (lebih tinggi = edge lebih kuat)
    - param2: threshold akumulasi (lebih rendah = sensitivitas lebih tinggi)
    - min_dist: jarak minimum antara pusat lingkaran
    """
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    return circles

# Step 4: Deteksi dengan sensitivitas default
circles = detect_circles(gray, param2=30)

if circles is not None:
    circles = np.uint16(np.around(circles))
    print(f"Jumlah lingkaran terdeteksi: {len(circles[0])}")
else:
    print("Tidak ada lingkaran terdeteksi")

# Step 5: Tingkatkan sensitivitas (param2 lebih rendah)
# param2 yang lebih rendah = sensitivitas lebih tinggi
circles_sensitive = detect_circles(gray, param2=20, min_dist=25)

# Step 6: Gambar lingkaran pada gambar
def draw_circles(image, circles, color=(255, 0, 0), thickness=2):
    """
    Gambar lingkaran yang terdeteksi pada gambar
    """
    output = image.copy()
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for circle in circles[0]:
            center = (circle[0], circle[1])
            radius = circle[2]
            
            # Gambar lingkaran luar
            cv2.circle(output, center, radius, color, thickness)
            # Gambar titik pusat
            cv2.circle(output, center, 2, color, 3)
    
    return output

# Gambar hasil deteksi
if circles_sensitive is not None:
    img_with_circles = draw_circles(img_rgb, circles_sensitive, color=(255, 0, 0))
    
    plt.figure(figsize=(12, 10))
    plt.imshow(img_with_circles)
    plt.title(f'Detected Circles (Sensitivity: High) - {len(circles_sensitive[0])} circles')
    plt.axis('off')
    plt.show()
    
    print(f"Jumlah lingkaran terdeteksi: {len(circles_sensitive[0])}")

# Step 7: Deteksi objek terang (untuk chip kuning)
# Gunakan threshold adaptif atau inverse
gray_inv = cv2.bitwise_not(gray)

circles_bright = detect_circles(gray_inv, param2=20, param1=30, min_dist=25)

# Step 8: Gambar objek gelap dan terang dengan warna berbeda
img_final = img_rgb.copy()

# Gambar objek gelap (merah)
if circles_sensitive is not None:
    img_final = draw_circles(img_final, circles_sensitive, color=(255, 0, 0), thickness=2)

# Gambar objek terang (biru)
if circles_bright is not None:
    img_final = draw_circles(img_final, circles_bright, color=(0, 0, 255), thickness=2)

plt.figure(figsize=(12, 10))
plt.imshow(img_final)
plt.title('All Detected Circles (Red: Dark, Blue: Bright)')
plt.axis('off')
plt.show()

# Step 9: Fungsi lengkap untuk deteksi dengan parameter custom
def detect_all_circles(image_path, dark_sensitivity=20, bright_sensitivity=20, 
                       min_radius=20, max_radius=25):
    """
    Deteksi semua lingkaran (gelap dan terang) dalam gambar
    
    Parameters:
    - image_path: path ke gambar
    - dark_sensitivity: sensitivitas untuk objek gelap (lebih rendah = lebih sensitif)
    - bright_sensitivity: sensitivitas untuk objek terang
    - min_radius & max_radius: range radius
    
    Returns:
    - image dengan lingkaran yang terdeteksi
    - jumlah lingkaran gelap dan terang
    """
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Deteksi objek gelap
    circles_dark = detect_circles(gray, min_radius, max_radius, 
                                  param2=dark_sensitivity, min_dist=25)
    
    # Deteksi objek terang
    gray_inv = cv2.bitwise_not(gray)
    circles_bright = detect_circles(gray_inv, min_radius, max_radius, 
                                   param2=bright_sensitivity, min_dist=25)
    
    # Gambar hasil
    result = img_rgb.copy()
    
    n_dark = 0
    n_bright = 0
    
    if circles_dark is not None:
        result = draw_circles(result, circles_dark, color=(255, 0, 0))
        n_dark = len(circles_dark[0])
    
    if circles_bright is not None:
        result = draw_circles(result, circles_bright, color=(0, 0, 255))
        n_bright = len(circles_bright[0])
    
    print(f"Lingkaran gelap: {n_dark}")
    print(f"Lingkaran terang: {n_bright}")
    print(f"Total: {n_dark + n_bright}")
    
    return result, n_dark, n_bright

# Contoh penggunaan
# result, n_dark, n_bright = detect_all_circles('coloredChips.png')
# plt.figure(figsize=(12, 10))
# plt.imshow(result)
# plt.title(f'Total: {n_dark + n_bright} circles')
# plt.show()
