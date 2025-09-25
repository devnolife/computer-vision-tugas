import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from skimage import io, transform
from PIL import Image
import os

# Buat folder hasil jika belum ada
os.makedirs('hasil', exist_ok=True)

class ImageProcessor:
    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image_path = image_path
        self.crop_coords = None
        
    def crop_image(self, coords=None):
        """Crop image using specified coordinates"""
        if coords is None:
            coords = self.crop_coords
            
        if coords is None:
            print("Tidak ada koordinat crop yang tersedia. Jalankan interactive_crop() terlebih dahulu.")
            return None
            
        # Python uses [y1:y2, x1:x2] indexing
        cropped = self.original_image[coords['ymin']:coords['ymin']+coords['height'], 
                                    coords['xmin']:coords['xmin']+coords['width']]
        return cropped
    
    def resize_image(self, scale_factor, interpolation='cubic'):
        """Resize image with different interpolation methods"""
        if interpolation == 'nearest':
            interp = cv2.INTER_NEAREST
        elif interpolation == 'bilinear':
            interp = cv2.INTER_LINEAR
        elif interpolation == 'cubic':
            interp = cv2.INTER_CUBIC
        
        new_size = (int(self.original_image.shape[1] * scale_factor),
                   int(self.original_image.shape[0] * scale_factor))
        resized = cv2.resize(self.original_image, new_size, interpolation=interp)
        return resized
    
    def resize_by_subsampling(self, factor=2):
        """Resize by subsampling (equivalent to MATLAB's indexing method)"""
        return self.original_image[::factor, ::factor]
    
    def flip_image(self, direction):
        """Flip image upside down or left-right"""
        if direction == 'ud':  # upside down (equivalent to flipud)
            return cv2.flip(self.original_image, 0)
        elif direction == 'lr':  # left-right (equivalent to fliplr)
            return cv2.flip(self.original_image, 1)
    
    def rotate_image(self, angle, interpolation='cubic', crop_output=False):
        """Rotate image by specified angle"""
        if interpolation == 'nearest':
            interp = cv2.INTER_NEAREST
        elif interpolation == 'bilinear':
            interp = cv2.INTER_LINEAR
        elif interpolation == 'cubic':
            interp = cv2.INTER_CUBIC
        
        if crop_output:
            # Rotate and crop to original size
            rotated = transform.rotate(self.original_image, angle, preserve_range=True)
            rotated = rotated.astype(np.uint8)
        else:
            # Rotate with expanded canvas (similar to MATLAB's default behavior)
            center = (self.original_image.shape[1]//2, self.original_image.shape[0]//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate new dimensions
            cos_theta = abs(rotation_matrix[0, 0])
            sin_theta = abs(rotation_matrix[0, 1])
            new_width = int((self.original_image.shape[0] * sin_theta) + 
                           (self.original_image.shape[1] * cos_theta))
            new_height = int((self.original_image.shape[0] * cos_theta) + 
                            (self.original_image.shape[1] * sin_theta))
            
            # Adjust rotation matrix for new center
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            rotated = cv2.warpAffine(self.original_image, rotation_matrix, 
                                   (new_width, new_height), flags=interp)
        
        return rotated

def save_results_7_1():
    """Generate and save Tutorial 7.1 results"""
    print("=== Menghasilkan dan Menyimpan Hasil Tutorial 7.1 ===\n")
    
    # Initialize processor
    image_path = '../assets/cameraman2.tif'
    if not os.path.exists(image_path):
        alt_paths = ['../assets/moon.tif', '../assets/lena.png', '../assets/tire.tif']
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                image_path = alt_path
                print(f"Menggunakan citra alternatif: {image_path}")
                break
    
    processor = ImageProcessor(image_path)
    
    # 1. Simpan citra asli
    plt.figure(figsize=(8, 6))
    plt.imshow(processor.original_image, cmap='gray')
    plt.title('Citra Asli', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('hasil/01_citra_asli.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Manual cropping
    x1, y1, x2, y2 = 100, 80, 200, 180  # Koordinat crop yang disesuaikan
    manual_coords = {
        'xmin': x1, 'ymin': y1,
        'width': x2-x1, 'height': y2-y1
    }
    
    cropped_img = processor.crop_image(manual_coords)
    if cropped_img is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(cropped_img, cmap='gray')
        plt.title('Citra yang Dipotong Manual', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('hasil/02_citra_crop.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Image enlargement with different interpolations
    enlarged_cubic = processor.resize_image(3, 'cubic')
    enlarged_nearest = processor.resize_image(3, 'nearest') 
    enlarged_bilinear = processor.resize_image(3, 'bilinear')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].imshow(processor.original_image, cmap='gray')
    axes[0,0].set_title('Citra Asli', fontsize=12, fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(enlarged_cubic, cmap='gray')
    axes[0,1].set_title('Diperbesar dengan Interpolasi Bikubik', fontsize=12, fontweight='bold')
    axes[0,1].axis('off')
    
    axes[1,0].imshow(enlarged_nearest, cmap='gray')
    axes[1,0].set_title('Diperbesar dengan Interpolasi Tetangga Terdekat', fontsize=12, fontweight='bold')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(enlarged_bilinear, cmap='gray')
    axes[1,1].set_title('Diperbesar dengan Interpolasi Bilinear', fontsize=12, fontweight='bold')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('hasil/03_perbandingan_interpolasi.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Image shrinking by subsampling
    shrunk_subsample = processor.resize_by_subsampling(2)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(processor.original_image, cmap='gray')
    axes[0].set_title('Citra Asli', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(shrunk_subsample, cmap='gray')
    axes[1].set_title('Dikecilkan dengan Subsampling', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('hasil/04_subsampling.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Shrinking with interpolation
    shrunk_nearest = processor.resize_image(0.5, 'nearest')
    shrunk_bilinear = processor.resize_image(0.5, 'bilinear') 
    shrunk_cubic = processor.resize_image(0.5, 'cubic')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(shrunk_nearest, cmap='gray')
    axes[0].set_title('Interpolasi Tetangga Terdekat', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(shrunk_bilinear, cmap='gray')
    axes[1].set_title('Interpolasi Bilinear', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(shrunk_cubic, cmap='gray')
    axes[2].set_title('Interpolasi Bikubik', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('hasil/05_pengecilan_interpolasi.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Image flipping
    flipped_ud = processor.flip_image('ud')
    flipped_lr = processor.flip_image('lr')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(processor.original_image, cmap='gray')
    axes[0].set_title('Citra Asli', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(flipped_ud, cmap='gray')
    axes[1].set_title('Dibalik Atas-Bawah', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(flipped_lr, cmap='gray')
    axes[2].set_title('Dibalik Kiri-Kanan', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('hasil/06_pembalikan.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Image rotation - buat citra untuk rotasi
    moon_path = '../assets/moon.tif'
    if os.path.exists(moon_path):
        moon_processor = ImageProcessor(moon_path)
    else:
        # Buat sample citra untuk rotasi
        sample_img = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(sample_img, (100, 80), 30, 255, -1)
        cv2.circle(sample_img, (156, 176), 30, 255, -1)
        cv2.rectangle(sample_img, (80, 120), (180, 140), 128, -1)
        cv2.imwrite('hasil/sample_rotation.tif', sample_img)
        moon_processor = ImageProcessor('hasil/sample_rotation.tif')
    
    # Rotate 35 degrees counterclockwise
    rotated_35 = moon_processor.rotate_image(35)
    rotated_35_bilinear = moon_processor.rotate_image(35, 'bilinear')
    rotated_35_cropped = moon_processor.rotate_image(35, 'bilinear', crop_output=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].imshow(moon_processor.original_image, cmap='gray')
    axes[0,0].set_title('Citra Asli', fontsize=12, fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(rotated_35, cmap='gray')
    axes[0,1].set_title('Dirotasi 35° (Bikubik)', fontsize=12, fontweight='bold')
    axes[0,1].axis('off')
    
    axes[1,0].imshow(rotated_35_bilinear, cmap='gray')
    axes[1,0].set_title('Dirotasi 35° (Bilinear)', fontsize=12, fontweight='bold')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(rotated_35_cropped, cmap='gray')
    axes[1,1].set_title('Dirotasi 35° (Dipotong)', fontsize=12, fontweight='bold')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('hasil/07_rotasi.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Semua hasil Tutorial 7.1 berhasil disimpan di folder 'hasil'")

if __name__ == "__main__":
    save_results_7_1()
