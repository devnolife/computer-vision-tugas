import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from skimage import io, transform
from PIL import Image
import os

class ImageProcessor:
    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image_path = image_path
        self.crop_coords = None
        
    def interactive_crop(self):
        """Interactive cropping similar to MATLAB's imtool crop function"""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(self.original_image, cmap='gray')
        ax.set_title('Pilih area untuk dipotong (drag untuk memilih, tekan Enter jika selesai)')
        
        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            
            # Store coordinates in MATLAB convention (column, row)
            self.crop_coords = {
                'x1': min(x1, x2), 'y1': min(y1, y2),
                'x2': max(x1, x2), 'y2': max(y1, y2),
                'xmin': min(x1, x2), 'ymin': min(y1, y2),
                'width': abs(x2 - x1), 'height': abs(y2 - y1)
            }
            
            print(f"Koordinat kiri atas (x1, y1): ({self.crop_coords['x1']}, {self.crop_coords['y1']})")
            print(f"Koordinat kanan bawah (x2, y2): ({self.crop_coords['x2']}, {self.crop_coords['y2']})")
            print(f"Persegi panjang crop [xmin ymin width height]: [{self.crop_coords['xmin']} {self.crop_coords['ymin']} {self.crop_coords['width']} {self.crop_coords['height']}]")
        
        selector = RectangleSelector(ax, onselect, useblit=True, button=[1], 
                                   minspanx=5, minspany=5, spancoords='pixels', 
                                   interactive=True)
        
        plt.show()
        return self.crop_coords
    
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

# Tutorial 7.1 Implementation
def tutorial_7_1():
    """Complete Tutorial 7.1 implementation"""
    
    # Load cameraman image (you'll need to provide this image)
    # For demonstration, let's create a sample image or load any grayscale image
    print("=== TUTORIAL 7.1: PEMOTONGAN, PENGUBAHAN UKURAN, PEMBALIKAN, DAN ROTASI CITRA ===\n")
    
    # Step 1-3: Interactive cropping
    print("Langkah 1-3: Pemotongan Citra Secara Interaktif")
    print("Pastikan Anda memiliki file 'cameraman.tif' atau file citra serupa")
    
    # Initialize processor (replace with your image path)
    image_path = '../assets/cameraman.tif'  # Use cameraman2.tif from assets
    if not os.path.exists(image_path):
        # Try alternative paths
        alt_paths = ['../assets/moon.tif', '../assets/lena.png', '../assets/tire.tif']
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                image_path = alt_path
                print(f"Menggunakan citra alternatif: {image_path}")
                break
        else:
            print(f"Peringatan: Tidak ditemukan citra yang sesuai. Membuat citra contoh untuk demonstrasi.")
            # Create a sample image for demonstration
            sample_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            cv2.imwrite('sample_image.tif', sample_img)
            image_path = 'sample_image.tif'
    
    processor = ImageProcessor(image_path)
    
    # Display original image
    plt.figure(figsize=(8, 6))
    plt.imshow(processor.original_image, cmap='gray')
    plt.title('Citra Asli')
    plt.axis('off')
    plt.show()
    
    # Step 4: Manual cropping with coordinates
    print("\nLangkah 4-7: Pemotongan manual dengan koordinat yang ditentukan")
    # Example coordinates (replace with your recorded values)
    x1, y1, x2, y2 = 186, 105, 211, 159
    manual_coords = {
        'xmin': x1, 'ymin': y1,
        'width': x2-x1, 'height': y2-y1
    }
    
    cropped_img = processor.crop_image(manual_coords)
    if cropped_img is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(cropped_img, cmap='gray')
        plt.title('Citra yang Dipotong Manual')
        plt.axis('off')
        plt.show()
    
    # Step 8-10: Image enlargement with different interpolations
    print("\nLangkah 8-10: Pembesaran citra dengan interpolasi yang berbeda")
    
    enlarged_cubic = processor.resize_image(3, 'cubic')
    enlarged_nearest = processor.resize_image(3, 'nearest') 
    enlarged_bilinear = processor.resize_image(3, 'bilinear')
    
    # Display enlarged images
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].imshow(processor.original_image, cmap='gray')
    axes[0,0].set_title('Citra Asli')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(enlarged_cubic, cmap='gray')
    axes[0,1].set_title('Diperbesar dengan Interpolasi Bikubik')
    axes[0,1].axis('off')
    
    axes[1,0].imshow(enlarged_nearest, cmap='gray')
    axes[1,0].set_title('Diperbesar dengan Interpolasi Tetangga Terdekat')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(enlarged_bilinear, cmap='gray')
    axes[1,1].set_title('Diperbesar dengan Interpolasi Bilinear')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nPertanyaan 2: Bandingkan ketiga hasil pembesaran citra.")
    print("- Tetangga Terdekat: Terlihat kotak-kotak (pixelated), tepi tajam, tidak ada nilai pixel baru")
    print("- Bilinear: Lebih halus daripada tetangga terdekat, sedikit blur")
    print("- Bikubik: Hasil paling halus, kualitas terbaik tapi komputasi paling berat")
    
    # Step 11-12: Image shrinking by subsampling
    print("\nLangkah 11-12: Pengecilan citra dengan subsampling")
    shrunk_subsample = processor.resize_by_subsampling(2)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(processor.original_image, cmap='gray')
    plt.title('Citra Asli')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(shrunk_subsample, cmap='gray')
    plt.title('Dikecilkan dengan Subsampling')
    plt.axis('off')
    plt.show()
    
    print("\nPertanyaan 3: Bagaimana cara kita mengecilkan citra?")
    print("Kita mengambil setiap pixel ke-2 dalam kedua dimensi (I[::2, ::2])")
    print("\nPertanyaan 4: Keterbatasan teknik ini:")
    print("- Hanya bisa mengubah ukuran dengan faktor integer")
    print("- Dapat menyebabkan aliasing artifacts")
    print("- Kehilangan informasi yang tidak dapat dipulihkan")
    
    # Step 13: Shrinking with interpolation
    print("\nLangkah 13: Pengecilan dengan metode interpolasi yang berbeda")
    shrunk_nearest = processor.resize_image(0.5, 'nearest')
    shrunk_bilinear = processor.resize_image(0.5, 'bilinear') 
    shrunk_cubic = processor.resize_image(0.5, 'cubic')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(shrunk_nearest, cmap='gray')
    axes[0].set_title('Interpolasi Tetangga Terdekat')
    axes[0].axis('off')
    
    axes[1].imshow(shrunk_bilinear, cmap='gray')
    axes[1].set_title('Interpolasi Bilinear')
    axes[1].axis('off')
    
    axes[2].imshow(shrunk_cubic, cmap='gray')
    axes[2].set_title('Interpolasi Bikubik')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Step 14-16: Image flipping
    print("\nLangkah 14-16: Pembalikan citra")
    flipped_ud = processor.flip_image('ud')
    flipped_lr = processor.flip_image('lr')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(processor.original_image, cmap='gray')
    axes[0].set_title('Citra Asli')
    axes[0].axis('off')
    
    axes[1].imshow(flipped_ud, cmap='gray')
    axes[1].set_title('Dibalik Atas-Bawah')
    axes[1].axis('off')
    
    axes[2].imshow(flipped_lr, cmap='gray')
    axes[2].set_title('Dibalik Kiri-Kanan')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Step 17-20: Image rotation
    print("\nLangkah 17-20: Rotasi citra")
    
    # Load eight.tif or use alternative image
    eight_path = '../assets/moon.tif'  # Use moon.tif as alternative
    if not os.path.exists(eight_path):
        # Try other alternatives
        alt_paths = ['../assets/pout.tif', '../assets/tire.tif', '../assets/lindsay.tif']
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                eight_path = alt_path
                print(f"Menggunakan citra alternatif untuk rotasi: {eight_path}")
                break
        else:
            # Create a sample image with circular objects
            eight_img = np.zeros((256, 256), dtype=np.uint8)
            cv2.circle(eight_img, (100, 80), 30, 255, -1)
            cv2.circle(eight_img, (156, 176), 30, 255, -1)
            cv2.imwrite('eight_sample.tif', eight_img)
            eight_path = 'eight_sample.tif'
            print("Membuat citra contoh untuk demo rotasi")
    
    eight_processor = ImageProcessor(eight_path)
    
    # Rotate 35 degrees counterclockwise
    rotated_35 = eight_processor.rotate_image(35)
    rotated_35_bilinear = eight_processor.rotate_image(35, 'bilinear')
    rotated_35_cropped = eight_processor.rotate_image(35, 'bilinear', crop_output=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].imshow(eight_processor.original_image, cmap='gray')
    axes[0,0].set_title('Citra Asli')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(rotated_35, cmap='gray')
    axes[0,1].set_title('Dirotasi 35° (Bikubik)')
    axes[0,1].axis('off')
    
    axes[1,0].imshow(rotated_35_bilinear, cmap='gray')
    axes[1,0].set_title('Dirotasi 35° (Bilinear)')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(rotated_35_cropped, cmap='gray')
    axes[1,1].set_title('Dirotasi 35° (Dipotong)')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nPertanyaan 5: Perbandingan ukuran")
    print(f"Ukuran citra asli: {eight_processor.original_image.shape}")
    print(f"Ukuran citra yang dirotasi: {rotated_35.shape}")
    print("Ukurannya berbeda karena rotasi memerlukan canvas yang lebih besar untuk menampung seluruh citra yang dirotasi.")
    
    print(f"\nPertanyaan 6: Untuk merotasi 35° searah jarum jam, gunakan sudut = -35")
    
    print(f"\nPertanyaan 7: Interpolasi bilinear menghasilkan tepi yang lebih halus dibandingkan tetangga terdekat.")
    
    print(f"\nPertanyaan 8: Pengaturan crop mempertahankan ukuran citra asli dengan memotong hasil output.")

# Run the tutorial
if __name__ == "__main__":
    tutorial_7_1()
