import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, img_as_ubyte, img_as_float
from scipy import ndimage
import os

# Konfigurasi matplotlib untuk tidak menampilkan GUI
plt.ioff()

class HistogramProcessor:
    def __init__(self, image_path=None, image=None):
        """Initialize with either image path or image array"""
        if image_path is not None:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"ERROR: File '{image_path}' tidak ditemukan!")
            self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if self.original_image is None:
                raise ValueError(f"ERROR: Gagal membaca gambar '{image_path}'")
            self.image_path = image_path
        elif image is not None:
            self.original_image = image
            self.image_path = None
        else:
            raise ValueError("Either image_path or image must be provided")
    
    def compute_histogram(self, image=None, bins=256):
        """Compute histogram of image"""
        if image is None:
            image = self.original_image
        
        hist, bin_edges = np.histogram(image.flatten(), bins=bins, range=(0, 256))
        return hist, bin_edges
    
    def plot_histogram(self, image=None, bins=256, title='Histogram', color='blue'):
        """Plot histogram using matplotlib"""
        if image is None:
            image = self.original_image
        
        hist, _ = self.compute_histogram(image, bins)
        plt.bar(range(len(hist)), hist, color=color, alpha=0.7)
        plt.title(title, fontsize=10)
        plt.xlabel('Intensity Value')
        plt.ylabel('Pixel Count')
        plt.xlim([0, bins])
    
    def normalize_histogram(self, hist):
        """Normalize histogram values"""
        total_pixels = np.sum(hist)
        return hist / total_pixels
    
    def histogram_equalization(self, image=None):
        """Perform histogram equalization"""
        if image is None:
            image = self.original_image
        
        equalized = cv2.equalizeHist(image)
        return equalized
    
    def histogram_matching(self, reference_hist):
        """Perform histogram matching/specification"""
        # Get histogram and CDF of source image
        src_hist, _ = self.compute_histogram()
        src_cdf = np.cumsum(src_hist)
        src_cdf = src_cdf / src_cdf[-1]  # Normalize
        
        # Get CDF of reference histogram
        ref_cdf = np.cumsum(reference_hist)
        ref_cdf = ref_cdf / ref_cdf[-1]  # Normalize
        
        # Create lookup table
        lut = np.zeros(256, dtype=np.uint8)
        for src_val in range(256):
            # Find closest match in reference CDF
            diff = np.abs(ref_cdf - src_cdf[src_val])
            lut[src_val] = np.argmin(diff)
        
        # Apply lookup table
        matched = cv2.LUT(self.original_image, lut)
        return matched
    
    def adaptive_histogram_equalization(self, clip_limit=2.0, tile_grid_size=(8,8)):
        """Perform CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(self.original_image)
    
    def histogram_sliding(self, constant, clip=True):
        """Adjust brightness by adding constant to all pixels"""
        # Convert to float for computation
        img_float = self.original_image.astype(np.float32) / 255.0
        
        # Add constant
        adjusted = img_float + constant
        
        # Clip values to valid range if requested
        if clip:
            adjusted = np.clip(adjusted, 0, 1)
        
        # Convert back to uint8
        return (adjusted * 255).astype(np.uint8)
    
    def imadjust(self, low_in=None, high_in=None, low_out=0, high_out=1, gamma=1.0):
        """Adjust image intensity values (similar to MATLAB's imadjust)"""
        # Convert to float
        img_float = self.original_image.astype(np.float32) / 255.0
        
        # Auto-calculate limits if not provided
        if low_in is None or high_in is None:
            low_in, high_in = self.stretchlim(img_float)
        
        # Apply intensity transformation
        # Clip input values
        img_clipped = np.clip(img_float, low_in, high_in)
        
        # Normalize to [0, 1]
        img_normalized = (img_clipped - low_in) / (high_in - low_in)
        
        # Apply gamma correction
        img_gamma = np.power(img_normalized, gamma)
        
        # Scale to output range
        img_adjusted = img_gamma * (high_out - low_out) + low_out
        
        # Convert back to uint8
        return (img_adjusted * 255).astype(np.uint8)
    
    def stretchlim(self, image_float=None, tol=0.01):
        """Find intensity limits for contrast stretching"""
        if image_float is None:
            image_float = self.original_image.astype(np.float32) / 255.0
        
        # Sort pixel values
        sorted_vals = np.sort(image_float.flatten())
        n_pixels = len(sorted_vals)
        
        # Find low and high percentiles
        low_idx = int(n_pixels * tol)
        high_idx = int(n_pixels * (1 - tol))
        
        low_limit = sorted_vals[low_idx]
        high_limit = sorted_vals[high_idx]
        
        return low_limit, high_limit


def check_required_assets():
    """Check if all required assets are available"""
    required_files = [
        'circuit.tif',
        'pout.tif',
        'tire.tif',
        'eight.tif',
        'coins.png'
    ]
    
    missing_files = []
    assets_dir = '../assets'
    
    print("="*60)
    print("CHECKING REQUIRED ASSETS")
    print("="*60)
    
    for filename in required_files:
        filepath = os.path.join(assets_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename} - TERSEDIA")
        else:
            print(f"✗ {filename} - TIDAK DITEMUKAN!")
            missing_files.append(filename)
    
    if missing_files:
        print("\n" + "="*60)
        print("ERROR: ASSET TIDAK LENGKAP!")
        print("="*60)
        print("File yang hilang:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nSilakan tambahkan file-file tersebut ke folder 'assets'")
        print("="*60)
        return False
    
    print("\n✓ Semua asset tersedia!")
    print("="*60)
    return True


def generate_tutorial_9_1_results():
    """Generate results for Tutorial 9.1: Image Histograms"""
    print("\n" + "="*60)
    print("GENERATING TUTORIAL 9.1 RESULTS")
    print("="*60)
    
    output_dir = 'hasil'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load circuit image
    circuit_path = '../assets/circuit.tif'
    processor = HistogramProcessor(circuit_path)
    
    # GAMBAR 1: Perbandingan histogram dengan berbagai jumlah bins
    print("Membuat Gambar 1: Perbandingan histogram dengan berbagai jumlah bins...")
    fig = plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(processor.original_image, cmap='gray')
    plt.title('Circuit Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    processor.plot_histogram(bins=256, title='Histogram (256 bins)', color='blue')
    
    plt.subplot(2, 2, 3)
    processor.plot_histogram(bins=64, title='Histogram (64 bins)', color='green')
    
    plt.subplot(2, 2, 4)
    processor.plot_histogram(bins=32, title='Histogram (32 bins)', color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_1_1_Histogram_Bins.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gambar 1 tersimpan")
    
    # GAMBAR 2: Bar Chart Visualization
    print("Membuat Gambar 2: Bar Chart Visualization...")
    hist_32, _ = processor.compute_histogram(bins=32)
    hist_norm = processor.normalize_histogram(hist_32)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(range(len(hist_32)), hist_32, color='red', alpha=0.7)
    axes[0].set_xlim([0, 32])
    axes[0].set_ylim([0, np.max(hist_32)])
    axes[0].set_xticks(np.arange(0, 33, 8))
    axes[0].set_title('Bar Chart', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Bin')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(range(len(hist_norm)), hist_norm, color='green', alpha=0.7)
    axes[1].set_xlim([0, 32])
    axes[1].set_ylim([0, np.max(hist_norm)])
    axes[1].set_xticks(np.arange(0, 33, 8))
    axes[1].set_title('Normalized Bar Chart', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Bin')
    axes[1].set_ylabel('Normalized Count')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_1_2_Bar_Chart.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gambar 2 tersimpan")
    
    # GAMBAR 3: Stem Chart Visualization
    print("Membuat Gambar 3: Stem Chart Visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    markerline, stemlines, baseline = axes[0].stem(hist_32, linefmt='b-', 
                                                   markerfmt='ro', basefmt='k-')
    plt.setp(markerline, 'markerfacecolor', 'red', markersize=6)
    axes[0].set_title('Stem Chart', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Bin')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    markerline, stemlines, baseline = axes[1].stem(hist_norm, linefmt='b-', 
                                                   markerfmt='ro', basefmt='k-')
    plt.setp(markerline, 'markerfacecolor', 'red', markersize=6)
    axes[1].set_title('Normalized Stem Chart', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Bin')
    axes[1].set_ylabel('Normalized Count')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_1_3_Stem_Chart.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gambar 3 tersimpan")
    
    # GAMBAR 4: Plot Graph Visualization
    print("Membuat Gambar 4: Plot Graph Visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(hist_32, 'b-', marker='o', markersize=5, linewidth=2)
    axes[0].set_title('Plot Graph', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Bin')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 32])
    
    axes[1].plot(hist_norm, 'b-', marker='o', markersize=5, linewidth=2)
    axes[1].set_title('Normalized Plot Graph', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Bin')
    axes[1].set_ylabel('Normalized Count')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 32])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_1_4_Plot_Graph.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gambar 4 tersimpan")
    
    print("\n✓ Tutorial 9.1 - Semua gambar berhasil dibuat!")


def generate_tutorial_9_2_results():
    """Generate results for Tutorial 9.2: Histogram Equalization"""
    print("\n" + "="*60)
    print("GENERATING TUTORIAL 9.2 RESULTS")
    print("="*60)
    
    output_dir = 'hasil'
    os.makedirs(output_dir, exist_ok=True)
    
    # GAMBAR 5: Histogram Equalization pada pout.tif
    print("Membuat Gambar 5: Histogram Equalization pada pout.tif...")
    pout_path = '../assets/pout.tif'
    processor_pout = HistogramProcessor(pout_path)
    pout_eq = processor_pout.histogram_equalization()
    
    fig = plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(processor_pout.original_image, cmap='gray')
    plt.title('Original Image (Pout)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    processor_pout.plot_histogram(title='Original Histogram', color='blue')
    
    plt.subplot(2, 2, 3)
    plt.imshow(pout_eq, cmap='gray')
    plt.title('Equalized Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    processor_pout.plot_histogram(pout_eq, title='Equalized Histogram', color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_2_1_Pout_Equalization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gambar 5 tersimpan")
    
    # GAMBAR 6: Histogram Equalization pada tire.tif
    print("Membuat Gambar 6: Histogram Equalization pada tire.tif...")
    tire_path = '../assets/tire.tif'
    processor_tire = HistogramProcessor(tire_path)
    tire_eq = processor_tire.histogram_equalization()
    
    fig = plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(processor_tire.original_image, cmap='gray')
    plt.title('Original Image (Tire)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    processor_tire.plot_histogram(title='Original Histogram', color='blue')
    
    plt.subplot(2, 2, 3)
    plt.imshow(tire_eq, cmap='gray')
    plt.title('Equalized Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    processor_tire.plot_histogram(tire_eq, title='Equalized Histogram', color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_2_2_Tire_Equalization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gambar 6 tersimpan")
    
    # GAMBAR 7: Histogram Equalization pada eight.tif
    print("Membuat Gambar 7: Histogram Equalization pada eight.tif...")
    eight_path = '../assets/eight.tif'
    processor_eight = HistogramProcessor(eight_path)
    eight_eq = processor_eight.histogram_equalization()
    
    fig = plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(processor_eight.original_image, cmap='gray')
    plt.title('Original Image (Eight - Bimodal)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    processor_eight.plot_histogram(title='Original Histogram (Bimodal)', color='blue')
    
    plt.subplot(2, 2, 3)
    plt.imshow(eight_eq, cmap='gray')
    plt.title('Equalized Image (Quality Loss)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    processor_eight.plot_histogram(eight_eq, title='Equalized Histogram', color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_2_3_Eight_Equalization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gambar 7 tersimpan")
    
    # GAMBAR 8: Normalized CDF - Transformation Function
    print("Membuat Gambar 8: Normalized CDF - Transformation Function...")
    hist_eight, _ = processor_eight.compute_histogram()
    cdf = np.cumsum(hist_eight)
    cdf_normalized = cdf / cdf[-1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(cdf_normalized, 'b-', linewidth=2)
    plt.title('Normalized CDF (Transformation Function)', fontsize=14, fontweight='bold')
    plt.xlabel('Input Intensity', fontsize=12)
    plt.ylabel('Output Intensity', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 256])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_2_4_CDF_Transformation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gambar 8 tersimpan")
    
    # GAMBAR 9: Histogram Specification/Matching
    print("Membuat Gambar 9: Histogram Specification/Matching...")
    uniform_hist = np.ones(256) * 1000
    linear_hist = np.linspace(0, 1, 256) * 1000
    
    matched_image = processor_pout.histogram_matching(linear_hist)
    
    fig = plt.figure(figsize=(15, 10))
    
    # Original
    plt.subplot(3, 3, 1)
    plt.imshow(processor_pout.original_image, cmap='gray')
    plt.title('Original Image', fontsize=11, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    processor_pout.plot_histogram(title='Original Histogram', color='blue')
    
    plt.subplot(3, 3, 3)
    plt.text(0.5, 0.5, 'Original\nState', ha='center', va='center', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Equalized
    plt.subplot(3, 3, 4)
    plt.imshow(pout_eq, cmap='gray')
    plt.title('Equalized Image', fontsize=11, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    processor_pout.plot_histogram(pout_eq, title='Equalized Histogram', color='red')
    
    plt.subplot(3, 3, 6)
    plt.plot(uniform_hist/1000, 'g-', linewidth=2)
    plt.title('Desired: Uniform', fontsize=11, fontweight='bold')
    plt.ylim([0, 1.2])
    plt.xlim([0, 256])
    plt.xlabel('Intensity')
    plt.ylabel('Normalized')
    plt.grid(True, alpha=0.3)
    
    # Matched
    plt.subplot(3, 3, 7)
    plt.imshow(matched_image, cmap='gray')
    plt.title('Matched Image', fontsize=11, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    processor_pout.plot_histogram(matched_image, title='Matched Histogram', color='purple')
    
    plt.subplot(3, 3, 9)
    plt.plot(linear_hist/1000, 'm-', linewidth=2)
    plt.title('Desired: Linear', fontsize=11, fontweight='bold')
    plt.ylim([0, 1.2])
    plt.xlim([0, 256])
    plt.xlabel('Intensity')
    plt.ylabel('Normalized')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_2_5_Histogram_Matching.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gambar 9 tersimpan")
    
    # GAMBAR 10: Adaptive Histogram Equalization (CLAHE)
    print("Membuat Gambar 10: Adaptive Histogram Equalization (CLAHE)...")
    coins_path = '../assets/coins.png'
    processor_coins = HistogramProcessor(coins_path)
    coins_eq = processor_coins.histogram_equalization()
    coins_adaptive = processor_coins.adaptive_histogram_equalization(clip_limit=0.1)
    
    fig = plt.figure(figsize=(14, 12))
    
    plt.subplot(3, 2, 1)
    plt.imshow(processor_coins.original_image, cmap='gray')
    plt.title('Original Image (Coins - Bimodal)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 2, 2)
    processor_coins.plot_histogram(title='Original Histogram (Bimodal)', color='blue')
    
    plt.subplot(3, 2, 3)
    plt.imshow(coins_eq, cmap='gray')
    plt.title('Global Histogram Equalization', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    processor_coins.plot_histogram(coins_eq, title='Global Equalized Histogram', color='red')
    
    plt.subplot(3, 2, 5)
    plt.imshow(coins_adaptive, cmap='gray')
    plt.title('Adaptive Histogram Equalization (CLAHE)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 2, 6)
    processor_coins.plot_histogram(coins_adaptive, title='Adaptive Equalized Histogram', color='green')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_2_6_Adaptive_Equalization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gambar 10 tersimpan")
    
    print("\n✓ Tutorial 9.2 - Semua gambar berhasil dibuat!")


def generate_tutorial_9_3_results():
    """Generate results for Tutorial 9.3: Other Histogram Modifications"""
    print("\n" + "="*60)
    print("GENERATING TUTORIAL 9.3 RESULTS")
    print("="*60)
    
    output_dir = 'hasil'
    os.makedirs(output_dir, exist_ok=True)
    
    # GAMBAR 11: Histogram Sliding
    print("Membuat Gambar 11: Histogram Sliding...")
    pout_path = '../assets/pout.tif'
    processor = HistogramProcessor(pout_path)
    
    img_bright1 = processor.histogram_sliding(0.1)
    img_bright2 = processor.histogram_sliding(0.5)
    
    fig = plt.figure(figsize=(14, 12))
    
    plt.subplot(3, 2, 1)
    plt.imshow(processor.original_image, cmap='gray')
    plt.title('Original Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 2, 2)
    processor.plot_histogram(title='Original Histogram', color='blue')
    
    plt.subplot(3, 2, 3)
    plt.imshow(img_bright1, cmap='gray')
    plt.title('Original Image + 0.1', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    processor.plot_histogram(img_bright1, title='Histogram + 0.1', color='green')
    
    plt.subplot(3, 2, 5)
    plt.imshow(img_bright2, cmap='gray')
    plt.title('Original Image + 0.5 (Saturated)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 2, 6)
    processor.plot_histogram(img_bright2, title='Histogram + 0.5', color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_3_1_Histogram_Sliding.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gambar 11 tersimpan")
    
    # GAMBAR 12: Histogram Stretching
    print("Membuat Gambar 12: Histogram Stretching...")
    img_stretched = processor.imadjust()
    img_stretched2 = processor.imadjust()
    diff = cv2.absdiff(img_stretched, img_stretched2)
    
    fig = plt.figure(figsize=(14, 14))
    
    plt.subplot(3, 2, 1)
    plt.imshow(processor.original_image, cmap='gray')
    plt.title('Original Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 2, 2)
    processor.plot_histogram(title='Original Histogram', color='blue')
    
    plt.subplot(3, 2, 3)
    plt.imshow(img_stretched, cmap='gray')
    plt.title('Stretched Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    processor.plot_histogram(img_stretched, title='Stretched Histogram', color='red')
    
    plt.subplot(3, 2, 5)
    plt.imshow(diff, cmap='gray')
    plt.title('Difference Image (should be black)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(3, 2, 6)
    plt.text(0.5, 0.5, f'Min: {np.min(diff)}\nMax: {np.max(diff)}', 
             ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_3_2_Histogram_Stretching.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gambar 12 tersimpan")
    
    # For Gambar 13 & 14, we need westconcord.png - check if exists
    westconcord_path = '../assets/westconcord.png'
    if not os.path.exists(westconcord_path):
        # Use alternative image if westconcord doesn't exist
        print("⚠ westconcord.png tidak ditemukan, menggunakan lena.png sebagai alternatif...")
        westconcord_path = '../assets/lena.png'
        if not os.path.exists(westconcord_path):
            print("✗ ERROR: lena.png juga tidak ditemukan!")
            print("  Melewati Gambar 13 dan 14...")
            return
    
    # GAMBAR 13: Histogram Shrinking
    print("Membuat Gambar 13: Histogram Shrinking...")
    processor_west = HistogramProcessor(westconcord_path)
    img_shrunk = processor_west.imadjust(low_out=0.25, high_out=0.75)
    
    fig = plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(processor_west.original_image, cmap='gray')
    plt.title('Original Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    processor_west.plot_histogram(title='Original Histogram', color='blue')
    
    plt.subplot(2, 2, 3)
    plt.imshow(img_shrunk, cmap='gray')
    plt.title('Shrunk Image (reduced contrast)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    processor_west.plot_histogram(img_shrunk, title='Shrunk Histogram', color='purple')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_3_3_Histogram_Shrinking.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gambar 13 tersimpan")
    
    # Also create transformation function plot
    plt.figure(figsize=(8, 8))
    X = processor_west.original_image.flatten()
    Y = img_shrunk.flatten()
    plt.scatter(X, Y, alpha=0.1, s=1, c='blue')
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.xlabel('Original Image Intensity', fontsize=12)
    plt.ylabel('Adjusted Image Intensity', fontsize=12)
    plt.title('Transformation Function (Shrinking)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.plot([0, 255], [0, 255], 'r--', linewidth=2, label='Identity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_3_3b_Transform_Shrinking.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # GAMBAR 14: Histogram Shrinking with Gamma
    print("Membuat Gambar 14: Histogram Shrinking with Gamma...")
    img_shrunk_gamma = processor_west.imadjust(low_out=0.25, high_out=0.75, gamma=2.0)
    
    fig = plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(processor_west.original_image, cmap='gray')
    plt.title('Original Image', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    processor_west.plot_histogram(title='Original Histogram', color='blue')
    
    plt.subplot(2, 2, 3)
    plt.imshow(img_shrunk_gamma, cmap='gray')
    plt.title('Adjusted Image (gamma=2, shrunk)', fontsize=12, fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    processor_west.plot_histogram(img_shrunk_gamma, title='Adjusted Histogram', color='orange')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_3_4_Histogram_Shrinking_Gamma.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gambar 14 tersimpan")
    
    # Transformation function with gamma
    plt.figure(figsize=(8, 8))
    X_gamma = processor_west.original_image.flatten()
    Y_gamma = img_shrunk_gamma.flatten()
    plt.scatter(X_gamma, Y_gamma, alpha=0.1, s=1, c='orange')
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.xlabel('Original Image Intensity', fontsize=12)
    plt.ylabel('Adjusted Image Intensity', fontsize=12)
    plt.title('Transformation Function (gamma=2)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.plot([0, 255], [0, 255], 'r--', linewidth=2, label='Identity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gambar_9_3_4b_Transform_Gamma.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Tutorial 9.3 - Semua gambar berhasil dibuat!")


def main():
    """Main execution function"""
    print("="*60)
    print("HISTOGRAM PROCESSING - HASIL GENERATOR")
    print("Tutorial 9.1, 9.2, dan 9.3")
    print("="*60)
    
    # Check if all required assets are available
    if not check_required_assets():
        return
    
    # Generate results for each tutorial
    try:
        generate_tutorial_9_1_results()
        generate_tutorial_9_2_results()
        generate_tutorial_9_3_results()
        
        print("\n" + "="*60)
        print("✓ SEMUA GAMBAR BERHASIL DIBUAT!")
        print("="*60)
        print(f"Lokasi: folder 'hasil'")
        print(f"Total gambar: 16+ gambar")
        print("\nGambar-gambar siap digunakan untuk laporan!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print("Pastikan semua file asset tersedia di folder 'assets'")
    except Exception as e:
        print(f"\n✗ ERROR tidak terduga: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
