import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, img_as_ubyte, img_as_float
from scipy import ndimage
import os

class HistogramProcessor:
    def __init__(self, image_path=None, image=None):
        """Initialize with either image path or image array"""
        if image_path is not None:
            self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
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
        plt.title(title)
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
        # Compute CDF of source image
        source_hist, _ = self.compute_histogram(self.original_image)
        source_cdf = np.cumsum(source_hist)
        source_cdf_normalized = source_cdf / source_cdf[-1]
        
        # Compute CDF of reference histogram
        reference_cdf = np.cumsum(reference_hist)
        reference_cdf_normalized = reference_cdf / reference_cdf[-1]
        
        # Create lookup table for histogram matching
        lookup_table = np.zeros(256, dtype=np.uint8)
        g_j = 0
        for g_i in range(256):
            while g_j < 255 and reference_cdf_normalized[g_j] < source_cdf_normalized[g_i]:
                g_j += 1
            lookup_table[g_i] = g_j
        
        # Apply lookup table
        matched = cv2.LUT(self.original_image, lookup_table)
        return matched
    
    def adaptive_histogram_equalization(self, clip_limit=2.0, tile_grid_size=(8,8)):
        """Perform adaptive histogram equalization (CLAHE)"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        equalized = clahe.apply(self.original_image)
        return equalized
    
    def histogram_sliding(self, constant, clip=True):
        """Adjust brightness by adding constant to all pixels"""
        if self.original_image.dtype == np.uint8:
            # Convert to float for calculation
            image_float = self.original_image.astype(np.float32) / 255.0
            adjusted = image_float + constant
        else:
            adjusted = self.original_image + constant
        
        if clip:
            adjusted = np.clip(adjusted, 0, 1)
        
        # Convert back to uint8
        return (adjusted * 255).astype(np.uint8)
    
    def imadjust(self, low_in=None, high_in=None, low_out=0, high_out=1, gamma=1.0):
        """
        Adjust image intensity similar to MATLAB's imadjust
        Parameters are in normalized range [0, 1]
        """
        # Convert to float
        image_float = self.original_image.astype(np.float32) / 255.0
        
        # Auto-calculate limits if not provided
        if low_in is None or high_in is None:
            low_in, high_in = self.stretchlim(image_float)
        
        # Clip input values
        image_clipped = np.clip(image_float, low_in, high_in)
        
        # Normalize to [0, 1]
        image_normalized = (image_clipped - low_in) / (high_in - low_in)
        
        # Apply gamma correction
        image_gamma = np.power(image_normalized, gamma)
        
        # Scale to output range
        image_adjusted = image_gamma * (high_out - low_out) + low_out
        
        # Clip to valid range
        image_adjusted = np.clip(image_adjusted, 0, 1)
        
        # Convert back to uint8
        return (image_adjusted * 255).astype(np.uint8)
    
    def stretchlim(self, image_float=None, tol=0.01):
        """
        Find limits for contrast stretching
        Similar to MATLAB's stretchlim
        """
        if image_float is None:
            image_float = self.original_image.astype(np.float32) / 255.0
        
        # Compute histogram
        hist, bins = np.histogram(image_float.flatten(), bins=256, range=(0, 1))
        
        # Compute CDF
        cdf = np.cumsum(hist) / np.sum(hist)
        
        # Find low and high limits based on tolerance
        low_idx = np.where(cdf >= tol)[0][0]
        high_idx = np.where(cdf >= (1 - tol))[0][0]
        
        low_in = bins[low_idx]
        high_in = bins[high_idx]
        
        return low_in, high_in


# ================ TUTORIAL 9.1: IMAGE HISTOGRAMS ================
def tutorial_9_1():
    """Tutorial 9.1: Image Histograms"""
    print("="*60)
    print("TUTORIAL 9.1: IMAGE HISTOGRAMS")
    print("="*60)
    
    # Create sample image if circuit.tif doesn't exist
    image_path = 'circuit.tif'
    if not os.path.exists(image_path):
        print("Creating sample circuit image...")
        circuit = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
        # Add some patterns
        circuit[50:100, 50:250] = 200
        circuit[150:200, 100:200] = 100
        cv2.imwrite(image_path, circuit)
    
    processor = HistogramProcessor(image_path)
    
    # Step 1: Display image and histogram with 256 bins
    print("\nStep 1: Display image and histogram")
    fig = plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(processor.original_image, cmap='gray')
    plt.title('Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    processor.plot_histogram(bins=256, title='Histogram (256 bins)')
    
    # Step 2: Display histograms with different number of bins
    print("Step 2: Display histograms with different bin counts")
    
    plt.subplot(2, 2, 3)
    processor.plot_histogram(bins=64, title='Histogram with 64 bins')
    
    plt.subplot(2, 2, 4)
    processor.plot_histogram(bins=32, title='Histogram with 32 bins')
    
    plt.tight_layout()
    plt.show()
    
    print("\nQuestion 1: Jelaskan perubahan drastis pada nilai sumbu Y ketika histogram")
    print("ditampilkan dengan jumlah bin yang lebih sedikit.")
    
    # Step 3: Get histogram values
    print("\nStep 3: Get histogram bin values")
    hist_32, _ = processor.compute_histogram(bins=32)
    print(f"Histogram with 32 bins shape: {hist_32.shape}")
    print(f"First 5 bin values: {hist_32[:5]}")
    
    # Step 4: Normalize histogram
    print("\nStep 4: Normalize histogram values")
    hist_norm = processor.normalize_histogram(hist_32)
    print(f"First 5 normalized values: {hist_norm[:5]}")
    
    print("\nQuestion 2: Apa yang dilakukan fungsi numel/size?")
    print("Question 3: Verifikasi bahwa jumlah nilai yang dinormalisasi adalah 1")
    print(f"Sum of normalized values: {np.sum(hist_norm)}")
    
    # Step 6: Display histogram using bar chart
    print("\nStep 6-8: Display histogram as bar chart")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(range(len(hist_32)), hist_32, color='red')
    axes[0].set_xlim([0, 32])
    axes[0].set_ylim([0, np.max(hist_32)])
    axes[0].set_xticks(np.arange(0, 33, 8))
    axes[0].set_title('Bar Chart')
    axes[0].set_xlabel('Bin')
    axes[0].set_ylabel('Count')
    
    # Step 9: Normalized bar chart
    axes[1].bar(range(len(hist_norm)), hist_norm, color='green')
    axes[1].set_xlim([0, 32])
    axes[1].set_ylim([0, np.max(hist_norm)])
    axes[1].set_xticks(np.arange(0, 33, 8))
    axes[1].set_title('Normalized Bar Chart')
    axes[1].set_xlabel('Bin')
    axes[1].set_ylabel('Normalized Count')
    
    plt.tight_layout()
    plt.show()
    
    print("\nQuestion 4: Bagaimana cara mengubah lebar bar dalam bar chart?")
    
    # Step 11: Stem charts
    print("\nStep 11: Display stem charts")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    markerline, stemlines, baseline = axes[0].stem(hist_32, linefmt='b-', 
                                                   markerfmt='ro', basefmt='k-')
    plt.setp(markerline, 'markerfacecolor', 'red')
    axes[0].set_title('Stem Chart')
    axes[0].set_xlabel('Bin')
    axes[0].set_ylabel('Count')
    
    markerline, stemlines, baseline = axes[1].stem(hist_norm, linefmt='b-', 
                                                   markerfmt='ro', basefmt='k-')
    plt.setp(markerline, 'markerfacecolor', 'red')
    axes[1].set_title('Normalized Stem Chart')
    axes[1].set_xlabel('Bin')
    axes[1].set_ylabel('Normalized Count')
    
    plt.tight_layout()
    plt.show()
    
    print("\nQuestion 5: Bagaimana membuat garis menjadi putus-putus?")
    print("Question 6: Sesuaikan batas sumbu dan tick marks")
    
    # Step 12: Plot graphs
    print("\nStep 12: Display plot graphs")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(hist_32, 'b-', marker='o', markersize=4)
    axes[0].set_title('Plot Graph')
    axes[0].set_xlabel('Bin')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(hist_norm, 'b-', marker='o', markersize=4)
    axes[1].set_title('Normalized Plot Graph')
    axes[1].set_xlabel('Bin')
    axes[1].set_ylabel('Normalized Count')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nQuestion 7: Bagaimana membuat titik-titik lebih terlihat jelas?")
    print("\nTutorial 9.1 selesai!")


# ================ TUTORIAL 9.2: HISTOGRAM EQUALIZATION ================
def tutorial_9_2():
    """Tutorial 9.2: Histogram Equalization and Specification"""
    print("\n" + "="*60)
    print("TUTORIAL 9.2: HISTOGRAM EQUALIZATION AND SPECIFICATION")
    print("="*60)
    
    # Step 1-3: Histogram equalization on pout.tif
    print("\nStep 1-3: Histogram equalization on pout image")
    
    # Create sample pout image if doesn't exist
    pout_path = 'pout.tif'
    if not os.path.exists(pout_path):
        print("Creating sample pout image...")
        pout = np.random.normal(100, 20, (256, 256)).astype(np.uint8)
        cv2.imwrite(pout_path, pout)
    
    processor_pout = HistogramProcessor(pout_path)
    pout_eq = processor_pout.histogram_equalization()
    
    fig = plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(processor_pout.original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    processor_pout.plot_histogram(title='Original Histogram')
    
    plt.subplot(2, 2, 3)
    plt.imshow(pout_eq, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    processor_pout.plot_histogram(pout_eq, title='Equalized Histogram')
    
    plt.tight_layout()
    plt.show()
    
    print("\nQuestion 1: Mengapa harus menyertakan parameter kedua (256)?")
    print("Question 2: Apa efek histogram equalization pada gambar dengan kontras rendah?")
    
    # Step 5: Histogram equalization on tire image
    print("\nStep 5: Histogram equalization on tire image")
    
    tire_path = 'tire.tif'
    if not os.path.exists(tire_path):
        tire = np.random.exponential(50, (256, 256)).astype(np.uint8)
        cv2.imwrite(tire_path, tire)
    
    processor_tire = HistogramProcessor(tire_path)
    tire_eq = processor_tire.histogram_equalization()
    
    fig = plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(processor_tire.original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    processor_tire.plot_histogram(title='Original Histogram')
    
    plt.subplot(2, 2, 3)
    plt.imshow(tire_eq, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    processor_tire.plot_histogram(tire_eq, title='Equalized Histogram')
    
    plt.tight_layout()
    plt.show()
    
    print("\nQuestion 3: Apa yang dapat dikatakan tentang brightness keseluruhan?")
    print("Question 4: Bagaimana histogram equalization mempengaruhi brightness?")
    
    # Step 7: Histogram equalization on eight image
    print("\nStep 7: Histogram equalization on eight image")
    
    eight_path = 'eight.tif'
    if not os.path.exists(eight_path):
        eight = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(eight, (100, 80), 40, 200, -1)
        cv2.circle(eight, (156, 176), 40, 200, -1)
        cv2.imwrite(eight_path, eight)
    
    processor_eight = HistogramProcessor(eight_path)
    eight_eq = processor_eight.histogram_equalization()
    
    fig = plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(processor_eight.original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    processor_eight.plot_histogram(title='Original Histogram')
    
    plt.subplot(2, 2, 3)
    plt.imshow(eight_eq, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    processor_eight.plot_histogram(eight_eq, title='Equalized Histogram')
    
    plt.tight_layout()
    plt.show()
    
    print("\nQuestion 5: Mengapa terjadi penurunan kualitas gambar yang signifikan?")
    
    # Step 8-9: Display transformation function (CDF)
    print("\nStep 8-9: Display transformation function (CDF)")
    
    hist_eight, _ = processor_eight.compute_histogram()
    cdf = np.cumsum(hist_eight)
    cdf_normalized = cdf / cdf[-1]
    
    plt.figure(figsize=(10, 5))
    plt.plot(cdf_normalized)
    plt.title('Normalized CDF (Transformation Function)')
    plt.xlabel('Input Intensity')
    plt.ylabel('Output Intensity')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nQuestion 6: Apa yang dilakukan fungsi cumsum?")
    
    # Step 11-13: Histogram specification/matching
    print("\nStep 11-13: Histogram specification/matching")
    
    # Create desired histogram shapes
    uniform_hist = np.ones(256) * 0.5
    linear_hist = np.linspace(0, 1, 256)
    
    # Perform histogram matching for linear histogram
    matched_image = processor_pout.histogram_matching(linear_hist * 1000)
    
    fig = plt.figure(figsize=(15, 10))
    
    # Original
    plt.subplot(3, 3, 1)
    plt.imshow(processor_pout.original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    processor_pout.plot_histogram(title='Original Histogram')
    
    # Equalized
    plt.subplot(3, 3, 4)
    plt.imshow(pout_eq, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    processor_pout.plot_histogram(pout_eq, title='Equalized Histogram')
    
    plt.subplot(3, 3, 6)
    plt.plot(uniform_hist)
    plt.title('Desired Histogram Shape (Uniform)')
    plt.ylim([0, 1])
    plt.xlim([0, 256])
    
    # Matched
    plt.subplot(3, 3, 7)
    plt.imshow(matched_image, cmap='gray')
    plt.title('Matched Image')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    processor_pout.plot_histogram(matched_image, title='Matched Histogram')
    
    plt.subplot(3, 3, 9)
    plt.plot(linear_hist)
    plt.title('Desired Histogram Shape (Linear)')
    plt.ylim([0, 1])
    plt.xlim([0, 256])
    
    plt.tight_layout()
    plt.show()
    
    print("\nQuestion 7-9: Interactive Histogram Matching demo questions")
    
    # Step 17: Local (adaptive) histogram equalization
    print("\nStep 17: Adaptive histogram equalization (CLAHE)")
    
    coins_path = 'coins.png'
    if not os.path.exists(coins_path):
        coins = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
        for i in range(5):
            x, y = np.random.randint(50, 250, 2)
            cv2.circle(coins, (x, y), 30, 180, -1)
        cv2.imwrite(coins_path, coins)
    
    processor_coins = HistogramProcessor(coins_path)
    coins_eq = processor_coins.histogram_equalization()
    coins_adaptive = processor_coins.adaptive_histogram_equalization(clip_limit=0.1)
    
    fig = plt.figure(figsize=(12, 15))
    
    plt.subplot(3, 2, 1)
    plt.imshow(processor_coins.original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 2, 2)
    processor_coins.plot_histogram(title='Original Histogram')
    
    plt.subplot(3, 2, 3)
    plt.imshow(coins_eq, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    processor_coins.plot_histogram(coins_eq, title='Equalized Histogram')
    
    plt.subplot(3, 2, 5)
    plt.imshow(coins_adaptive, cmap='gray')
    plt.title('Adaptive Histogram Equalization')
    plt.axis('off')
    
    plt.subplot(3, 2, 6)
    processor_coins.plot_histogram(coins_adaptive, title='Adaptive Hist Eq Histogram')
    
    plt.tight_layout()
    plt.show()
    
    print("\nQuestion 10: Apa fungsi parameter ClipLimit?")
    print("Question 11: Berapa ukuran tile default untuk adapthisteq?")
    print("\nTutorial 9.2 selesai!")


# ================ TUTORIAL 9.3: OTHER HISTOGRAM MODIFICATIONS ================
def tutorial_9_3():
    """Tutorial 9.3: Other Histogram Modification Techniques"""
    print("\n" + "="*60)
    print("TUTORIAL 9.3: OTHER HISTOGRAM MODIFICATION TECHNIQUES")
    print("="*60)
    
    # Step 1-3: Histogram sliding (brightness adjustment)
    print("\nStep 1-3: Histogram sliding")
    
    pout_path = 'pout.tif'
    if not os.path.exists(pout_path):
        pout = np.random.normal(100, 20, (256, 256)).astype(np.uint8)
        cv2.imwrite(pout_path, pout)
    
    processor = HistogramProcessor(pout_path)
    
    # Add 0.1 to image
    img_bright1 = processor.histogram_sliding(0.1)
    
    # Add 0.5 to image
    img_bright2 = processor.histogram_sliding(0.5)
    
    fig = plt.figure(figsize=(12, 12))
    
    plt.subplot(3, 2, 1)
    plt.imshow(processor.original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 2, 2)
    processor.plot_histogram(title='Original Histogram')
    
    plt.subplot(3, 2, 3)
    plt.imshow(img_bright1, cmap='gray')
    plt.title('Original Image + 0.1')
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    processor.plot_histogram(img_bright1, title='Original Hist + 0.1')
    
    plt.subplot(3, 2, 5)
    plt.imshow(img_bright2, cmap='gray')
    plt.title('Original Image + 0.5')
    plt.axis('off')
    
    plt.subplot(3, 2, 6)
    processor.plot_histogram(img_bright2, title='Original Hist + 0.5')
    
    plt.tight_layout()
    plt.show()
    
    print("\nQuestion 1: Bagaimana histogram berubah setelah penyesuaian?")
    print("Question 2: Apa yang dikandung variabel bad_values?")
    print("Question 3: Mengapa plot ketiga menunjukkan jumlah pixel dengan nilai 1 yang berlebihan?")
    
    # Step 5: Histogram stretching
    print("\nStep 5-6: Histogram stretching")
    
    img_stretched = processor.imadjust()
    img_stretched2 = processor.imadjust()  # Using default parameters
    
    fig = plt.figure(figsize=(12, 12))
    
    plt.subplot(3, 2, 1)
    plt.imshow(processor.original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 2, 2)
    processor.plot_histogram(title='Original Histogram')
    
    plt.subplot(3, 2, 3)
    plt.imshow(img_stretched, cmap='gray')
    plt.title('Stretched Image')
    plt.axis('off')
    
    plt.subplot(3, 2, 4)
    processor.plot_histogram(img_stretched, title='Stretched Histogram')
    
    plt.subplot(3, 2, 5)
    plt.imshow(img_stretched2, cmap='gray')
    plt.title('Stretched Image (default params)')
    plt.axis('off')
    
    plt.subplot(3, 2, 6)
    processor.plot_histogram(img_stretched2, title='Stretched Histogram')
    
    plt.tight_layout()
    plt.show()
    
    # Difference image
    diff = cv2.absdiff(img_stretched, img_stretched2)
    plt.figure(figsize=(8, 6))
    plt.imshow(diff, cmap='gray')
    plt.title('Difference Image')
    plt.colorbar()
    plt.show()
    
    print(f"Min difference: {np.min(diff)}")
    print(f"Max difference: {np.max(diff)}")
    
    print("\nQuestion 4: Bagaimana histogram berubah setelah penyesuaian?")
    print("Question 5: Apa tujuan menggunakan fungsi stretchlim?")
    print("Question 6: Bagaimana tampilan difference image?")
    print("Question 7: Apa tujuan memeriksa nilai maksimum dan minimum?")
    
    # Step 8: Histogram shrinking
    print("\nStep 8-9: Histogram shrinking")
    
    # Create or load westconcord image
    westconcord_path = 'westconcord.png'
    if not os.path.exists(westconcord_path):
        westconcord = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
        cv2.imwrite(westconcord_path, westconcord)
    
    processor_west = HistogramProcessor(westconcord_path)
    img_shrunk = processor_west.imadjust(low_out=0.25, high_out=0.75)
    
    fig = plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(processor_west.original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    processor_west.plot_histogram(title='Original Histogram')
    
    plt.subplot(2, 2, 3)
    plt.imshow(img_shrunk_gamma, cmap='gray')
    plt.title('Adjusted Image (gamma=2)')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    processor_west.plot_histogram(img_shrunk_gamma, title='Adjusted Histogram')
    
    plt.tight_layout()
    plt.show()
    
    # Display transformation function with gamma
    X_gamma = processor_west.original_image.flatten()
    Y_gamma = img_shrunk_gamma.flatten()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(X_gamma, Y_gamma, alpha=0.1, s=1)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.xlabel('Original Image')
    plt.ylabel('Adjusted Image')
    plt.title('Transformation Function (gamma=2)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nQuestion 10: Mengapa plot transformation function menampilkan gap dari 0 hingga 12?")
    print("\nTutorial 9.3 selesai!")
    print("\n" + "="*60)
    print("SEMUA TUTORIAL SELESAI!")
    print("="*60)


# Main execution
if __name__ == "__main__":
    print("HISTOGRAM PROCESSING TUTORIALS")
    print("="*60)
    print("Tutorial 9.1: Image Histograms")
    print("Tutorial 9.2: Histogram Equalization and Specification")
    print("Tutorial 9.3: Other Histogram Modification Techniques")
    print("="*60)
    
    # Run all tutorials
    tutorial_9_1()
    tutorial_9_2()
    tutorial_9_3()
    
    print("\n\nPetunjuk penggunaan:")
    print("1. Pastikan semua library sudah terinstall:")
    print("   pip install opencv-python numpy matplotlib scikit-image scipy")
    print("2. Jalankan setiap tutorial secara terpisah jika diperlukan:")
    print("   tutorial_9_1()")
    print("   tutorial_9_2()")
    print("   tutorial_9_3()")
    print("3. Simpan semua gambar hasil untuk laporan")subplot(2, 2, 2)
    processor_west.plot_histogram(title='Original Histogram')
    
    plt.subplot(2, 2, 3)
    plt.imshow(img_shrunk, cmap='gray')
    plt.title('Shrunk Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    processor_west.plot_histogram(img_shrunk, title='Shrunk Histogram')
    
    plt.tight_layout()
    plt.show()
    
    # Display transformation function
    X = processor_west.original_image.flatten()
    Y = img_shrunk.flatten()
    
    plt.figure(figsize=(8, 8))
    plt.scatter(X, Y, alpha=0.1, s=1)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.xlabel('Original Image')
    plt.ylabel('Adjusted Image')
    plt.title('Transformation Function')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nQuestion 8: Apa yang dilakukan dua pernyataan pertama dalam kode?")
    print("Question 9: Apa fungsi xlabel dan ylabel?")
    
    # Step 11: Histogram shrinking with gamma
    print("\nStep 11: Histogram shrinking with gamma = 2")
    
    img_shrunk_gamma = processor_west.imadjust(low_out=0.25, high_out=0.75, gamma=2.0)
    
    fig = plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(processor_west.original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    processor_west.plot_histogram(title='Original Histogram')

    plt.subplot(2, 2, 3)
    plt.imshow(img_shrunk_gamma, cmap='gray')
    plt.title('Adjusted Image (gamma=2)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    processor_west.plot_histogram(img_shrunk_gamma, title='Adjusted Histogram')

    plt.tight_layout()
    plt.show()

    # Display transformation function with gamma
    X_gamma = processor_west.original_image.flatten()
    Y_gamma = img_shrunk_gamma.flatten()

    plt.figure(figsize=(8, 8))
    plt.scatter(X_gamma, Y_gamma, alpha=0.1, s=1)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.xlabel('Original Image')
    plt.ylabel('Adjusted Image')
    plt.title('Transformation Function (gamma=2)')
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\nQuestion 10: Mengapa plot transformation function menampilkan gap dari 0 hingga 12?")
    print("\nTutorial 9.3 selesai!")
    print("\n" + "="*60)
    print("SEMUA TUTORIAL SELESAI!")
    print("="*60)


# Main execution
if __name__ == "__main__":
    print("HISTOGRAM PROCESSING TUTORIALS")
    print("="*60)
    print("Tutorial 9.1: Image Histograms")
    print("Tutorial 9.2: Histogram Equalization and Specification")
    print("Tutorial 9.3: Other Histogram Modification Techniques")
    print("="*60)

    # Run all tutorials
    tutorial_9_1()
    tutorial_9_2()
    tutorial_9_3()

    print("\n\nPetunjuk penggunaan:")
    print("1. Pastikan semua library sudah terinstall:")
    print("   pip install opencv-python numpy matplotlib scikit-image scipy")
    print("2. Jalankan setiap tutorial secara terpisah jika diperlukan:")
    print("   tutorial_9_1()")
    print("   tutorial_9_2()")
    print("   tutorial_9_3()")
    print("3. Simpan semua gambar hasil untuk laporan")
