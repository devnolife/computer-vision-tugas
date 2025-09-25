#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7.4.5 Flipping Operations Demo
Implementasi fungsi flipping matrix/image seperti di IPT (Image Processing Toolbox)
- flipud: flip up to down (vertical flip)
- fliplr: flip left to right (horizontal flip)
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

class FlippingOperations:
    """
    Implementasi operasi flipping untuk matrix dan gambar
    Mirip dengan fungsi IPT (Image Processing Toolbox)
    """
    
    @staticmethod
    def flipud(matrix):
        """
        Flip matrix up to down (vertical flip)
        Equivalent to IPT flipud function
        
        Args:
            matrix: input matrix/image (numpy array)
        
        Returns:
            vertically flipped matrix
        """
        return np.flipud(matrix)
    
    @staticmethod
    def fliplr(matrix):
        """
        Flip matrix left to right (horizontal flip)  
        Equivalent to IPT fliplr function
        
        Args:
            matrix: input matrix/image (numpy array)
            
        Returns:
            horizontally flipped matrix
        """
        return np.fliplr(matrix)
    
    @staticmethod
    def flip_both(matrix):
        """
        Flip matrix both vertically and horizontally (180° rotation)
        
        Args:
            matrix: input matrix/image (numpy array)
            
        Returns:
            matrix flipped in both directions
        """
        return np.flipud(np.fliplr(matrix))
    
    @staticmethod
    def flip_diagonal(matrix):
        """
        Flip matrix along main diagonal (transpose)
        
        Args:
            matrix: input matrix (numpy array)
            
        Returns:
            transposed matrix
        """
        return np.transpose(matrix)
    
    @staticmethod
    def flip_anti_diagonal(matrix):
        """
        Flip matrix along anti-diagonal
        
        Args:
            matrix: input matrix (numpy array)
            
        Returns:
            matrix flipped along anti-diagonal
        """
        # Flip along anti-diagonal = transpose + flip both directions
        return np.flipud(np.fliplr(np.transpose(matrix)))

def demo_matrix_flipping():
    """
    Demo flipping operations pada matrix sederhana
    """
    print("=== DEMO MATRIX FLIPPING ===\n")
    
    # Buat matrix sederhana untuk demonstrasi
    matrix = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8], 
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
    
    print("Original Matrix:")
    print(matrix)
    print()
    
    # FlipUD - flip up to down
    flipped_ud = FlippingOperations.flipud(matrix)
    print("FlipUD (Vertical Flip):")
    print(flipped_ud)
    print()
    
    # FlipLR - flip left to right  
    flipped_lr = FlippingOperations.fliplr(matrix)
    print("FlipLR (Horizontal Flip):")
    print(flipped_lr)
    print()
    
    # Flip both directions
    flipped_both = FlippingOperations.flip_both(matrix)
    print("Flip Both (180° rotation):")
    print(flipped_both)
    print()
    
    # Diagonal flip
    flipped_diag = FlippingOperations.flip_diagonal(matrix)
    print("Flip Diagonal (Transpose):")
    print(flipped_diag)
    print()

def demo_image_flipping():
    """
    Demo flipping operations pada gambar
    """
    print("=== DEMO IMAGE FLIPPING ===\n")
    
    # Load gambar atau buat gambar test
    try:
        # Coba load dari assets
        image = cv2.imread('assets/lindsay.tif', cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError("lindsay.tif tidak ditemukan")
        print("Using lindsay.tif from assets")
    except:
        try:
            # Fallback ke lena.png
            image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError("lena.png tidak ditemukan")
            print("Using lena.png")
        except:
            # Buat gambar test dengan pola yang jelas
            image = create_test_image()
            print("Using synthetic test image")
    
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    
    # Apply different flipping operations
    flipud_img = FlippingOperations.flipud(image)
    fliplr_img = FlippingOperations.fliplr(image) 
    flipboth_img = FlippingOperations.flip_both(image)
    
    # Display results
    plt.figure(figsize=(16, 12))
    
    # Original image
    plt.subplot(3, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    
    # FlipUD
    plt.subplot(3, 4, 2)
    plt.imshow(flipud_img, cmap='gray')
    plt.title('FlipUD\n(Vertical Flip)', fontsize=12)
    plt.axis('off')
    
    # FlipLR
    plt.subplot(3, 4, 3)
    plt.imshow(fliplr_img, cmap='gray')
    plt.title('FlipLR\n(Horizontal Flip)', fontsize=12)
    plt.axis('off')
    
    # Flip both
    plt.subplot(3, 4, 4)
    plt.imshow(flipboth_img, cmap='gray')
    plt.title('Flip Both\n(180° Rotation)', fontsize=12)
    plt.axis('off')
    
    # Practical applications
    
    # Mirror effect (common in photography)
    mirror_left = np.hstack([image, fliplr_img])
    plt.subplot(3, 4, 5)
    plt.imshow(mirror_left, cmap='gray')
    plt.title('Mirror Effect\n(Left-Right)', fontsize=12)
    plt.axis('off')
    
    # Kaleidoscope effect
    kaleidoscope = np.vstack([
        np.hstack([image, fliplr_img]),
        np.hstack([flipud_img, flipboth_img])
    ])
    plt.subplot(3, 4, 6)
    plt.imshow(kaleidoscope, cmap='gray')
    plt.title('Kaleidoscope\n(4-way mirror)', fontsize=12)
    plt.axis('off')
    
    # Document correction simulation
    # Simulate upside down document
    upside_down = flipboth_img
    corrected = FlippingOperations.flip_both(upside_down)
    
    plt.subplot(3, 4, 7)
    plt.imshow(upside_down, cmap='gray')
    plt.title('Upside Down\n(Document Error)', fontsize=12)
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(corrected, cmap='gray')
    plt.title('Auto Corrected\n(Fixed Orientation)', fontsize=12)
    plt.axis('off')
    
    # Face recognition data augmentation
    augmented_faces = [image, fliplr_img]  # Original + horizontal flip
    
    plt.subplot(3, 4, 9)
    plt.imshow(image, cmap='gray')
    plt.title('Original Face\n(Training Data)', fontsize=12)
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(fliplr_img, cmap='gray')
    plt.title('Augmented Face\n(Mirrored)', fontsize=12)
    plt.axis('off')
    
    # Compare with OpenCV flip functions
    cv_flipud = cv2.flip(image, 0)  # Vertical flip
    cv_fliplr = cv2.flip(image, 1)  # Horizontal flip
    
    plt.subplot(3, 4, 11)
    plt.imshow(cv_flipud, cmap='gray')
    plt.title('OpenCV FlipUD\n(cv2.flip(img,0))', fontsize=12)
    plt.axis('off')
    
    plt.subplot(3, 4, 12)
    plt.imshow(cv_fliplr, cmap='gray')
    plt.title('OpenCV FlipLR\n(cv2.flip(img,1))', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Image Flipping Operations - IPT Style Functions', fontsize=16, y=0.98)
    plt.show()
    
    # Verify equivalence with OpenCV
    print("\n=== VERIFICATION ===")
    print(f"NumPy flipud == OpenCV flip(0): {np.array_equal(flipud_img, cv_flipud)}")
    print(f"NumPy fliplr == OpenCV flip(1): {np.array_equal(fliplr_img, cv_fliplr)}")

def create_test_image():
    """
    Buat gambar test dengan pola yang jelas untuk demonstrasi flipping
    """
    img = np.zeros((256, 256), dtype=np.uint8)
    
    # Background gradient
    for i in range(256):
        for j in range(256):
            img[i, j] = int((i + j) * 255 / 512)
    
    # Add asymmetric elements to show flipping clearly
    
    # Rectangle di kiri atas
    cv2.rectangle(img, (20, 20), (80, 80), 255, -1)
    
    # Circle di kanan bawah  
    cv2.circle(img, (200, 200), 30, 0, -1)
    
    # Triangle di kiri bawah
    triangle_pts = np.array([[20, 200], [80, 200], [50, 240]], dtype=np.int32)
    cv2.fillPoly(img, [triangle_pts], 128)
    
    # Text di kanan atas (akan terlihat jelas saat di-flip)
    cv2.putText(img, 'FLIP', (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 64, 2)
    
    # Arrow pointing right
    arrow_pts = np.array([[100, 120], [140, 120], [130, 110], [160, 128], 
                         [130, 146], [140, 136], [100, 136]], dtype=np.int32)
    cv2.fillPoly(img, [arrow_pts], 192)
    
    return img

def demo_practical_applications():
    """
    Demo aplikasi praktis dari flipping operations
    """
    print("=== PRACTICAL APPLICATIONS ===\n")
    
    # Load test image
    try:
        image = cv2.imread('assets/lindsay.tif', cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = create_test_image()
    except:
        image = create_test_image()
    
    plt.figure(figsize=(18, 10))
    
    # 1. Photo Booth Effects
    plt.subplot(2, 5, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Photo', fontsize=10)
    plt.axis('off')
    
    # 2. Social Media Filters
    mirror_selfie = np.hstack([FlippingOperations.fliplr(image), image])
    plt.subplot(2, 5, 2)
    plt.imshow(mirror_selfie, cmap='gray')
    plt.title('Mirror Selfie\n(Social Media)', fontsize=10)
    plt.axis('off')
    
    # 3. Document Scanning Auto-correct
    rotated_doc = FlippingOperations.flip_both(image)
    plt.subplot(2, 5, 3)
    plt.imshow(rotated_doc, cmap='gray')
    plt.title('Upside Down\nDocument', fontsize=10)
    plt.axis('off')
    
    # 4. Stereo Vision Simulation
    left_eye = image
    right_eye = FlippingOperations.fliplr(image)
    stereo = np.hstack([left_eye, right_eye])
    plt.subplot(2, 5, 4)
    plt.imshow(stereo, cmap='gray')
    plt.title('Stereo Pair\n(3D Vision)', fontsize=10)
    plt.axis('off')
    
    # 5. Pattern Generation
    pattern = np.vstack([
        np.hstack([image, FlippingOperations.fliplr(image)]),
        np.hstack([FlippingOperations.flipud(image), 
                  FlippingOperations.flip_both(image)])
    ])
    plt.subplot(2, 5, 5)
    plt.imshow(pattern, cmap='gray')
    plt.title('Symmetric Pattern\n(Textile Design)', fontsize=10)
    plt.axis('off')
    
    # 6. Medical Imaging
    plt.subplot(2, 5, 6)
    plt.imshow(FlippingOperations.fliplr(image), cmap='gray')
    plt.title('Medical Scan\n(Mirror View)', fontsize=10)
    plt.axis('off')
    
    # 7. Game Development - Sprite Mirroring
    plt.subplot(2, 5, 7)
    plt.imshow(FlippingOperations.fliplr(image), cmap='gray')
    plt.title('Game Sprite\n(Facing Left)', fontsize=10)
    plt.axis('off')
    
    # 8. Quality Control - Symmetry Check
    original_half = image[:, :image.shape[1]//2]
    mirrored_half = FlippingOperations.fliplr(image[:, image.shape[1]//2:])
    symmetry_check = np.hstack([original_half, mirrored_half])
    
    plt.subplot(2, 5, 8)
    plt.imshow(symmetry_check, cmap='gray')
    plt.title('Symmetry Check\n(Quality Control)', fontsize=10)
    plt.axis('off')
    
    # 9. Art Generation
    art_piece = np.vstack([
        image,
        FlippingOperations.flipud(image)
    ])
    plt.subplot(2, 5, 9)
    plt.imshow(art_piece, cmap='gray')
    plt.title('Digital Art\n(Reflection)', fontsize=10)
    plt.axis('off')
    
    # 10. Data Augmentation for AI
    augmentation_set = [
        image,
        FlippingOperations.flipud(image),
        FlippingOperations.fliplr(image),
        FlippingOperations.flip_both(image)
    ]
    
    # Create a 2x2 grid of augmented images
    aug_grid = np.vstack([
        np.hstack([augmentation_set[0], augmentation_set[1]]),
        np.hstack([augmentation_set[2], augmentation_set[3]])
    ])
    
    plt.subplot(2, 5, 10)
    plt.imshow(aug_grid, cmap='gray')
    plt.title('AI Training Set\n(4x Augmentation)', fontsize=10)
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Practical Applications of Image Flipping', fontsize=16, y=0.98)
    plt.show()

def performance_comparison():
    """
    Bandingkan performa berbagai metode flipping
    """
    print("=== PERFORMANCE COMPARISON ===\n")
    
    # Buat gambar test besar
    large_image = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)
    
    import time
    
    # Test NumPy flipud
    start_time = time.time()
    for i in range(100):
        result1 = np.flipud(large_image)
    numpy_flipud_time = time.time() - start_time
    
    # Test NumPy fliplr
    start_time = time.time()
    for i in range(100):
        result2 = np.fliplr(large_image)
    numpy_fliplr_time = time.time() - start_time
    
    # Test OpenCV flip
    start_time = time.time()
    for i in range(100):
        result3 = cv2.flip(large_image, 0)  # Vertical
        result4 = cv2.flip(large_image, 1)  # Horizontal
    opencv_flip_time = time.time() - start_time
    
    print(f"Image size: {large_image.shape}")
    print(f"Iterations: 100")
    print(f"NumPy flipud time: {numpy_flipud_time:.4f} seconds")
    print(f"NumPy fliplr time: {numpy_fliplr_time:.4f} seconds") 
    print(f"OpenCV flip time: {opencv_flip_time:.4f} seconds")
    print(f"Memory usage per image: {large_image.nbytes/1024/1024:.2f} MB")

if __name__ == "__main__":
    print("🔄 IMAGE FLIPPING OPERATIONS DEMO")
    print("Implementasi fungsi flipping seperti IPT (Image Processing Toolbox)")
    print("=" * 60)
    
    print("\n1️⃣  Matrix Flipping Demo...")
    demo_matrix_flipping()
    
    print("\n2️⃣  Image Flipping Demo...")
    demo_image_flipping()
    
    print("\n3️⃣  Practical Applications Demo...")
    demo_practical_applications()
    
    print("\n4️⃣  Performance Comparison...")
    performance_comparison()
    
    print("\n✅ Demo selesai!")
    print("\n📝 Fungsi yang tersedia:")
    print("   - FlippingOperations.flipud(matrix)  # Flip up-down")
    print("   - FlippingOperations.fliplr(matrix)  # Flip left-right") 
    print("   - FlippingOperations.flip_both(matrix)  # Flip both directions")
    print("   - FlippingOperations.flip_diagonal(matrix)  # Transpose")