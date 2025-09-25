import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform
import cv2

class AffineTransformations:
    """
    Implementasi berbagai jenis Affine Transformations
    """
    
    @staticmethod
    def translation_matrix(dx, dy):
        """
        Membuat matriks transformasi untuk translation (perpindahan)
        
        Args:
            dx: perpindahan dalam arah x
            dy: perpindahan dalam arah y
        
        Returns:
            3x3 transformation matrix
        """
        return np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    @staticmethod
    def scaling_matrix(sx, sy):
        """
        Membuat matriks transformasi untuk scaling (penskalaan)
        
        Args:
            sx: faktor skala dalam arah x
            sy: faktor skala dalam arah y
        
        Returns:
            3x3 transformation matrix
        """
        return np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ], dtype=np.float32)
    
    @staticmethod
    def rotation_matrix(angle_degrees):
        """
        Membuat matriks transformasi untuk rotation (rotasi)
        
        Args:
            angle_degrees: sudut rotasi dalam derajat (counterclockwise)
        
        Returns:
            3x3 transformation matrix
        """
        angle_rad = np.radians(angle_degrees)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        return np.array([
            [cos_a, sin_a, 0],
            [-sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
    
    @staticmethod
    def shear_matrix(shx, shy):
        """
        Membuat matriks transformasi untuk shearing (pencondogan)
        
        Args:
            shx: faktor shear dalam arah x
            shy: faktor shear dalam arah y
        
        Returns:
            3x3 transformation matrix
        """
        return np.array([
            [1, shx, 0],
            [shy, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
    
    @staticmethod
    def apply_transform(image, transform_matrix):
        """
        Menerapkan transformasi affine pada gambar
        
        Args:
            image: input image (numpy array)
            transform_matrix: 3x3 transformation matrix
        
        Returns:
            transformed image
        """
        # Menggunakan OpenCV untuk transformasi
        # OpenCV membutuhkan matriks 2x3, jadi kita ambil 2 baris pertama
        transform_2x3 = transform_matrix[:2, :]
        
        rows, cols = image.shape[:2]
        transformed = cv2.warpAffine(image, transform_2x3, (cols, rows))
        
        return transformed
    
    @staticmethod
    def combine_transforms(*matrices):
        """
        Menggabungkan beberapa transformasi dengan perkalian matriks
        
        Args:
            *matrices: beberapa transformation matrices
        
        Returns:
            combined transformation matrix
        """
        result = np.eye(3)
        for matrix in matrices:
            result = np.dot(result, matrix)
        return result

def demo_transformations():
    """
    Demo penggunaan berbagai transformasi affine menggunakan gambar lena.png
    """
    # Load Lena image
    try:
        # Coba load lena.png dari direktori saat ini
        image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError("lena.png tidak ditemukan")
    except:
        try:
            # Fallback ke scikit-image jika ada
            from skimage import data
            image = data.camera()
            print("Menggunakan gambar alternatif karena lena.png tidak ditemukan")
        except:
            # Buat gambar sederhana sebagai fallback terakhir
            image = np.zeros((512, 512), dtype=np.uint8)
            cv2.rectangle(image, (100, 100), (400, 400), 255, -1)
            cv2.circle(image, (256, 256), 80, 0, -1)
            cv2.putText(image, 'LENA', (200, 270), cv2.FONT_HERSHEY_SIMPLEX, 2, 128, 3)
            print("Menggunakan gambar dummy karena lena.png tidak ditemukan")
    
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    
    # Pastikan gambar adalah grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    plt.figure(figsize=(16, 12))
    
    # Original Lena image
    plt.subplot(3, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Lena Image')
    plt.axis('off')
    
    # 1. Translation - Geser Lena
    trans_matrix = AffineTransformations.translation_matrix(80, 50)
    translated = AffineTransformations.apply_transform(image, trans_matrix)
    
    plt.subplot(3, 4, 2)
    plt.imshow(translated, cmap='gray')
    plt.title('Translation\n(Move right 80px, down 50px)')
    plt.axis('off')
    
    # 2. Scaling - Perbesar/kecil Lena
    scale_matrix = AffineTransformations.scaling_matrix(1.3, 0.7)
    scaled = AffineTransformations.apply_transform(image, scale_matrix)
    
    plt.subplot(3, 4, 3)
    plt.imshow(scaled, cmap='gray')
    plt.title('Scaling\n(Width x1.3, Height x0.7)')
    plt.axis('off')
    
    # 3. Rotation - Putar Lena
    rot_matrix = AffineTransformations.rotation_matrix(25)
    rotated = AffineTransformations.apply_transform(image, rot_matrix)
    
    plt.subplot(3, 4, 4)
    plt.imshow(rotated, cmap='gray')
    plt.title('Rotation\n(25 degrees counterclockwise)')
    plt.axis('off')
    
    # 4. Shearing - Condongkan Lena
    shear_matrix = AffineTransformations.shear_matrix(0.4, 0.2)
    sheared = AffineTransformations.apply_transform(image, shear_matrix)
    
    plt.subplot(3, 4, 5)
    plt.imshow(sheared, cmap='gray')
    plt.title('Shearing\n(shx=0.4, shy=0.2)')
    plt.axis('off')
    
    # 5. Rotation 90 degrees - Lena tegak
    rot90_matrix = AffineTransformations.rotation_matrix(90)
    rotated90 = AffineTransformations.apply_transform(image, rot90_matrix)
    
    plt.subplot(3, 4, 6)
    plt.imshow(rotated90, cmap='gray')
    plt.title('Rotation 90°\n(Portrait orientation)')
    plt.axis('off')
    
    # 6. Flip horizontal - Mirror Lena
    flip_matrix = AffineTransformations.scaling_matrix(-1, 1)
    # Perlu tambah translation untuk center
    center_x = image.shape[1] // 2
    flip_combined = AffineTransformations.combine_transforms(
        AffineTransformations.translation_matrix(center_x, 0),
        flip_matrix,
        AffineTransformations.translation_matrix(center_x, 0)
    )
    flipped = AffineTransformations.apply_transform(image, flip_combined)
    
    plt.subplot(3, 4, 7)
    plt.imshow(flipped, cmap='gray')
    plt.title('Horizontal Flip\n(Mirror effect)')
    plt.axis('off')
    
    # 7. Kombinasi: Kecil + Putar + Geser
    small_rot_trans = AffineTransformations.combine_transforms(
        AffineTransformations.scaling_matrix(0.6, 0.6),
        AffineTransformations.rotation_matrix(15),
        AffineTransformations.translation_matrix(100, 80)
    )
    combined1 = AffineTransformations.apply_transform(image, small_rot_trans)
    
    plt.subplot(3, 4, 8)
    plt.imshow(combined1, cmap='gray')
    plt.title('Small + Rotate + Move\n(0.6x scale, 15°, translate)')
    plt.axis('off')
    
    # 8. Perspective-like dengan shear
    perspective_matrix = AffineTransformations.shear_matrix(0.3, -0.1)
    perspective = AffineTransformations.apply_transform(image, perspective_matrix)
    
    plt.subplot(3, 4, 9)
    plt.imshow(perspective, cmap='gray')
    plt.title('Perspective Effect\n(Using shear transformation)')
    plt.axis('off')
    
    # 9. Extreme rotation
    rot_extreme = AffineTransformations.rotation_matrix(45)
    rotated_extreme = AffineTransformations.apply_transform(image, rot_extreme)
    
    plt.subplot(3, 4, 10)
    plt.imshow(rotated_extreme, cmap='gray')
    plt.title('45° Rotation\n(Diamond orientation)')
    plt.axis('off')
    
    # 10. Squeeze effect
    squeeze_matrix = AffineTransformations.scaling_matrix(0.5, 1.5)
    squeezed = AffineTransformations.apply_transform(image, squeeze_matrix)
    
    plt.subplot(3, 4, 11)
    plt.imshow(squeezed, cmap='gray')
    plt.title('Squeeze Effect\n(Width 0.5x, Height 1.5x)')
    plt.axis('off')
    
    # 11. Complex combination
    complex_matrix = AffineTransformations.combine_transforms(
        AffineTransformations.rotation_matrix(10),
        AffineTransformations.shear_matrix(0.2, 0.1),
        AffineTransformations.scaling_matrix(0.8, 1.1),
        AffineTransformations.translation_matrix(30, -20)
    )
    complex_result = AffineTransformations.apply_transform(image, complex_matrix)
    
    plt.subplot(3, 4, 12)
    plt.imshow(complex_result, cmap='gray')
    plt.title('Complex Transform\n(Rotate+Shear+Scale+Move)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def print_transformation_matrices():
    """
    Menampilkan contoh matriks transformasi
    """
    print("=== MATRIKS TRANSFORMASI AFFINE ===\n")
    
    print("1. Translation (dx=10, dy=20):")
    print(AffineTransformations.translation_matrix(10, 20))
    print()
    
    print("2. Scaling (sx=2, sy=0.5):")
    print(AffineTransformations.scaling_matrix(2, 0.5))
    print()
    
    print("3. Rotation (45 degrees):")
    print(AffineTransformations.rotation_matrix(45))
    print()
    
    print("4. Shearing (shx=0.5, shy=0.2):")
    print(AffineTransformations.shear_matrix(0.5, 0.2))
    print()

def manual_point_transformation():
    """
    Contoh transformasi manual pada titik-titik
    """
    print("=== TRANSFORMASI MANUAL PADA TITIK ===\n")
    
    # Definisikan beberapa titik
    points = np.array([
        [0, 0, 1],    # titik (0,0)
        [100, 0, 1],  # titik (100,0)
        [100, 100, 1],# titik (100,100)
        [0, 100, 1]   # titik (0,100)
    ]).T  # Transpose untuk mendapatkan format 3xN
    
    print("Titik-titik asli:")
    print("x:", points[0, :])
    print("y:", points[1, :])
    print()
    
    # Rotasi 45 derajat
    rot_matrix = AffineTransformations.rotation_matrix(45)
    rotated_points = np.dot(rot_matrix, points)
    
    print("Setelah rotasi 45 derajat:")
    print("x':", np.round(rotated_points[0, :], 2))
    print("y':", np.round(rotated_points[1, :], 2))
    print()
    
    # Scaling 2x di x, 0.5x di y
    scale_matrix = AffineTransformations.scaling_matrix(2, 0.5)
    scaled_points = np.dot(scale_matrix, points)
    
    print("Setelah scaling (sx=2, sy=0.5):")
    print("x':", scaled_points[0, :])
    print("y':", scaled_points[1, :])

def demo_lena_applications():
    """
    Demo aplikasi praktis transformasi affine pada gambar Lena
    """
    # Load Lena image
    try:
        image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError("lena.png tidak ditemukan")
    except:
        print("⚠️  lena.png tidak ditemukan!")
        print("Silakan download dari: https://upload.wikimedia.org/wikipedia/en/7/7d/Lena_%28test_image%29.png")
        print("Dan simpan sebagai 'lena.png' di direktori yang sama dengan script ini")
        return
    
    plt.figure(figsize=(18, 14))
    
    # Original
    plt.subplot(3, 5, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Lena\n512x512', fontsize=10)
    plt.axis('off')
    
    # Photo correction scenarios
    
    # 1. Document Scanner Correction
    skewed_matrix = AffineTransformations.rotation_matrix(8)  # Simulasi scan miring
    skewed = AffineTransformations.apply_transform(image, skewed_matrix)
    corrected_matrix = AffineTransformations.rotation_matrix(-8)  # Koreksi
    corrected = AffineTransformations.apply_transform(skewed, corrected_matrix)
    
    plt.subplot(3, 5, 2)
    plt.imshow(skewed, cmap='gray')
    plt.title('Skewed Scan\n(8° tilt)', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 5, 3)
    plt.imshow(corrected, cmap='gray')
    plt.title('Auto-Corrected\n(-8° correction)', fontsize=10)
    plt.axis('off')
    
    # 2. Social Media Filters
    # Instagram-style crop dan zoom
    zoom_crop = AffineTransformations.combine_transforms(
        AffineTransformations.scaling_matrix(1.5, 1.5),
        AffineTransformations.translation_matrix(-128, -128)
    )
    zoomed = AffineTransformations.apply_transform(image, zoom_crop)
    
    plt.subplot(3, 5, 4)
    plt.imshow(zoomed, cmap='gray')
    plt.title('Social Media Zoom\n(1.5x crop)', fontsize=10)
    plt.axis('off')
    
    # 3. Face Recognition Normalization
    # Simulasi face alignment
    face_align = AffineTransformations.combine_transforms(
        AffineTransformations.rotation_matrix(5),   # Koreksi tilt wajah
        AffineTransformations.scaling_matrix(1.1, 1.0),  # Normalisasi aspect ratio
        AffineTransformations.translation_matrix(-20, -10)
    )
    aligned_face = AffineTransformations.apply_transform(image, face_align)
    
    plt.subplot(3, 5, 5)
    plt.imshow(aligned_face, cmap='gray')
    plt.title('Face Alignment\n(for recognition)', fontsize=10)
    plt.axis('off')
    
    # 4. Medical Image Processing
    # Simulasi koreksi orientasi medical scan
    medical_matrix = AffineTransformations.combine_transforms(
        AffineTransformations.rotation_matrix(90),  # Orientasi standar
        AffineTransformations.scaling_matrix(0.8, 0.8)  # Resize untuk analysis
    )
    medical = AffineTransformations.apply_transform(image, medical_matrix)
    
    plt.subplot(3, 5, 6)
    plt.imshow(medical, cmap='gray')
    plt.title('Medical Orientation\n(90° + resize)', fontsize=10)
    plt.axis('off')
    
    # 5. Game Sprite Transformation
    # Simulasi karakter game berputar
    game_sprite = AffineTransformations.combine_transforms(
        AffineTransformations.scaling_matrix(0.6, 0.6),    # Ukuran sprite
        AffineTransformations.rotation_matrix(30),         # Rotasi karakter
        AffineTransformations.translation_matrix(80, 100)   # Posisi di game
    )
    sprite = AffineTransformations.apply_transform(image, game_sprite)
    
    plt.subplot(3, 5, 7)
    plt.imshow(sprite, cmap='gray')
    plt.title('Game Sprite\n(0.6x, 30°, moved)', fontsize=10)
    plt.axis('off')
    
    # 6. Augmentasi Data untuk AI
    # Variasi data training
    augment1 = AffineTransformations.combine_transforms(
        AffineTransformations.rotation_matrix(-12),
        AffineTransformations.scaling_matrix(0.9, 1.1),
        AffineTransformations.shear_matrix(0.1, 0.05)
    )
    augmented = AffineTransformations.apply_transform(image, augment1)
    
    plt.subplot(3, 5, 8)
    plt.imshow(augmented, cmap='gray')
    plt.title('Data Augmentation\n(AI training)', fontsize=10)
    plt.axis('off')
    
    # 7. Perspective Correction
    # Simulasi koreksi perspektif foto dokumen
    perspective_fix = AffineTransformations.combine_transforms(
        AffineTransformations.shear_matrix(-0.2, 0.1),
        AffineTransformations.scaling_matrix(1.2, 0.9)
    )
    perspective = AffineTransformations.apply_transform(image, perspective_fix)
    
    plt.subplot(3, 5, 9)
    plt.imshow(perspective, cmap='gray')
    plt.title('Perspective Fix\n(document scan)', fontsize=10)
    plt.axis('off')
    
    # 8. Artistic Effect
    # Efek seni digital
    artistic = AffineTransformations.combine_transforms(
        AffineTransformations.shear_matrix(0.3, -0.1),
        AffineTransformations.rotation_matrix(15),
        AffineTransformations.scaling_matrix(0.8, 1.2)
    )
    art_effect = AffineTransformations.apply_transform(image, artistic)
    
    plt.subplot(3, 5, 10)
    plt.imshow(art_effect, cmap='gray')
    plt.title('Artistic Effect\n(creative transform)', fontsize=10)
    plt.axis('off')
    
    # Performance metrics
    rows, cols = image.shape
    
    plt.subplot(3, 5, 11)
    plt.text(0.1, 0.8, f"Original Size: {rows}x{cols}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Data Type: {image.dtype}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Memory: {image.nbytes/1024:.1f} KB", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.2, "Affine Transform: O(n)", fontsize=12, transform=plt.gca().transAxes)
    plt.title('Image Info', fontsize=10)
    plt.axis('off')
    
    # Common applications text
    applications = [
        "📱 Instagram Filters",
        "🏥 Medical Imaging", 
        "🎮 Game Development",
        "🤖 AI Data Augmentation",
        "📄 Document Scanning",
        "👤 Face Recognition",
        "🎨 Digital Art",
        "📷 Photo Correction"
    ]
    
    plt.subplot(3, 5, 12)
    for i, app in enumerate(applications):
        plt.text(0.05, 0.9 - i*0.11, app, fontsize=11, transform=plt.gca().transAxes)
    plt.title('Real Applications', fontsize=10)
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Lena Image: Practical Affine Transformation Applications', fontsize=16, y=0.98)
    plt.show()

# Tambahkan fungsi baru untuk load dan test Lena
def test_lena_loading():
    """
    Test loading gambar Lena dan tampilkan info
    """
    try:
        # Coba load dengan OpenCV
        image_bgr = cv2.imread('lena.png')
        image_gray = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
        
        if image_bgr is None:
            print("❌ lena.png tidak ditemukan!")
            print("\n📥 Cara mendapatkan gambar Lena:")
            print("1. Download dari: https://upload.wikimedia.org/wikipedia/en/7/7d/Lena_%28test_image%29.png")
            print("2. Atau gunakan: https://www.cs.cmu.edu/~chuck/lennapg/lenna50.jpg")
            print("3. Simpan sebagai 'lena.png' di folder yang sama dengan script")
            print("4. Pastikan ukuran sekitar 512x512 pixels")
            return False
            
        print("✅ Lena image berhasil dimuat!")
        print(f"📊 Color shape: {image_bgr.shape}")
        print(f"📊 Gray shape: {image_gray.shape}")
        print(f"📊 Data type: {image_gray.dtype}")
        print(f"💾 File size: {image_gray.nbytes/1024:.1f} KB")
        
        # Tampilkan preview
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        plt.title('Lena (Color)')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(image_gray, cmap='gray')
        plt.title('Lena (Grayscale)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🖼️  AFFINE TRANSFORMATIONS DENGAN LENA IMAGE")
    print("=" * 50)
    
    # Test loading Lena image
    print("\n1️⃣  Testing Lena image loading...")
    lena_available = test_lena_loading()
    
    if lena_available:
        print("\n2️⃣  Running basic transformations demo...")
        demo_transformations()
        
        print("\n3️⃣  Running practical applications demo...")
        demo_lena_applications()
    else:
        print("\n⚠️  Menjalankan demo dengan gambar alternatif...")
        demo_transformations()
    
    print("\n4️⃣  Transformation matrices examples...")
    print_transformation_matrices()
    
    print("\n5️⃣  Manual point transformation...")
    manual_point_transformation()
    
    print("\n✅ Demo selesai!")
