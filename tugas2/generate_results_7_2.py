import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import os

# Buat folder hasil jika belum ada
os.makedirs('hasil', exist_ok=True)

class SpatialTransformer:
    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image_path = image_path
        
    def create_affine_transform(self, transform_type, **params):
        """Create different types of affine transformation matrices"""
        if transform_type == 'scale':
            sx = params.get('sx', 1)
            sy = params.get('sy', 1)
            matrix = np.array([[sx, 0, 0],
                              [0, sy, 0],
                              [0, 0, 1]], dtype=np.float32)
            
        elif transform_type == 'rotation':
            theta = params.get('theta', 0) * np.pi / 180
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            matrix = np.array([[cos_theta, -sin_theta, 0],
                              [sin_theta, cos_theta, 0],
                              [0, 0, 1]], dtype=np.float32)
            
        elif transform_type == 'translation':
            dx = params.get('dx', 0)
            dy = params.get('dy', 0)
            matrix = np.array([[1, 0, dx],
                              [0, 1, dy],
                              [0, 0, 1]], dtype=np.float32)
            
        elif transform_type == 'shear':
            shx = params.get('shx', 0)
            shy = params.get('shy', 0)
            matrix = np.array([[1, shy, 0],
                              [shx, 1, 0],
                              [0, 0, 1]], dtype=np.float32)
        else:
            matrix = np.eye(3, dtype=np.float32)
            
        return matrix
    
    def apply_affine_transform(self, transform_matrix, fill_value=0, output_shape=None):
        """Apply affine transformation to image"""
        transform_2x3 = transform_matrix[:2, :]
        
        if output_shape is None:
            output_shape = (self.original_image.shape[1], self.original_image.shape[0])
        
        transformed = cv2.warpAffine(self.original_image, transform_2x3, 
                                   output_shape, flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=fill_value)
        
        return transformed

def save_results_7_2():
    """Generate and save Tutorial 7.2 results"""
    print("=== Menghasilkan dan Menyimpan Hasil Tutorial 7.2 ===\n")
    
    # Load image
    image_path = '../assets/cameraman.tif'
    if not os.path.exists(image_path):
        alt_paths = ['../assets/cameraman2.tif', '../assets/moon.tif', '../assets/lena.png']
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                image_path = alt_path
                print(f"Menggunakan citra alternatif: {image_path}")
                break
    
    transformer = SpatialTransformer(image_path)
    
    # 1. Scaling transformation
    sx, sy = 2, 2
    scale_matrix = transformer.create_affine_transform('scale', sx=sx, sy=sy)
    scaled_affine = transformer.apply_affine_transform(scale_matrix, 
                                                      output_shape=(512, 512))
    
    # Compare with imresize
    scaled_resize = cv2.resize(transformer.original_image, (512, 512), 
                              interpolation=cv2.INTER_CUBIC)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(transformer.original_image, cmap='gray')
    axes[0].set_title('Citra Asli', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(scaled_affine, cmap='gray')
    axes[1].set_title('Menggunakan Transformasi Affine', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(scaled_resize, cmap='gray')
    axes[2].set_title('Menggunakan Image Resizing', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('hasil/08_scaling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Rotation transformation
    theta = 35
    rotation_matrix = transformer.create_affine_transform('rotation', theta=theta)
    
    # Calculate output size
    h, w = transformer.original_image.shape
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
    rotated_corners = rotation_matrix @ corners
    
    min_x, max_x = rotated_corners[0].min(), rotated_corners[0].max()
    min_y, max_y = rotated_corners[1].min(), rotated_corners[1].max()
    
    translation_matrix = transformer.create_affine_transform('translation', 
                                                           dx=-min_x, dy=-min_y)
    combined_matrix = translation_matrix @ rotation_matrix
    
    output_size = (int(max_x - min_x), int(max_y - min_y))
    rotated_affine = transformer.apply_affine_transform(combined_matrix, 
                                                       output_shape=output_size)
    
    # Compare with OpenCV rotation
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, theta, 1.0)
    
    cos_theta = abs(M[0, 0])
    sin_theta = abs(M[0, 1])
    new_w = int((h * sin_theta) + (w * cos_theta))
    new_h = int((h * cos_theta) + (w * sin_theta))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated_opencv = cv2.warpAffine(transformer.original_image, M, (new_w, new_h))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(transformer.original_image, cmap='gray')
    axes[0].set_title('Citra Asli', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(rotated_affine, cmap='gray')
    axes[1].set_title('Menggunakan Transformasi Affine', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(rotated_opencv, cmap='gray')
    axes[2].set_title('Menggunakan Rotasi OpenCV', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('hasil/09_rotation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Translation transformation
    delta_x, delta_y = 50, 100
    translation_matrix = transformer.create_affine_transform('translation', dx=delta_x, dy=delta_y)
    
    output_w = transformer.original_image.shape[1] + delta_x
    output_h = transformer.original_image.shape[0] + delta_y
    
    translated = transformer.apply_affine_transform(translation_matrix, 
                                                   fill_value=128,
                                                   output_shape=(output_w, output_h))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(transformer.original_image, cmap='gray')
    axes[0].set_title('Citra Asli', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(translated, cmap='gray')
    axes[1].set_title('Citra Ditranslasi (dx=50, dy=100)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('hasil/10_translation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Shearing transformation
    sh_x, sh_y = 0.3, 0.2  # Reduced shearing values for better visualization
    shear_matrix = transformer.create_affine_transform('shear', shx=sh_x, shy=sh_y)
    
    # Calculate bounds
    h, w = transformer.original_image.shape
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
    sheared_corners = shear_matrix @ corners
    
    min_x, max_x = sheared_corners[0].min(), sheared_corners[0].max()
    min_y, max_y = sheared_corners[1].min(), sheared_corners[1].max()
    
    if min_x < 0 or min_y < 0:
        translation_adj = transformer.create_affine_transform('translation', 
                                                            dx=-min(min_x, 0), 
                                                            dy=-min(min_y, 0))
        combined_shear = translation_adj @ shear_matrix
    else:
        combined_shear = shear_matrix
    
    output_size = (int(max_x - min_x), int(max_y - min_y))
    sheared = transformer.apply_affine_transform(combined_shear, output_shape=output_size)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(transformer.original_image, cmap='gray')
    axes[0].set_title('Citra Asli', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(sheared, cmap='gray')
    axes[1].set_title(f'Citra Shearing (shx={sh_x}, shy={sh_y})', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('hasil/11_shearing.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Create registration demo images
    base_image = transformer.original_image.copy()
    
    # Create a transformed version for unregistered image
    center = (base_image.shape[1]//2, base_image.shape[0]//2)
    M_demo = cv2.getRotationMatrix2D(center, 10, 0.95)  # 10 degrees, 0.95 scale
    M_demo[0, 2] += 15  # translate x
    M_demo[1, 2] += 10  # translate y
    unregistered_image = cv2.warpAffine(base_image, M_demo, 
                                       (base_image.shape[1], base_image.shape[0]))
    
    # Add some noise
    noise = np.random.normal(0, 5, unregistered_image.shape).astype(np.int16)
    unregistered_image = np.clip(unregistered_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Simple registration using similarity transform (for demo)
    try:
        # Create some control points for demo
        src_pts = np.array([[50, 50], [200, 50], [200, 200], [50, 200]], dtype=np.float32)
        dst_pts = np.array([[60, 55], [210, 48], [195, 205], [45, 198]], dtype=np.float32)
        
        # Estimate similarity transform
        transform_obj = transform.SimilarityTransform()
        transform_obj.estimate(src_pts, dst_pts)
        
        # Apply registration
        registered_image = transform.warp(unregistered_image, transform_obj.inverse, 
                                        output_shape=base_image.shape, preserve_range=True)
        registered_image = (registered_image).astype(np.uint8)
        
    except:
        # Simple fallback if transform fails
        registered_image = unregistered_image
    
    # Display registration results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].imshow(base_image, cmap='gray')
    axes[0,0].set_title('Citra Referensi (Base)', fontsize=12, fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(unregistered_image, cmap='gray')
    axes[0,1].set_title('Citra Tidak Teregistrasi', fontsize=12, fontweight='bold')
    axes[0,1].axis('off')
    
    axes[1,0].imshow(registered_image, cmap='gray')
    axes[1,0].set_title('Citra Teregistrasi', fontsize=12, fontweight='bold')
    axes[1,0].axis('off')
    
    # Overlay visualization
    overlay = np.zeros((*base_image.shape, 3))
    overlay[:,:,0] = base_image / 255.0  # Red for base
    overlay[:,:,1] = registered_image / 255.0  # Green for registered
    overlay = np.clip(overlay, 0, 1)
    
    axes[1,1].imshow(overlay)
    axes[1,1].set_title('Overlay (Merah: Base, Hijau: Teregistrasi)', fontsize=12, fontweight='bold')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('hasil/12_image_registration.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Semua hasil Tutorial 7.2 berhasil disimpan di folder 'hasil'")

if __name__ == "__main__":
    save_results_7_2()
