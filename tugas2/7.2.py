import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from skimage import transform, feature, measure
from scipy import ndimage
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os

class SpatialTransformer:
    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image_path = image_path
        
    def create_affine_transform(self, transform_type, **params):
        """Create different types of affine transformation matrices"""
        if transform_type == 'scale':
            sx = params.get('sx', 1)
            sy = params.get('sy', 1)
            # Transformation matrix: [sx 0 0; 0 sy 0; 0 0 1]
            matrix = np.array([[sx, 0, 0],
                              [0, sy, 0],
                              [0, 0, 1]], dtype=np.float32)
            
        elif transform_type == 'rotation':
            theta = params.get('theta', 0) * np.pi / 180  # Convert to radians
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            # Rotation matrix: [cos -sin 0; sin cos 0; 0 0 1]
            matrix = np.array([[cos_theta, -sin_theta, 0],
                              [sin_theta, cos_theta, 0],
                              [0, 0, 1]], dtype=np.float32)
            
        elif transform_type == 'translation':
            dx = params.get('dx', 0)
            dy = params.get('dy', 0)
            # Translation matrix: [1 0 dx; 0 1 dy; 0 0 1]
            matrix = np.array([[1, 0, dx],
                              [0, 1, dy],
                              [0, 0, 1]], dtype=np.float32)
            
        elif transform_type == 'shear':
            shx = params.get('shx', 0)
            shy = params.get('shy', 0)
            # Shear matrix: [1 shy 0; shx 1 0; 0 0 1]
            matrix = np.array([[1, shy, 0],
                              [shx, 1, 0],
                              [0, 0, 1]], dtype=np.float32)
        else:
            matrix = np.eye(3, dtype=np.float32)
            
        return matrix
    
    def apply_affine_transform(self, transform_matrix, fill_value=0, output_shape=None):
        """Apply affine transformation to image"""
        # Convert 3x3 matrix to 2x3 for cv2.warpAffine
        transform_2x3 = transform_matrix[:2, :]
        
        if output_shape is None:
            output_shape = (self.original_image.shape[1], self.original_image.shape[0])
        
        transformed = cv2.warpAffine(self.original_image, transform_2x3, 
                                   output_shape, flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=fill_value)
        
        return transformed

class ControlPointSelector:
    def __init__(self, image1, image2, window_title="Control Point Selection"):
        self.image1 = image1
        self.image2 = image2
        self.points1 = []
        self.points2 = []
        self.current_image = 1
        self.window_title = window_title
        
    def select_points_matplotlib(self, max_points=10):
        """Interactive control point selection using matplotlib"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1.imshow(self.image1, cmap='gray')
        ax1.set_title('Image 1 - Click to select points')
        ax1.axis('off')
        
        ax2.imshow(self.image2, cmap='gray')
        ax2.set_title('Image 2 - Click corresponding points')
        ax2.axis('off')
        
        points1, points2 = [], []
        colors = plt.cm.rainbow(np.linspace(0, 1, max_points))
        
        def onclick(event):
            if event.inaxes == ax1 and len(points1) < max_points:
                points1.append([event.xdata, event.ydata])
                ax1.plot(event.xdata, event.ydata, 'o', color=colors[len(points1)-1], 
                        markersize=8, markeredgecolor='white', markeredgewidth=2)
                ax1.text(event.xdata, event.ydata-10, f'{len(points1)}', 
                        color='white', fontweight='bold', ha='center')
                
            elif event.inaxes == ax2 and len(points2) < len(points1):
                points2.append([event.xdata, event.ydata])
                ax2.plot(event.xdata, event.ydata, 'o', color=colors[len(points2)-1], 
                        markersize=8, markeredgecolor='white', markeredgewidth=2)
                ax2.text(event.xdata, event.ydata-10, f'{len(points2)}', 
                        color='white', fontweight='bold', ha='center')
                
                if len(points2) == max_points:
                    print(f"Selected {len(points1)} control points. Close the window to continue.")
                    
            plt.draw()
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        print(f"Instructions:")
        print(f"1. Click on distinctive features in Image 1")
        print(f"2. Click on corresponding features in Image 2")
        print(f"3. Select up to {max_points} point pairs")
        print(f"4. Close the window when done")
        
        plt.show()
        
        self.points1 = np.array(points1, dtype=np.float32) if points1 else np.array([])
        self.points2 = np.array(points2, dtype=np.float32) if points2 else np.array([])
        
        return self.points1, self.points2

class ImageRegistrar:
    def __init__(self):
        pass
    
    def fine_tune_points(self, points1, points2, image1, image2, window_size=11):
        """Fine-tune control points using correlation (equivalent to cpcorr)"""
        refined_points1 = []
        
        for i, (p1, p2) in enumerate(zip(points1, points2)):
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])
            
            # Extract template from image1
            half_win = window_size // 2
            if (y1-half_win >= 0 and y1+half_win < image1.shape[0] and 
                x1-half_win >= 0 and x1+half_win < image1.shape[1]):
                
                template = image1[y1-half_win:y1+half_win+1, x1-half_win:x1+half_win+1]
                
                # Search in image2
                search_size = window_size * 2
                y2_start = max(0, y2-search_size)
                y2_end = min(image2.shape[0], y2+search_size)
                x2_start = max(0, x2-search_size)
                x2_end = min(image2.shape[1], x2+search_size)
                
                search_region = image2[y2_start:y2_end, x2_start:x2_end]
                
                # Template matching
                if search_region.size > 0 and template.size > 0:
                    result = cv2.matchTemplate(search_region.astype(np.float32), 
                                             template.astype(np.float32), 
                                             cv2.TM_CCOEFF_NORMED)
                    _, _, _, max_loc = cv2.minMaxLoc(result)
                    
                    # Refined position
                    refined_x = x2_start + max_loc[0] + half_win
                    refined_y = y2_start + max_loc[1] + half_win
                    refined_points1.append([refined_x, refined_y])
                else:
                    refined_points1.append([x2, y2])
            else:
                refined_points1.append([x2, y2])
        
        return np.array(refined_points1, dtype=np.float32)
    
    def estimate_transform(self, points1, points2, transform_type='similarity'):
        """Estimate transformation parameters from control points"""
        if transform_type == 'similarity':
            # Non-reflective similarity transform (translation, rotation, scaling)
            transform_obj = transform.SimilarityTransform()
        elif transform_type == 'affine':
            transform_obj = transform.AffineTransform()
        elif transform_type == 'projective':
            transform_obj = transform.ProjectiveTransform()
        
        # Fit the transformation
        transform_obj.estimate(points1, points2)
        
        return transform_obj
    
    def apply_registration(self, image, transform_obj, output_shape=None, preserve_range=True):
        """Apply the estimated transformation to register the image"""
        if output_shape is None:
            output_shape = image.shape
            
        registered = transform.warp(image, transform_obj.inverse, 
                                  output_shape=output_shape, preserve_range=preserve_range)
        
        if preserve_range:
            registered = (registered * 255).astype(np.uint8)
            
        return registered

def tutorial_7_2():
    """Complete Tutorial 7.2 implementation"""
    print("=== TUTORIAL 7.2: TRANSFORMASI SPASIAL DAN REGISTRASI CITRA ===\n")
    
    # Load cameraman image
    image_path = '../assets/cameraman.tif'
    if not os.path.exists(image_path):
        # Try alternative paths
        alt_paths = ['../assets/cameraman2.tif', '../assets/moon.tif', '../assets/lena.png']
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                image_path = alt_path
                print(f"Menggunakan citra alternatif: {image_path}")
                break
        else:
            print(f"Peringatan: Tidak ditemukan citra yang sesuai. Membuat citra contoh untuk demonstrasi.")
            sample_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            cv2.imwrite('sample_image.tif', sample_img)
            image_path = 'sample_image.tif'
    
    transformer = SpatialTransformer(image_path)
    
    print("Langkah 1-4: Transformasi Affine - Penskalaan")
    
    # Step 1-4: Scaling transformation
    sx, sy = 2, 2
    scale_matrix = transformer.create_affine_transform('scale', sx=sx, sy=sy)
    scaled_affine = transformer.apply_affine_transform(scale_matrix, 
                                                      output_shape=(512, 512))
    
    # Compare with imresize
    scaled_resize = cv2.resize(transformer.original_image, (512, 512), 
                              interpolation=cv2.INTER_CUBIC)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(transformer.original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(scaled_affine, cmap='gray')
    axes[1].set_title('Using Affine Transformation')
    axes[1].axis('off')
    
    axes[2].imshow(scaled_resize, cmap='gray')
    axes[2].set_title('Using Image Resizing')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Question 1: Compare the two resulting images (scaled_affine and scaled_resize):")
    print(f"Affine result size: {scaled_affine.shape}")
    print(f"Resize result size: {scaled_resize.shape}")
    print("Differences: Affine transformation may include padding and different interpolation behavior.")
    
    # Step 5-7: Rotation transformation
    print("\nStep 5-7: Affine transformations - Rotation")
    
    theta = 35
    rotation_matrix = transformer.create_affine_transform('rotation', theta=theta)
    
    # Calculate output size to accommodate rotation
    h, w = transformer.original_image.shape
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
    rotated_corners = rotation_matrix @ corners
    
    min_x, max_x = rotated_corners[0].min(), rotated_corners[0].max()
    min_y, max_y = rotated_corners[1].min(), rotated_corners[1].max()
    
    # Adjust translation to center the rotated image
    translation_matrix = transformer.create_affine_transform('translation', 
                                                           dx=-min_x, dy=-min_y)
    combined_matrix = translation_matrix @ rotation_matrix
    
    output_size = (int(max_x - min_x), int(max_y - min_y))
    rotated_affine = transformer.apply_affine_transform(combined_matrix, 
                                                       output_shape=output_size)
    
    # Compare with imrotate equivalent
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
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(rotated_affine, cmap='gray')
    axes[1].set_title('Using Affine Transformation')
    axes[1].axis('off')
    
    axes[2].imshow(rotated_opencv, cmap='gray')
    axes[2].set_title('Using Image Rotation')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Question 2: Compare the two resulting images (rotated_affine and rotated_opencv):")
    print("Both methods produce similar results but may differ in boundary handling and interpolation.")
    
    # Step 8-10: Translation transformation
    print("\nStep 8-10: Affine transformations - Translation")
    
    delta_x, delta_y = 50, 100
    translation_matrix = transformer.create_affine_transform('translation', dx=delta_x, dy=delta_y)
    
    # Calculate output size to accommodate translation
    output_w = transformer.original_image.shape[1] + delta_x
    output_h = transformer.original_image.shape[0] + delta_y
    
    translated = transformer.apply_affine_transform(translation_matrix, 
                                                   fill_value=128,  # gray fill
                                                   output_shape=(output_w, output_h))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(transformer.original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(translated, cmap='gray')
    axes[1].set_title('Translated Image (dx=50, dy=100)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Question 3: Compare the two images (original and translated):")
    print(f"Original size: {transformer.original_image.shape}")
    print(f"Translated size: {translated.shape}")
    print("The translated image is larger to accommodate the offset, with gray fill for empty areas.")
    
    # Step 11-13: Shearing transformation
    print("\nStep 11-13: Affine transformations - Shearing")
    
    sh_x, sh_y = 2, 1.5
    shear_matrix = transformer.create_affine_transform('shear', shx=sh_x, shy=sh_y)
    
    # Calculate bounds for sheared image
    h, w = transformer.original_image.shape
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
    sheared_corners = shear_matrix @ corners
    
    min_x, max_x = sheared_corners[0].min(), sheared_corners[0].max()
    min_y, max_y = sheared_corners[1].min(), sheared_corners[1].max()
    
    # Adjust for negative coordinates
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
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(sheared, cmap='gray')
    axes[1].set_title(f'Sheared Image (shx={sh_x}, shy={sh_y})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Image Registration Section
    print("\n" + "="*60)
    print("IMAGE REGISTRATION SECTION")
    print("="*60)
    
    # Step 14: Load base and unregistered images
    print("\nStep 14: Load base and unregistered images")
    
    # Load images for registration demo
    base_path = '../assets/cameraman.tif'
    
    # Initialize variables
    base_image = None
    unregistered_image = None
    
    # Try to load from available assets
    available_images = ['../assets/cameraman.tif', '../assets/moon.tif', '../assets/lena.png', '../assets/tire.tif']
    
    for img_path in available_images:
        if os.path.exists(img_path):
            base_path = img_path
            print(f"Using {base_path} for registration demo, creating transformed version...")
            break
    
    if os.path.exists(base_path):
        # Load the base image
        base_image = cv2.imread(base_path, cv2.IMREAD_GRAYSCALE)
        
        if base_image is not None:
            # Create a transformed version for unregistered image
            center = (base_image.shape[1]//2, base_image.shape[0]//2)
            M_demo = cv2.getRotationMatrix2D(center, 10, 0.95)  # 10 degrees, 0.95 scale
            M_demo[0, 2] += 15  # translate x
            M_demo[1, 2] += 10  # translate y
            unregistered_image = cv2.warpAffine(base_image, M_demo, 
                                               (base_image.shape[1], base_image.shape[0]))
            
            # Add some noise for realism
            noise = np.random.normal(0, 5, unregistered_image.shape).astype(np.int16)
            unregistered_image = np.clip(unregistered_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # If no assets available, create sample images
    if base_image is None:
        print("No suitable images found in assets. Creating sample images for registration demo...")
        
        # Create base image
        base_image = np.zeros((300, 400), dtype=np.uint8)
        cv2.rectangle(base_image, (50, 50), (150, 150), 255, -1)
        cv2.circle(base_image, (250, 100), 40, 128, -1)
        cv2.rectangle(base_image, (200, 200), (350, 250), 200, -1)
        
        # Create unregistered image (rotated and translated version)
        center = (200, 150)
        M_demo = cv2.getRotationMatrix2D(center, 15, 0.9)  # 15 degrees, 0.9 scale
        M_demo[0, 2] += 30  # translate x
        M_demo[1, 2] += 20  # translate y
        unregistered_image = cv2.warpAffine(base_image, M_demo, (400, 300))
        
        # Add some noise
        noise = np.random.normal(0, 10, unregistered_image.shape).astype(np.int16)
        unregistered_image = np.clip(unregistered_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Ensure we have valid images
    if base_image is None or unregistered_image is None:
        print("Error: Could not load images properly. Exiting registration section.")
        return
    
    # Display images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(base_image, cmap='gray')
    axes[0].set_title('Base Image')
    axes[0].axis('off')
    
    axes[1].imshow(unregistered_image, cmap='gray')
    axes[1].set_title('Unregistered Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Step 15: Control point selection
    print("\nStep 15: Control Point Selection")
    print("Interactive control point selection will open. Select corresponding points in both images.")
    
    selector = ControlPointSelector(unregistered_image, base_image)
    input_points, base_points = selector.select_points_matplotlib(max_points=10)
    
    if len(input_points) > 0 and len(base_points) > 0:
        print(f"\nSelected {len(input_points)} control point pairs")
        print("Input points (unregistered image):", input_points[:5])  # Show first 5
        print("Base points (base image):", base_points[:5])
        
        # Step 17: Fine-tune control points
        print("\nStep 17: Fine-tuning control points")
        registrar = ImageRegistrar()
        input_points_adj = registrar.fine_tune_points(input_points, base_points, 
                                                    unregistered_image, base_image)
        
        print("Adjusted input points:", input_points_adj[:5])
        print("Question 4: Compare input_points_adj with input_points.")
        print("Fine-tuning may show small adjustments based on correlation matching.")
        
        # Step 18-20: Estimate and apply transformation
        print("\nStep 18-20: Estimate and apply transformation")
        
        # Use similarity transformation (translation, rotation, scaling)
        if len(input_points) >= 2:  # Need at least 2 points for similarity transform
            transform_obj = registrar.estimate_transform(input_points, base_points, 
                                                       transform_type='similarity')
            
            # Apply registration
            registered_image = registrar.apply_registration(unregistered_image, transform_obj,
                                                          output_shape=base_image.shape)
            
            # Step 21: Display results
            print("\nStep 21: Display registered image overlaid on base image")
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original images
            axes[0,0].imshow(base_image, cmap='gray')
            axes[0,0].set_title('Base Image')
            axes[0,0].axis('off')
            
            axes[0,1].imshow(unregistered_image, cmap='gray')
            axes[0,1].set_title('Unregistered Image')
            axes[0,1].axis('off')
            
            # Registered result
            axes[1,0].imshow(registered_image, cmap='gray')
            axes[1,0].set_title('Registered Image')
            axes[1,0].axis('off')
            
            # Overlay visualization
            overlay = np.zeros((*base_image.shape, 3))
            overlay[:,:,0] = base_image / 255.0  # Red channel for base
            overlay[:,:,1] = registered_image / 255.0  # Green channel for registered
            overlay = np.clip(overlay, 0, 1)
            
            axes[1,1].imshow(overlay)
            axes[1,1].set_title('Overlay (Red: Base, Green: Registered)')
            axes[1,1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Calculate registration quality metrics
            diff_before = np.mean((base_image.astype(np.float32) - unregistered_image.astype(np.float32))**2)
            diff_after = np.mean((base_image.astype(np.float32) - registered_image.astype(np.float32))**2)
            
            print(f"\nRegistration Quality Assessment:")
            print(f"Mean squared difference before registration: {diff_before:.2f}")
            print(f"Mean squared difference after registration: {diff_after:.2f}")
            print(f"Improvement: {((diff_before - diff_after) / diff_before * 100):.1f}%")
            
            print("\nQuestion 5: Are you happy with the results?")
            print("The registration quality depends on:")
            print("- Accuracy of control point selection")
            print("- Number of control points used")
            print("- Type of distortion in the unregistered image")
            print("- Appropriate transformation model choice")
            
        else:
            print("Need at least 2 control points for similarity transformation")
    else:
        print("No control points selected. Registration cannot proceed.")
    
    print("\n" + "="*60)
    print("TUTORIAL COMPLETED")
    print("="*60)
    
    print("\nSummary of Python implementations:")
    print("1. Affine transformations: scaling, rotation, translation, shearing")
    print("2. Interactive control point selection")
    print("3. Control point fine-tuning using template matching")
    print("4. Transformation estimation and image registration")
    print("5. Registration quality assessment")

# Additional utility functions
def save_results(images_dict, prefix="tutorial_7_2"):
    """Save all resulting images"""
    for name, image in images_dict.items():
        filename = f"{prefix}_{name}.png"
        cv2.imwrite(filename, image)
        print(f"Saved: {filename}")

def compare_with_matlab_results(python_result, matlab_result_path):
    """Compare Python results with MATLAB results if available"""
    if os.path.exists(matlab_result_path):
        matlab_result = cv2.imread(matlab_result_path, cv2.IMREAD_GRAYSCALE)
        diff = cv2.absdiff(python_result, matlab_result)
        mse = np.mean(diff**2)
        print(f"Difference with MATLAB result (MSE): {mse:.2f}")
        return diff
    else:
        print(f"MATLAB result file {matlab_result_path} not found for comparison")
        return None

# Run the tutorial
if __name__ == "__main__":
    tutorial_7_2()
