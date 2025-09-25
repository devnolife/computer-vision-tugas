#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial 6 Computer Vision dengan GUI Interaktif
Fitur:
- Navigasi step by step
- Tampilan gambar berdampingan
- Tombol Next/Previous
- Deskripsi untuk setiap langkah
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from skimage.draw import polygon2mask

class TutorialGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Computer Vision devnolife")
        self.root.geometry("1200x800")
        
        # Current step tracking
        self.current_step = 0
        self.steps = []
        self.images = {}  # Cache for loaded images
        
        # Asset mapping according to requirements
        self.assets = {
            # Step 1: Brighten
            'tire.tif': 'assets/tire.tif',
            
            # Step 2: Blend
            'rice.png': 'assets/rice.png',
            'cameraman.tif': 'assets/cameraman.tif',
            
            # Steps 3,9: Subtract, AbsDiff, XOR
            'cameraman2.tif': 'assets/cameraman2.tif',
            
            # Steps 4,6: Dynamic scaling and division vs multiplication
            'moon.tif': 'assets/moon.tif',
            
            # Step 5: 3D planet effect
            'earth1.tif': 'assets/earth1.tif',
            'earth2.tif': 'assets/earth2.tif',
            
            # Step 7: Background division
            'gradient.tif': 'assets/gradient.tif',
            'gradient_with_text.tif': 'assets/gradient_with_text.tif',
            
            # Step 8: ROI mask
            'pout.tif': 'assets/pout.tif',
            
            # Steps 11-13, Q11: ROI darker/original
            'lindsay.tif': 'assets/lindsay.tif'
        }
        
        # Initialize steps
        self.init_steps()
        
        # Create UI
        self.create_widgets()
        
        # Load first step
        self.show_step()
    
    def init_steps(self):
        """Initialize all tutorial steps"""
        self.steps = [
            {
                'title': 'Step 1: Image Brightening',
                'description': 'Menambahkan konstanta ke setiap pixel untuk mencerahkan gambar',
                'function': self.step_1_brighten,
                'images_count': 2
            },
            {
                'title': 'Step 2: Image Blending',
                'description': 'Menggabungkan dua gambar dengan operasi penjumlahan',
                'function': self.step_2_blend,
                'images_count': 3
            },
            {
                'title': 'Step 3: Image Subtraction',
                'description': 'Operasi pengurangan antar gambar untuk deteksi perbedaan',
                'function': self.step_3_subtract,
                'images_count': 4
            },
            {
                'title': 'Step 4: Dynamic Scaling vs Brightening',
                'description': 'Perbandingan antara perkalian (scaling) dan penjumlahan (brightening)',
                'function': self.step_4_scaling,
                'images_count': 3
            },
            {
                'title': 'Step 5: 3D Effect with Multiplication',
                'description': 'Membuat efek 3D dengan mengalikan dua gambar',
                'function': self.step_5_3d_effect,
                'images_count': 3
            },
            {
                'title': 'Step 6: Division vs Multiplication',
                'description': 'Perbandingan pembagian dan perkalian untuk menggelapkan gambar',
                'function': self.step_6_division,
                'images_count': 3
            },
            {
                'title': 'Step 7: Background Subtraction',
                'description': 'Menghapus background dengan operasi pembagian',
                'function': self.step_7_background,
                'images_count': 4
            },
            {
                'title': 'Step 8: ROI and Logic Operations',
                'description': 'Operasi logika dan Region of Interest (ROI) dengan berbagai bentuk',
                'function': self.step_8_roi,
                'images_count': 6
            },
            {
                'title': 'Step 9: XOR Operations',
                'description': 'Operasi XOR untuk deteksi perbedaan antar gambar',
                'function': self.step_9_xor,
                'images_count': 3
            },
            {
                'title': 'Step 10: Custom ROI dengan Koordinat Titik',
                'description': 'Membuat ROI dengan koordinat titik kustom untuk pemotongan presisi',
                'function': self.step_10_custom_roi,
                'images_count': 4
            }
        ]
    
    def create_widgets(self):
        """Create UI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title and description frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.title_label = ttk.Label(info_frame, text="", font=('Arial', 16, 'bold'))
        self.title_label.pack()
        
        self.desc_label = ttk.Label(info_frame, text="", font=('Arial', 10), wraplength=1000)
        self.desc_label.pack(pady=(5, 0))
        
        # Progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT)
        self.progress_var = tk.StringVar(value="Step 1 of 9")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(side=tk.LEFT, padx=(10, 0))
        
        # Image display frame
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Navigation frame
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X)
        
        self.prev_btn = ttk.Button(nav_frame, text="◀ Previous", command=self.prev_step)
        self.prev_btn.pack(side=tk.LEFT)
        
        self.next_btn = ttk.Button(nav_frame, text="Next ▶", command=self.next_step)
        self.next_btn.pack(side=tk.RIGHT)
        
        # Process button
        self.process_btn = ttk.Button(nav_frame, text="🔄 Process Step", command=self.process_current_step)
        self.process_btn.pack(side=tk.LEFT, padx=(20, 0))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(nav_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=(20, 0))
    
    def show_step(self):
        """Display current step information"""
        if 0 <= self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            self.title_label.config(text=step['title'])
            self.desc_label.config(text=step['description'])
            self.progress_var.set(f"Step {self.current_step + 1} of {len(self.steps)}")
            
            # Update button states
            self.prev_btn.config(state=tk.NORMAL if self.current_step > 0 else tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL if self.current_step < len(self.steps) - 1 else tk.DISABLED)
            
            # Clear previous images
            self.clear_images()
    
    def clear_images(self):
        """Clear image display area"""
        for widget in self.image_frame.winfo_children():
            widget.destroy()
    
    def display_images(self, images_data):
        """Display images in a grid layout"""
        self.clear_images()
        
        if not images_data:
            return
        
        # Calculate grid layout
        n_images = len(images_data)
        if n_images <= 2:
            rows, cols = 1, n_images
        elif n_images <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
        
        # Create matplotlib figure
        fig = Figure(figsize=(12, 6), tight_layout=True)
        
        for i, (img, title) in enumerate(images_data):
            if i >= rows * cols:
                break
                
            ax = fig.add_subplot(rows, cols, i + 1)
            
            if img is not None:
                if len(img.shape) == 2:  # Grayscale
                    ax.imshow(img, cmap='gray')
                else:  # Color (convert BGR to RGB)
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.text(0.5, 0.5, 'Image not available', ha='center', va='center')
                
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, self.image_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def safe_read(self, path, flags=cv2.IMREAD_GRAYSCALE):
        """Read image safely with caching and assets mapping"""
        # Use assets mapping if available
        actual_path = self.assets.get(path, path)
        
        if actual_path in self.images:
            return self.images[actual_path]
            
        if not os.path.exists(actual_path):
            print(f"[WARN] File not found: {actual_path} (original: {path})")
            self.images[actual_path] = None
            return None
            
        img = cv2.imread(actual_path, flags)
        if img is None:
            print(f"[WARN] Could not read image: {actual_path} (original: {path})")
            
        self.images[actual_path] = img
        return img
    
    def ensure_uint8(self, x):
        """Clip to 0..255 and convert to uint8"""
        return np.clip(x, 0, 255).astype(np.uint8)
    
    def scale_for_display(self, x):
        """Scale to [0,1] for display"""
        x = x.astype(np.float32)
        mn, mx = x.min(), x.max()
        if mx > mn:
            x = (x - mn) / (mx - mn)
        else:
            x = np.zeros_like(x)
        return x
    
    def process_current_step(self):
        """Process current step"""
        self.status_var.set("Processing...")
        self.root.update()
        
        try:
            step = self.steps[self.current_step]
            step['function']()
            self.status_var.set("Complete")
        except Exception as e:
            messagebox.showerror("Error", f"Error processing step: {str(e)}")
            self.status_var.set("Error")
    
    def next_step(self):
        """Go to next step"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.show_step()
    
    def prev_step(self):
        """Go to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.show_step()
    
    # Step implementations
    def step_1_brighten(self):
        """Step 1: Image brightening"""
        I = self.safe_read('tire.tif')
        if I is not None:
            I2 = self.ensure_uint8(I.astype(np.int16) + 75)
            
            images_data = [
                (I, 'Original Image (tire.tif)'),
                (I2, 'Brightened Image (+75)')
            ]
            self.display_images(images_data)
            
            print(f"[Step 1] Original min/max: {I.min()}/{I.max()} | Adjusted min/max: {I2.min()}/{I2.max()}")
            print(f"Pixels at max (255): original={np.sum(I==255)}, adjusted={np.sum(I2==255)}")
    
    def step_2_blend(self):
        """Step 2: Image blending"""
        Ia = self.safe_read('rice.png')
        Ib = self.safe_read('cameraman.tif')
        
        if Ia is not None and Ib is not None:
            # Resize to same size if different
            if Ia.shape != Ib.shape:
                Ib = cv2.resize(Ib, (Ia.shape[1], Ia.shape[0]))
            
            Ic = self.ensure_uint8(Ia.astype(np.int16) + Ib.astype(np.int16))
            
            images_data = [
                (Ia, 'Rice.png'),
                (Ib, 'Cameraman.tif'),
                (Ic, 'Blended (Addition)')
            ]
            self.display_images(images_data)
        else:
            print("Cannot blend: images not available")
    
    def step_3_subtract(self):
        """Step 3: Image subtraction"""
        I = self.safe_read('cameraman.tif')
        J = self.safe_read('cameraman2.tif')
        
        if I is not None and J is not None:
            # Resize to same size if different
            if I.shape != J.shape:
                J = cv2.resize(J, (I.shape[1], I.shape[0]))
            
            # Subtraction
            diffim = I.astype(np.int16) - J.astype(np.int16)
            diffim_u8 = self.ensure_uint8(diffim)
            
            # Absolute difference
            diffim2 = cv2.absdiff(I, J)
            
            # Scaled version
            diffim_scaled = self.ensure_uint8(self.scale_for_display(diffim) * 255)
            
            images_data = [
                (I, 'Original (cameraman.tif)'),
                (J, 'Second Image (cameraman2.tif)'),
                (diffim2, 'Absolute Difference'),
                (diffim_scaled, 'Subtraction (Scaled)')
            ]
            self.display_images(images_data)
            
            # Print analysis information
            print(f"[Step 3] Image shapes - I: {I.shape}, J: {J.shape}")
            print(f"[Step 3] Difference stats - Min: {diffim.min()}, Max: {diffim.max()}")
            print(f"[Step 3] AbsDiff stats - Min: {diffim2.min()}, Max: {diffim2.max()}")
            print(f"[Step 3] Non-zero differences: {np.count_nonzero(diffim2)} pixels")
            print(f"[Step 3] Percentage different: {(np.count_nonzero(diffim2)/diffim2.size)*100:.2f}%")
    
    def step_4_scaling(self):
        """Step 4: Scaling vs brightening"""
        I = self.safe_read('moon.tif')
        if I is not None:
            I2 = self.ensure_uint8(I.astype(np.int16) + 50)  # brightening
            I3 = self.ensure_uint8(I.astype(np.float32) * 1.2)  # scaling
            
            images_data = [
                (I, 'Original Lindsay'),
                (I2, 'Brightening (+50)'),
                (I3, 'Scaling (*1.2)')
            ]
            self.display_images(images_data)
    
    def step_5_3d_effect(self):
        """Step 5: 3D effect with multiplication"""
        I = self.safe_read('earth1.tif')
        J = self.safe_read('earth2.tif')
        
        if I is not None and J is not None and I.shape == J.shape:
            I_f = I.astype(np.float32) / 255.0
            J_f = J.astype(np.float32) / 255.0
            K = I_f * J_f
            K_display = self.ensure_uint8(self.scale_for_display(K) * 255)
            
            images_data = [
                (I, 'Planet Image'),
                (J, 'Gradient'),
                (K_display, '3D Effect (Multiplication)')
            ]
            self.display_images(images_data)
    
    def step_6_division(self):
        """Step 6: Division vs multiplication"""
        I = self.safe_read('moon.tif')
        if I is not None:
            I_div = self.ensure_uint8(I.astype(np.float32) / 2.0)
            I_mul = self.ensure_uint8(I.astype(np.float32) * 0.5)
            
            images_data = [
                (I, 'Original Moon'),
                (I_div, 'Division (/2)'),
                (I_mul, 'Multiplication (*0.5)')
            ]
            self.display_images(images_data)
            
            print(f"Division and multiplication same result: {np.array_equal(I_div, I_mul)}")
    
    def step_7_background(self):
        """Step 7: Background subtraction"""
        notext = self.safe_read('gradient.tif')
        text = self.safe_read('gradient_with_text.tif')
        
        if notext is not None and text is not None and notext.shape == text.shape:
            # Threshold attempt
            _, BW = cv2.threshold(text, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Background division
            notext_f = notext.astype(np.float32)
            notext_f[notext_f == 0] = 1e-6  # Avoid division by zero
            fixed = text.astype(np.float32) / notext_f
            fixed_display = self.ensure_uint8(self.scale_for_display(fixed) * 255)
            
            images_data = [
                (text, 'Text with Background'),
                (BW, 'Otsu Threshold'),
                (notext, 'Background Only'),
                (fixed_display, 'Background Removed')
            ]
            self.display_images(images_data)
    
    def step_8_roi(self):
        """Step 8: ROI and logic operations with custom points"""
        I = self.safe_read('pout.tif')
        if I is not None:
            h, w = I.shape
            
            # ROI 1: Rectangular ROI (traditional)
            mask1 = np.zeros((h, w), dtype=np.uint8)
            center_x, center_y = w//2, h//2
            mask1[center_y-40:center_y+40, center_x-60:center_x+60] = 255
            I_masked1 = cv2.bitwise_and(I, mask1)
            
            # ROI 2: Custom polygon ROI with specific points
            mask2 = np.zeros((h, w), dtype=np.uint8)
            # Define custom points for polygon ROI
            points = np.array([
                [w//4, h//4],      # Top-left
                [3*w//4, h//4],    # Top-right  
                [3*w//4, 3*h//4],  # Bottom-right
                [w//2, 3*h//4],    # Bottom-center
                [w//4, h//2]       # Left-center
            ], dtype=np.int32)
            
            cv2.fillPoly(mask2, [points], 255)
            I_masked2 = cv2.bitwise_and(I, mask2)
            
            # ROI 3: Circle ROI with center point
            mask3 = np.zeros((h, w), dtype=np.uint8)
            circle_center = (w//3, 2*h//3)  # Custom center point
            radius = min(w, h) // 6
            cv2.circle(mask3, circle_center, radius, 255, -1)
            I_masked3 = cv2.bitwise_and(I, mask3)
            
            # ROI 4: Multiple point-based ROI
            mask4 = np.zeros((h, w), dtype=np.uint8)
            # Define multiple small circular ROIs at specific points
            roi_points = [
                (w//6, h//6), (w//2, h//6), (5*w//6, h//6),
                (w//6, h//2), (5*w//6, h//2),
                (w//6, 5*h//6), (w//2, 5*h//6), (5*w//6, 5*h//6)
            ]
            
            for point in roi_points:
                cv2.circle(mask4, point, 15, 255, -1)
            I_masked4 = cv2.bitwise_and(I, mask4)
            
            images_data = [
                (I, 'Original Image (pout.tif)'),
                (I_masked1, 'Rectangular ROI'),
                (I_masked2, 'Polygon ROI (Custom Points)'),
                (I_masked3, f'Circle ROI (Center: {circle_center})'),
                (I_masked4, 'Multi-Point ROI'),
                (mask2, 'Polygon Mask')
            ]
            self.display_images(images_data)
            
            # Print ROI information
            print(f"[ROI Info] Image size: {w}x{h}")
            print(f"[ROI Info] Polygon points: {points.tolist()}")
            print(f"[ROI Info] Circle center: {circle_center}, radius: {radius}")
            print(f"[ROI Info] Multi-point locations: {roi_points}")
    
    def step_9_xor(self):
        """Step 9: XOR operations"""
        A = self.safe_read('cameraman.tif')
        B = self.safe_read('cameraman2.tif')
        
        if A is not None and B is not None:
            # Resize to same size
            if A.shape != B.shape:
                B = cv2.resize(B, (A.shape[1], A.shape[0]))
            
            I_xor = cv2.bitwise_xor(A, B)
            I_xor_scaled = self.ensure_uint8(self.scale_for_display(I_xor) * 255)
            
            images_data = [
                (A, 'Image 1 (cameraman.tif)'),
                (B, 'Image 2 (cameraman2.tif)'),
                (I_xor_scaled, 'XOR Result (Scaled)')
            ]
            self.display_images(images_data)
    
    def step_10_custom_roi(self):
        """Step 10: Custom ROI dengan koordinat titik yang dapat disesuaikan"""
        I = self.safe_read('lindsay.tif')
        if I is not None:
            h, w = I.shape
            
            # ROI 1: Koordinat titik untuk memotong wajah (contoh)
            face_points = np.array([
                [w//3, h//4],         # Dahi kiri
                [2*w//3, h//4],       # Dahi kanan  
                [3*w//4, h//2],       # Pipi kanan
                [2*w//3, 3*h//4],     # Dagu kanan
                [w//3, 3*h//4],       # Dagu kiri
                [w//4, h//2]          # Pipi kiri
            ], dtype=np.int32)
            
            mask_face = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_face, [face_points], 255)
            I_face = cv2.bitwise_and(I, mask_face)
            
            # ROI 2: Koordinat titik untuk memotong area mata
            eye_left = np.array([
                [w//3 - 20, h//3],
                [w//3 + 20, h//3],
                [w//3 + 15, h//3 + 15],
                [w//3 - 15, h//3 + 15]
            ], dtype=np.int32)
            
            eye_right = np.array([
                [2*w//3 - 20, h//3],
                [2*w//3 + 20, h//3], 
                [2*w//3 + 15, h//3 + 15],
                [2*w//3 - 15, h//3 + 15]
            ], dtype=np.int32)
            
            mask_eyes = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_eyes, [eye_left, eye_right], 255)
            I_eyes = cv2.bitwise_and(I, mask_eyes)
            
            # ROI 3: Koordinat presisi untuk area mulut
            mouth_points = np.array([
                [w//2 - 25, 2*h//3],    # Kiri mulut
                [w//2 + 25, 2*h//3],    # Kanan mulut
                [w//2 + 20, 2*h//3 + 20], # Bawah kanan
                [w//2 - 20, 2*h//3 + 20]  # Bawah kiri
            ], dtype=np.int32)
            
            mask_mouth = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_mouth, [mouth_points], 255)
            I_mouth = cv2.bitwise_and(I, mask_mouth)
            
            images_data = [
                (I, 'Original Image (lindsay.tif)'),
                (I_face, 'Face ROI (Custom Points)'),
                (I_eyes, 'Eyes ROI (Precise Coordinates)'),
                (I_mouth, 'Mouth ROI (Point-based Cut)')
            ]
            self.display_images(images_data)
            
            # Print koordinat yang digunakan
            print(f"[Custom ROI] Face points: {face_points.tolist()}")
            print(f"[Custom ROI] Left eye: {eye_left.tolist()}")
            print(f"[Custom ROI] Right eye: {eye_right.tolist()}")
            print(f"[Custom ROI] Mouth: {mouth_points.tolist()}")
            print(f"[Custom ROI] Image dimensions: {w}x{h}")

def main():
    root = tk.Tk()
    app = TutorialGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
