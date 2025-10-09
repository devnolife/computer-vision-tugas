from flask import Flask, render_template
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)

class CircleDetectionExplainer:
    def __init__(self, image_path='coloredChips.png'):
        self.image_path = image_path
        
    def create_plot(self, image, title, cmap=None):
        """Membuat plot dan mengkonversi ke base64"""
        plt.figure(figsize=(8, 6))
        if cmap:
            plt.imshow(image, cmap=cmap)
        else:
            plt.imshow(image)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Konversi ke base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100, facecolor='white')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
    
    def detect_circles(self, image, min_radius=20, max_radius=25, param1=50, param2=30, min_dist=30):
        """Deteksi lingkaran menggunakan HoughCircles"""
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
    
    def draw_circles(self, image, circles, color=(255, 0, 0), thickness=2):
        """Menggambar lingkaran yang terdeteksi"""
        output = image.copy()
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for circle in circles[0]:
                center = (circle[0], circle[1])
                radius = circle[2]
                
                cv2.circle(output, center, radius, color, thickness)
                cv2.circle(output, center, 2, color, 3)
        
        return output
    
    def run_complete_analysis(self):
        """Menjalankan analisis lengkap dan menghasilkan semua hasil"""
        steps = []
        
        # Step 1: Load Image
        img = cv2.imread(self.image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img1_base64 = self.create_plot(img_rgb, 'Gambar Asli')
        
        steps.append({
            'step': 1,
            'title': 'Memuat dan Konversi Gambar',
            'description': 'Langkah pertama adalah memuat gambar dari file dan mengkonversi format warna dari BGR ke RGB.',
            'code': '''# Memuat gambar dari file
img = cv2.imread('coloredChips.png')

# Konversi dari BGR (Blue-Green-Red) ke RGB (Red-Green-Blue)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)''',
            'explanation': 'OpenCV secara default membaca gambar dalam format BGR, sedangkan matplotlib dan kebanyakan library lain menggunakan RGB. Konversi ini penting agar warna tampil dengan benar.',
            'why': 'Tanpa konversi ini, warna merah akan tampak biru dan sebaliknya saat ditampilkan.',
            'image': img1_base64,
            'key_points': [
                'cv2.imread() membaca gambar dalam format BGR',
                'cv2.cvtColor() mengkonversi format warna',
                'RGB diperlukan untuk tampilan yang benar'
            ]
        })
        
        # Step 2: Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2_base64 = self.create_plot(gray, 'Gambar Grayscale', cmap='gray')
        
        steps.append({
            'step': 2,
            'title': 'Konversi ke Grayscale',
            'description': 'Mengubah gambar berwarna menjadi gambar abu-abu (grayscale) untuk mempermudah deteksi bentuk.',
            'code': '''# Konversi ke grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)''',
            'explanation': 'Algoritma HoughCircles bekerja pada gambar grayscale. Grayscale menghilangkan informasi warna dan fokus pada intensitas cahaya, sehingga bentuk lingkaran lebih mudah dideteksi.',
            'why': 'Dengan grayscale, algoritma dapat fokus pada kontras dan tepi objek tanpa terganggu variasi warna.',
            'image': img2_base64,
            'key_points': [
                'HoughCircles membutuhkan input grayscale',
                'Mengurangi kompleksitas dari 3 channel ke 1 channel',
                'Fokus pada intensitas, bukan warna'
            ]
        })
        
        # Step 3: Detect Dark Circles
        circles_dark = self.detect_circles(gray, param2=20, min_dist=25)
        count_dark = 0
        if circles_dark is not None:
            img_dark = self.draw_circles(img_rgb, circles_dark, color=(255, 0, 0))
            count_dark = len(circles_dark[0])
        else:
            img_dark = img_rgb.copy()
        img3_base64 = self.create_plot(img_dark, f'Deteksi Lingkaran Gelap - {count_dark} lingkaran')
        
        steps.append({
            'step': 3,
            'title': 'Deteksi Lingkaran Gelap',
            'description': f'Mendeteksi chip berwarna gelap (merah, biru, hijau, orange) menggunakan HoughCircles. Terdeteksi: {count_dark} lingkaran.',
            'code': '''# Deteksi lingkaran dengan HoughCircles
circles = cv2.HoughCircles(
    gray,                    # Input gambar grayscale
    cv2.HOUGH_GRADIENT,     # Metode deteksi
    dp=1,                   # Resolusi akumulator
    minDist=25,             # Jarak minimum antar lingkaran
    param1=50,              # Threshold untuk Canny edge
    param2=20,              # Threshold akumulator (sensitivitas)
    minRadius=20,           # Radius minimum
    maxRadius=25            # Radius maksimum
)''',
            'explanation': 'HoughCircles menggunakan transformasi Hough untuk mendeteksi lingkaran. Parameter param2=20 memberikan sensitivitas tinggi untuk mendeteksi chip gelap.',
            'why': 'Chip berwarna gelap memiliki kontras yang baik dengan latar belakang terang, sehingga mudah dideteksi dengan parameter standar.',
            'image': img3_base64,
            'count': count_dark,
            'key_points': [
                'param2 rendah = sensitivitas tinggi',
                'minDist mencegah deteksi ganda',
                'minRadius/maxRadius sesuai ukuran chip'
            ]
        })
        
        # Step 4: Detect Bright Circles
        gray_inv = cv2.bitwise_not(gray)
        circles_bright = self.detect_circles(gray_inv, param2=20, param1=30, min_dist=25)
        count_bright = 0
        if circles_bright is not None:
            img_bright = self.draw_circles(img_rgb, circles_bright, color=(0, 0, 255))
            count_bright = len(circles_bright[0])
        else:
            img_bright = img_rgb.copy()
        img4_base64 = self.create_plot(img_bright, f'Deteksi Lingkaran Terang - {count_bright} lingkaran')
        
        steps.append({
            'step': 4,
            'title': 'Deteksi Lingkaran Terang',
            'description': f'Mendeteksi chip kuning (terang) dengan teknik inversi gambar. Terdeteksi: {count_bright} lingkaran.',
            'code': '''# Inversi gambar untuk deteksi objek terang
gray_inv = cv2.bitwise_not(gray)

# Deteksi lingkaran pada gambar yang diinversi
circles_bright = cv2.HoughCircles(
    gray_inv,               # Input gambar yang diinversi
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=25,
    param1=30,              # Threshold lebih rendah
    param2=20,
    minRadius=20,
    maxRadius=25
)''',
            'explanation': 'Chip kuning sulit dideteksi pada gambar asli karena intensitasnya mirip dengan latar belakang. Dengan inversi (bitwise_not), chip terang menjadi gelap dan mudah dideteksi.',
            'why': 'Inversi mengubah pixel terang menjadi gelap, sehingga chip kuning yang tadinya terang kini memiliki kontras tinggi dengan latar belakang.',
            'image': img4_base64,
            'count': count_bright,
            'key_points': [
                'bitwise_not() membalik intensitas pixel',
                'Chip terang menjadi gelap setelah inversi',
                'Kontras tinggi memudahkan deteksi'
            ]
        })
        
        # Step 5: Final Result
        result = img_rgb.copy()
        if circles_dark is not None:
            result = self.draw_circles(result, circles_dark, color=(255, 0, 0))
        if circles_bright is not None:
            result = self.draw_circles(result, circles_bright, color=(0, 0, 255))
        
        total_count = count_dark + count_bright
        img5_base64 = self.create_plot(result, f'Hasil Akhir - Total {total_count} lingkaran')
        
        steps.append({
            'step': 5,
            'title': 'Hasil Akhir',
            'description': f'Penggabungan hasil deteksi lingkaran gelap dan terang. Total terdeteksi: {total_count} lingkaran.',
            'code': '''# Gambar hasil akhir
result = img_rgb.copy()

# Gambar lingkaran gelap dengan warna merah
if circles_dark is not None:
    result = draw_circles(result, circles_dark, (255, 0, 0))

# Gambar lingkaran terang dengan warna biru  
if circles_bright is not None:
    result = draw_circles(result, circles_bright, (0, 0, 255))''',
            'explanation': 'Menggabungkan hasil deteksi dari kedua metode untuk mendapatkan deteksi yang komprehensif terhadap semua jenis chip.',
            'why': 'Kombinasi dua metode ini memungkinkan deteksi chip dengan berbagai tingkat kecerahan, dari yang gelap hingga yang sangat terang.',
            'image': img5_base64,
            'count_dark': count_dark,
            'count_bright': count_bright,
            'total': total_count,
            'key_points': [
                f'Lingkaran merah: {count_dark} chip gelap/berwarna',
                f'Lingkaran biru: {count_bright} chip kuning/terang',
                f'Total akurasi deteksi: {total_count} chip'
            ]
        })
        
        return steps
    
    def detect_circles(self, image, min_radius=20, max_radius=25, param1=50, param2=30, min_dist=30):
        """Deteksi lingkaran menggunakan HoughCircles"""
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
    
    def draw_circles(self, image, circles, color=(255, 0, 0), thickness=2):
        """Menggambar lingkaran yang terdeteksi"""
        output = image.copy()
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for circle in circles[0]:
                center = (circle[0], circle[1])
                radius = circle[2]
                
                cv2.circle(output, center, radius, color, thickness)
                cv2.circle(output, center, 2, color, 3)
        
        return output
    
    def detect_dark_circles(self):
        """Deteksi lingkaran gelap (chip berwarna gelap)"""
        circles = self.detect_circles(self.gray, param2=20, min_dist=25)
        
        if circles is not None:
            img_with_circles = self.draw_circles(self.img_rgb, circles, color=(255, 0, 0))
            count = len(circles[0])
            self.results['dark_circles'] = count
            return self.create_plot(img_with_circles, f'Dark Circles Detected: {count}'), count
        else:
            self.results['dark_circles'] = 0
            return self.create_plot(self.img_rgb, 'No Dark Circles Detected'), 0
    
    def detect_bright_circles(self):
        """Deteksi lingkaran terang (chip kuning)"""
        gray_inv = cv2.bitwise_not(self.gray)
        circles = self.detect_circles(gray_inv, param2=20, param1=30, min_dist=25)
        
        if circles is not None:
            img_with_circles = self.draw_circles(self.img_rgb, circles, color=(0, 0, 255))
            count = len(circles[0])
            self.results['bright_circles'] = count
            return self.create_plot(img_with_circles, f'Bright Circles Detected: {count}'), count
        else:
            self.results['bright_circles'] = 0
            return self.create_plot(self.img_rgb, 'No Bright Circles Detected'), 0
    
    def detect_all_circles(self):
        """Deteksi semua lingkaran (gelap dan terang)"""
        # Deteksi lingkaran gelap
        circles_dark = self.detect_circles(self.gray, param2=20, min_dist=25)
        
        # Deteksi lingkaran terang
        gray_inv = cv2.bitwise_not(self.gray)
        circles_bright = self.detect_circles(gray_inv, param2=20, param1=30, min_dist=25)
        
        result = self.img_rgb.copy()
        n_dark = 0
        n_bright = 0
        
        if circles_dark is not None:
            result = self.draw_circles(result, circles_dark, color=(255, 0, 0))
            n_dark = len(circles_dark[0])
        
        if circles_bright is not None:
            result = self.draw_circles(result, circles_bright, color=(0, 0, 255))
            n_bright = len(circles_bright[0])
        
        total = n_dark + n_bright
        self.results['total'] = total
        self.results['dark_final'] = n_dark
        self.results['bright_final'] = n_bright
        
        return self.create_plot(result, f'All Circles (Red: Dark={n_dark}, Blue: Bright={n_bright}, Total={total})'), total, n_dark, n_bright
    
    def create_plot(self, image, title, cmap=None):
        """Membuat plot dan mengkonversi ke base64"""
        plt.figure(figsize=(10, 8))
        if cmap:
            plt.imshow(image, cmap=cmap)
        else:
            plt.imshow(image)
        plt.title(title)
        plt.axis('off')
        
        # Konversi ke base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64

# Global detector instance
detector = CircleDetectionExplainer()

@app.route('/')
def index():
    # Jalankan analisis lengkap
    steps = detector.run_complete_analysis()
    return render_template('index.html', steps=steps)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
