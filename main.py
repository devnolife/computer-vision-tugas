import cv2, numpy as np
from skimage.draw import polygon2mask

# Helper
def save_img(name, img):
    cv2.imwrite(name, img)
    print(f"Saved: {name}")

def scale_01(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)

# === STEP 1: Brightening tire.tif ===
I = cv2.imread('tire.tif', 0)
I2 = np.clip(I.astype(np.int16) + 75, 0, 255).astype(np.uint8)
save_img("Gambar1_Tire_Original.png", I)
save_img("Gambar2_Tire_Brightened.png", I2)

# === STEP 2: Blending rice + cameraman ===
Ia = cv2.imread('rice.png', 0)
Ib = cv2.imread('cameraman.tif', 0)
if Ia.shape != Ib.shape:
    Ib = cv2.resize(Ib, (Ia.shape[1], Ia.shape[0]))
Ic = np.clip(Ia.astype(np.int16) + Ib.astype(np.int16), 0, 255).astype(np.uint8)
save_img("Gambar3_Rice.png", Ia)
save_img("Gambar4_Cameraman.png", Ib)
save_img("Gambar5_Blend.png", Ic)

# === STEP 3: Subtraction & AbsDiff ===
I = cv2.imread('cameraman.tif', 0)
J = cv2.imread('cameraman2.tif', 0)
diff  = I.astype(np.int16) - J.astype(np.int16)
absd  = cv2.absdiff(I, J)
diff_scaled = (scale_01(diff)*255).astype(np.uint8)
absd_scaled = (scale_01(absd)*255).astype(np.uint8)
save_img("Gambar6_Cameraman1.png", I)
save_img("Gambar7_Cameraman2.png", J)
save_img("Gambar8_Subtraction.png", np.clip(diff,0,255).astype(np.uint8))
save_img("Gambar9_AbsDiff.png", absd)
save_img("Gambar10_Subtraction_Scaled.png", diff_scaled)
save_img("Gambar11_AbsDiff_Scaled.png", absd_scaled)

# === STEP 4: Dynamic scaling vs brightening (moon) ===
I = cv2.imread('moon.tif', 0)
I_add = np.clip(I.astype(np.int16)+50,0,255).astype(np.uint8)
I_mul = np.clip(I.astype(np.float32)*1.2,0,255).astype(np.uint8)
save_img("Gambar12_Moon_Original.png", I)
save_img("Gambar13_Moon_Brightened.png", I_add)
save_img("Gambar14_Moon_DynamicScaling.png", I_mul)

# === STEP 5: 3D effect (earth1 × earth2) ===
I = cv2.imread('earth1.tif', 0).astype(np.float32)/255
J = cv2.imread('earth2.tif', 0).astype(np.float32)/255
K = I*J
save_img("Gambar15_Earth1.png", (I*255).astype(np.uint8))
save_img("Gambar16_Earth2.png", (J*255).astype(np.uint8))
save_img("Gambar17_3DPlanet.png", (scale_01(K)*255).astype(np.uint8))

# === STEP 6: Division vs Multiplication (moon) ===
I = cv2.imread('moon.tif', 0).astype(np.float32)
Idiv = np.clip(I/2.0,0,255).astype(np.uint8)
Imul = np.clip(I*0.5,0,255).astype(np.uint8)
save_img("Gambar18_Moon_Division.png", Idiv)
save_img("Gambar19_Moon_Multiplication.png", Imul)

# === STEP 7: Background extraction ===
text = cv2.imread('gradient_with_text.tif', 0).astype(np.float32)
notext = cv2.imread('gradient.tif', 0).astype(np.float32)
_, BW = cv2.threshold(text.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
notext[notext==0]=1e-6
fixed = text/notext
fixed_disp = (scale_01(fixed)*255).astype(np.uint8)
save_img("Gambar20_TextWithBackground.png", text.astype(np.uint8))
save_img("Gambar21_BackgroundOnly.png", notext.astype(np.uint8))
save_img("Gambar22_OtsuThreshold.png", BW)
save_img("Gambar23_BackgroundRemoved.png", fixed_disp)

# === STEP 8: ROI (pout) ===
I = cv2.imread('pout.tif',0)
h,w = I.shape
poly = np.array([[60,60],[200,80],[220,200],[80,220]])
mask_bool = polygon2mask((h,w), np.fliplr(poly))
bw255 = (mask_bool.astype(np.uint8)*255)
AND_img = cv2.bitwise_and(I, bw255)
bw_cmp = cv2.bitwise_not(bw255)
OR_img = cv2.bitwise_or(I, bw_cmp)
save_img("Gambar24_Pout.png", I)
save_img("Gambar25_ROIMask.png", bw255)
save_img("Gambar26_AND.png", AND_img)
save_img("Gambar27_OR.png", OR_img)

# === STEP 9: XOR (cameraman) ===
A = cv2.imread('cameraman.tif',0)
B = cv2.imread('cameraman2.tif',0)
XOR = cv2.bitwise_xor(A,B)
save_img("Gambar28_Cameraman1.png", A)
save_img("Gambar29_Cameraman2.png", B)
save_img("Gambar30_XOR.png", XOR)

# === STEP 10: Custom ROI (lindsay) ===
I = cv2.imread('lindsay.tif',0)
h,w = I.shape
pts = np.array([[120,60],[200,60],[210,140],[110,150]],np.int32)
mask = np.zeros((h,w),np.uint8)
cv2.fillPoly(mask,[pts],255)
ROI = cv2.bitwise_and(I,mask)
save_img("Gambar31_Lindsay.png", I)
save_img("Gambar32_CustomROI.png", mask)
save_img("Gambar33_ROIResult.png", ROI)

# === STEP 11–13: ROI darker vs original ===
I = cv2.imread('lindsay.tif',0)
I_adj = np.clip(I.astype(np.float32)/1.5,0,255).astype(np.uint8)
poly = np.array([[100,50],[200,60],[180,150],[120,160]])
bw_bool = polygon2mask(I.shape, np.fliplr(poly))
new_img = np.where(bw_bool, I_adj, I).astype(np.uint8)
new_img_q11 = np.where(bw_bool, I, I_adj).astype(np.uint8)
save_img("Gambar34_Lindsay_Original.png", I)
save_img("Gambar35_Lindsay_DarkerROI.png", new_img)
save_img("Gambar36_Lindsay_Q11.png", new_img_q11)
