import numpy as np
import cv2
import mahotas
from typing import Tuple
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from scipy.stats import skew

_HOG_PIX_PER_CELL = (8, 8)
_HOG_CELLS_PER_BLOCK = (2, 2)
_HOG_ORIENT = 9


def _ensure_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def feat_hog(img):
    g = _ensure_gray(img)
    f = hog(
        g,
        orientations=_HOG_ORIENT,
        pixels_per_cell=_HOG_PIX_PER_CELL,
        cells_per_block=_HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True
    )
    return f.astype(np.float32)


def feat_lbp(img, P=8, R=1):
    g = _ensure_gray(img)
    lbp = local_binary_pattern(g, P=P, R=R, method='uniform')
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), density=True)
    return hist.astype(np.float32)


def feat_gabor(img, freq=0.2, thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    g = _ensure_gray(img)
    feats = []
    for th in thetas:
        real, imag = gabor(g, frequency=freq, theta=th)
        feats.extend([real.mean(), real.var(), imag.mean(), imag.var()])
    return np.array(feats, dtype=np.float32)


def feat_color_hsv(img, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def feat_hu_moments(img):
    g = _ensure_gray(img)
    m = cv2.moments(g)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)  
    return hu.astype(np.float32)


def feat_haralick(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    g = _ensure_gray(img)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    glcm = graycomatrix(
        g, 
        distances=distances, 
        angles=angles, 
        levels=256, 
        symmetric=True, 
        normed=True
    )
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    feats = []
    for p in props:
        feats.extend(graycoprops(glcm, p).flatten())
    return np.array(feats, dtype=np.float32)


def feat_zernike(img, radius=50, degree=8):
    g = _ensure_gray(img)
    g = cv2.resize(g, (radius*2, radius*2)) 
    g = g.astype(np.float32)
    g = g / 255.0 
    feats = mahotas.features.zernike_moments(g, radius, degree)
    return np.array(feats, dtype=np.float32)



def feat_color_moments(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    feats = []
    for i in range(3):  
        channel = hsv[:, :, i].ravel()
        feats.extend([
            channel.mean(), 
            channel.std(), 
            skew(channel)
        ])
    return np.array(feats, dtype=np.float32)


def extract_all(img):
    return (
        feat_hog(img),
        feat_lbp(img),
        feat_gabor(img),
        feat_color_hsv(img),
        feat_hu_moments(img),
        feat_haralick(img),
        feat_zernike(img),
        feat_color_moments(img),
    )