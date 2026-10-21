r"""
features.py
===========
Module trích xuất đặc trưng hình ảnh chuẩn hóa (Canonical Feature Extraction).
Đảm bảo 100% tính nhất quán toán học giữa Training, Benchmarking và Production API.

Các nhóm đặc trưng hỗ trợ:
1. LBP (59 chiều): Local Binary Patterns (nri_uniform, P=8, R=1, density=True)
2. GLCM (16 chiều): Gray-Level Co-occurrence Matrix (distance=3, angles=4, properties=4)
3. ELA (48 chiều): Error Level Analysis (JPEG Quality=90, 3 channels x [5 stats + 11 bins])
4. Hist_HSV_64 (192 chiều): Color Histogram HSV (64 bins x 3 channels, normalized)
5. Top_20_Hist_HSV_64 (39 chiều): 39 đặc trưng HSV nổi trội nhất qua Feature Selection
6. ELA + Hist_HSV_64 (240 chiều): Nối đặc trưng ELA (48) và Hist HSV (192)
"""

import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# Tên các cột tiêu chuẩn
HSV_CHANNELS = ["H", "S", "V"]
GLCM_PROPERTIES = ["contrast", "correlation", "energy", "homogeneity"]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

FEATURE_DIMS = {
    "LBP": 59,
    "GLCM": 16,
    "ELA": 48,
    "Hist_HSV_64": 192,
    "Top_20_Hist_HSV_64": 39,
    "ELA + Hist_HSV_64": 240
}

STANDARD_COLS = {
    "LBP": [f"LBP_{i}" for i in range(59)],
    "GLCM": [f"GLCM_{i}" for i in range(16)],
    "ELA": [f"ELA_{i}" for i in range(48)],
    "Hist_HSV_64": [f"Hist_{c}_64_{i}" for c in HSV_CHANNELS for i in range(64)],
    "ELA + Hist_HSV_64": [f"ELA_{i}" for i in range(48)] + [f"Hist_{c}_64_{i}" for c in HSV_CHANNELS for i in range(64)]
}


def extract_lbp(image_bgr):
    """Trích xuất 59 đặc trưng LBP (nri_uniform)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="nri_uniform")
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
    return hist_lbp.astype(np.float32)


def extract_glcm(image_bgr):
    """Trích xuất 16 đặc trưng GLCM (4 thuộc tính x 4 góc hướng)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[3], angles=GLCM_ANGLES, symmetric=True, normed=True)
    feats = []
    for prop in GLCM_PROPERTIES:
        feats.extend(graycoprops(glcm, prop).flatten())
    return np.array(feats, dtype=np.float32)


def extract_ela(image_bgr, quality=90):
    """Trích xuất 48 đặc trưng ELA (JPEG compression error analysis)."""
    _, enc = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    comp = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(image_bgr, comp)
    feats = []
    # Kênh B, G, R từ cv2.split
    for ch in cv2.split(ela):
        flat = ch.flatten()
        stats = [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.max(flat)),
            float(skew(flat)),
            float(kurtosis(flat))
        ]
        h_b, _ = np.histogram(flat, bins=11, range=(0, 11))
        h_norm = h_b.astype(np.float32) / (np.sum(h_b) + 1e-7)
        feats.extend(stats)
        feats.extend(h_norm.tolist())
    return np.array(feats, dtype=np.float32)


def extract_hsv_hist(image_bgr):
    """Trích xuất 192 đặc trưng HSV Histogram (64 bins x 3 kênh)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hist_h = cv2.calcHist([h], [0], None, [64], [0, 256]).flatten()
    cv2.normalize(hist_h, hist_h)
    hist_s = cv2.calcHist([s], [0], None, [64], [0, 256]).flatten()
    cv2.normalize(hist_s, hist_s)
    hist_v = cv2.calcHist([v], [0], None, [64], [0, 256]).flatten()
    cv2.normalize(hist_v, hist_v)
    feat_vec = np.concatenate([hist_h, hist_s, hist_v], axis=0).astype(np.float32)
    hsv_dict = {"H": hist_h.tolist(), "S": hist_s.tolist(), "V": hist_v.tolist()}
    return feat_vec, hsv_dict


def extract_feature_dict(image_bgr, feature_type):
    """
    Trích xuất từ điển {tên_cột: giá_trị} từ ảnh gốc,
    đảm bảo tên cột khớp 100% với tập dữ liệu huấn luyện.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid or empty image provided")

    if feature_type in ["Hist_HSV_64", "Top_20_Hist_HSV_64"]:
        vec, _ = extract_hsv_hist(image_bgr)
        cols = STANDARD_COLS["Hist_HSV_64"]
        return {cols[i]: float(vec[i]) for i in range(len(cols))}

    elif feature_type == "LBP":
        vec = extract_lbp(image_bgr)
        cols = STANDARD_COLS["LBP"]
        return {cols[i]: float(vec[i]) for i in range(len(cols))}

    elif feature_type == "GLCM":
        vec = extract_glcm(image_bgr)
        cols = STANDARD_COLS["GLCM"]
        return {cols[i]: float(vec[i]) for i in range(len(cols))}

    elif feature_type == "ELA":
        vec = extract_ela(image_bgr)
        cols = STANDARD_COLS["ELA"]
        return {cols[i]: float(vec[i]) for i in range(len(cols))}

    elif feature_type == "ELA + Hist_HSV_64":
        d_ela = extract_feature_dict(image_bgr, "ELA")
        d_hsv = extract_feature_dict(image_bgr, "Hist_HSV_64")
        d_ela.update(d_hsv)
        return d_ela

    raise ValueError(f"Unknown feature type: {feature_type}")


def extract_features_for_model(image_bgr, feature_type, selected_cols=None):
    """
    Trích xuất vector 1D (numpy array) sẵn sàng cho pipeline.predict(feat.reshape(1, -1)).
    Hỗ trợ trả về cả vector và hsv_dict để phục vụ hiển thị Web UI nếu cần.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid or empty image provided")

    hsv_dict = {}

    if feature_type == "Hist_HSV_64":
        vec, hsv_dict = extract_hsv_hist(image_bgr)
        return vec, hsv_dict

    elif feature_type == "Top_20_Hist_HSV_64":
        vec, hsv_dict = extract_hsv_hist(image_bgr)
        if selected_cols and len(selected_cols) == 39:
            all_cols = STANDARD_COLS["Hist_HSV_64"]
            idx_map = {name: i for i, name in enumerate(all_cols)}
            indices = [idx_map[c] for c in selected_cols if c in idx_map]
            if len(indices) == 39:
                return vec[indices].astype(np.float32), hsv_dict
        raise ValueError("Top_20_Hist_HSV_64 requires exactly 39 valid 'selected_cols' matching the trained fold/model!")

    elif feature_type == "LBP":
        return extract_lbp(image_bgr), hsv_dict

    elif feature_type == "GLCM":
        return extract_glcm(image_bgr), hsv_dict

    elif feature_type == "ELA":
        return extract_ela(image_bgr), hsv_dict

    elif feature_type == "ELA + Hist_HSV_64":
        ela_vec = extract_ela(image_bgr)
        hsv_vec, hsv_dict = extract_hsv_hist(image_bgr)
        fused = np.concatenate([ela_vec, hsv_vec], axis=0).astype(np.float32)
        return fused, hsv_dict

    raise ValueError(f"Unknown feature type: {feature_type}")
