r"""
pipeline.py
===========
Pipeline quản lý trích xuất đặc trưng và suy luận Machine Learning (Frozen Production Weights).
Quy tắc:
- Toàn bộ 36 mô hình đã được đóng băng tại weights/{model}_{feature}.joblib
- ZERO TRAINING trong quá trình chạy server/suy luận
- Nạp trọng số tức thì từ file .joblib
- Trích xuất đặc trưng chuẩn xác đồng nhất với feature_extraction_unsegmented.py
"""

import os
import time
import json
import joblib
import warnings
import cv2
import numpy as np
import pandas as pd

from app.features import extract_features_for_model, FEATURE_DIMS

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY_DIR = os.path.dirname(CURRENT_DIR)
WEIGHTS_DIR = os.path.join(DEPLOY_DIR, "weights")
METADATA_JSON = os.path.join(WEIGHTS_DIR, "models_metadata.json")


class CoffeeClassifierManager:
    """
    Quản lý trích xuất đặc trưng và suy luận mô hình cà phê đã đóng băng (Pure Frozen Deployment).
    Hỗ trợ 6 mô hình: LR, KNN, SVM, RF, ET, LGBM
    Hỗ trợ 6 nhóm đặc trưng: LBP, GLCM, ELA, Hist_HSV_64, Top_20_Hist_HSV_64, ELA + Hist_HSV_64
    """

    AVAILABLE_MODELS = {
        "LR": {"name": "Logistic Regression (LR)", "desc": "Multinomial Logistic Regression with Best Hyperparameters"},
        "KNN": {"name": "K-Nearest Neighbors (KNN)", "desc": "K-Nearest Neighbors with Paper Best Distance Metric"},
        "SVM": {"name": "Support Vector Machine (SVM)", "desc": "Support Vector Classifier with RBF Kernel & Probability"},
        "RF": {"name": "Random Forest (RF)", "desc": "Optimized Random Forest Ensemble"},
        "ET": {"name": "Extra Trees (ET)", "desc": "Extremely Randomized Trees Ensemble"},
        "LGBM": {"name": "Light Gradient Boosting Machine (LGBM)", "desc": "LightGBM Multi-class GBDT Engine"}
    }

    AVAILABLE_FEATURES = {
        "LBP": {"name": "Local Binary Patterns (LBP)", "dim": 59, "desc": "Texture descriptor (nri_uniform 59 bins)"},
        "GLCM": {"name": "Gray-Level Co-occurrence Matrix (GLCM)", "dim": 16, "desc": "Spatial gray-level matrix (4 directions x 4 properties)"},
        "ELA": {"name": "Error Level Analysis (ELA)", "dim": 48, "desc": "JPEG compression error analysis (48 statistical & histogram properties)"},
        "Hist_HSV_64": {"name": "Histogram HSV (HSV-CH)", "dim": 192, "desc": "HSV Color Histogram (64 bins x 3 channels)"},
        "Top_20_Hist_HSV_64": {"name": "Top 20% Histogram HSV", "dim": 39, "desc": "Top 39 salient HSV features via Feature Selection"},
        "ELA + Hist_HSV_64": {"name": "Fused ELA + Histogram HSV", "dim": 240, "desc": "Fused ELA (48) and HSV Histogram 64 (192)"}
    }

    CLASSES = ["Expresso", "Kenya", "Starbucks"]

    def __init__(self):
        self.model_cache = {}
        self.metadata = {}
        self.baseline_accuracies = {}
        self._load_metadata()

        for dep_k, dep_v in self.metadata.get("deployments", {}).items():
            self.baseline_accuracies[dep_k] = round(float(dep_v.get("paper_val_accuracy", 0.0)), 2)

        # Cache default pipeline
        self.get_or_load_pipeline("LGBM", "Hist_HSV_64")

    def _load_metadata(self):
        if os.path.exists(METADATA_JSON):
            try:
                with open(METADATA_JSON, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"[!] Warning reading metadata: {e}")

    def get_or_load_pipeline(self, model_name, feature_type):
        """Nạp mô hình production từ weights/ (Zero-Retraining)"""
        cache_key = f"{model_name}_{feature_type}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        weight_file = f"{model_name}_{feature_type}.joblib"
        weight_path = os.path.join(WEIGHTS_DIR, weight_file)

        if os.path.exists(weight_path):
            pipeline = joblib.load(weight_path)

            # Giải pháp 1: Khóa đơn luồng (n_jobs=1) để triệt tiêu độ trễ OpenMP thread wake-up & sync trên ARM
            try:
                if hasattr(pipeline, "named_steps"):
                    clf = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("model")
                    if clf is not None and hasattr(clf, "n_jobs"):
                        clf.set_params(n_jobs=1)
                elif hasattr(pipeline, "n_jobs"):
                    pipeline.set_params(n_jobs=1)
            except Exception as e:
                print(f"[!] Warning setting single-thread for {cache_key}: {e}")

            # Khởi động (Warm-up) mô hình ngay khi nạp để nạp trọng số vào CPU cache
            try:
                dim = self.AVAILABLE_FEATURES.get(feature_type, {}).get("dim", 192)
                dummy = np.zeros((1, dim), dtype=np.float32)
                if hasattr(pipeline, "predict_proba"):
                    _ = pipeline.predict_proba(dummy)
                else:
                    _ = pipeline.predict(dummy)
            except Exception:
                pass

            self.model_cache[cache_key] = pipeline
            return pipeline

        raise FileNotFoundError(f"Tập tin trọng số không tồn tại: {weight_path}. Hãy chạy freeze_models.py!")

    def extract_features_from_image(self, image_bgr, feature_type, model_name="LGBM"):
        """
        Trích xuất đặc trưng trực tiếp từ ảnh đầu vào theo feature_type sử dụng module app.features.
        """
        selected_cols = None
        if feature_type == "Top_20_Hist_HSV_64":
            dep_key = f"{model_name}_{feature_type}"
            if dep_key in self.metadata.get("deployments", {}):
                selected_cols = self.metadata["deployments"][dep_key].get("selected_cols")
            if not selected_cols or len(selected_cols) != 39:
                raise RuntimeError(
                    f"Không tìm thấy 39 cột đặc trưng tối ưu cho '{dep_key}' trong metadata. "
                    "Hãy kiểm tra file weights/models_metadata.json!"
                )

        return extract_features_for_model(image_bgr, feature_type, selected_cols=selected_cols)

    def predict(self, image_bgr, model_name="LGBM", feature_type="Hist_HSV_64"):
        if image_bgr is None or image_bgr.size == 0:
            return {"success": False, "error": "Ảnh không hợp lệ"}

        h, w = image_bgr.shape[:2]

        # Nạp mô hình đã freeze
        pipeline = self.get_or_load_pipeline(model_name, feature_type)

        # 1. Trích xuất đặc trưng
        t_feat_start = time.perf_counter()
        feat_vec, hsv_dict = self.extract_features_from_image(image_bgr, feature_type, model_name=model_name)
        feat_time = (time.perf_counter() - t_feat_start) * 1000.0

        # 2. Suy luận mô hình (Tối ưu hóa đơn luồng num_threads=1 & giảm thiểu overhead)
        t_infer_start = time.perf_counter()
        feat_input = feat_vec.reshape(1, -1)

        if hasattr(pipeline, "named_steps"):
            scaler = pipeline.named_steps.get("scaler")
            classifier = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("model")

            if scaler is not None:
                feat_scaled = (feat_input - scaler.mean_) / scaler.scale_
            else:
                feat_scaled = feat_input

            if hasattr(classifier, "predict_proba"):
                try:
                    probs = classifier.predict_proba(feat_scaled, num_threads=1)[0]
                except TypeError:
                    probs = classifier.predict_proba(feat_scaled)[0]
            else:
                try:
                    pred = classifier.predict(feat_scaled, num_threads=1)[0]
                except TypeError:
                    pred = classifier.predict(feat_scaled)[0]
                probs = np.zeros(3)
                probs[pred] = 1.0
        else:
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(feat_input)[0]
            else:
                pred = pipeline.predict(feat_input)[0]
                probs = np.zeros(3)
                probs[pred] = 1.0

        pred_idx = int(np.argmax(probs))
        infer_time = (time.perf_counter() - t_infer_start) * 1000.0

        total_time = feat_time + infer_time
        fps = 1000.0 / total_time if total_time > 0 else 0.0

        prob_dict = {cls_name: float(probs[i]) for i, cls_name in enumerate(self.CLASSES)}
        predicted_class = self.CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        # Lấy benchmark accuracy từ metadata
        dep_key = f"{model_name}_{feature_type}"
        expected_acc = "N/A"
        if dep_key in self.metadata.get("deployments", {}):
            val_acc = self.metadata["deployments"][dep_key].get("paper_val_accuracy")
            if val_acc is not None:
                expected_acc = f"{val_acc:.2f}%"

        return {
            "success": True,
            "is_batch": False,
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100.0, 1),
            "probabilities": {k: round(v * 100.0, 1) for k, v in prob_dict.items()},
            "model_name": model_name,
            "model_full_name": self.AVAILABLE_MODELS.get(model_name, {}).get("name", model_name),
            "feature_type": self.AVAILABLE_FEATURES.get(feature_type, {}).get("name", feature_type),
            "feature_key": feature_type,
            "feature_dim": len(feat_vec),
            "expected_accuracy": expected_acc,
            "latency": {
                "feature_extraction_ms": round(feat_time, 2),
                "inference_ms": round(infer_time, 2),
                "total_ms": round(total_time, 2),
                "fps": round(fps, 0)
            },
            "image_info": {
                "width": w,
                "height": h
            }
        }

    def predict_batch(self, images_bgr, model_name="LGBM", feature_type="Hist_HSV_64"):
        """
        Thực hiện phân loại hàng loạt (Batch Classification) cho danh sách N ảnh hạt cà phê.
        Tối ưu hóa xử lý ma trận N x D trong một lần suy luận (Vectorized Batch Inference).
        """
        if not images_bgr:
            return {"success": False, "error": "Danh sách ảnh trống"}

        # Nạp mô hình đã freeze
        pipeline = self.get_or_load_pipeline(model_name, feature_type)

        # 1. Trích xuất đặc trưng cho từng ảnh trong mẻ
        t_feat_start = time.perf_counter()
        feat_vecs = []
        for img in images_bgr:
            f_vec, _ = self.extract_features_from_image(img, feature_type, model_name=model_name)
            feat_vecs.append(f_vec)
        feat_time_total = (time.perf_counter() - t_feat_start) * 1000.0

        # 2. Suy luận toàn bộ ma trận đặc trưng N x D (Tối ưu hóa đơn luồng num_threads=1)
        t_infer_start = time.perf_counter()
        batch_matrix = np.array(feat_vecs, dtype=np.float32)

        if hasattr(pipeline, "named_steps"):
            scaler = pipeline.named_steps.get("scaler")
            classifier = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("model")

            if scaler is not None:
                batch_scaled = (batch_matrix - scaler.mean_) / scaler.scale_
            else:
                batch_scaled = batch_matrix

            if hasattr(classifier, "predict_proba"):
                try:
                    probs_matrix = classifier.predict_proba(batch_scaled, num_threads=1)
                except TypeError:
                    probs_matrix = classifier.predict_proba(batch_scaled)
            else:
                try:
                    preds = classifier.predict(batch_scaled, num_threads=1)
                except TypeError:
                    preds = classifier.predict(batch_scaled)
                probs_matrix = np.zeros((len(images_bgr), len(self.CLASSES)), dtype=np.float32)
                for i, p in enumerate(preds):
                    probs_matrix[i, p] = 1.0
        else:
            if hasattr(pipeline, "predict_proba"):
                probs_matrix = pipeline.predict_proba(batch_matrix)
            else:
                preds = pipeline.predict(batch_matrix)
                probs_matrix = np.zeros((len(images_bgr), len(self.CLASSES)), dtype=np.float32)
                for i, p in enumerate(preds):
                    probs_matrix[i, p] = 1.0

        infer_time_total = (time.perf_counter() - t_infer_start) * 1000.0

        n = len(images_bgr)
        total_time = feat_time_total + infer_time_total
        fps = (n * 1000.0) / total_time if total_time > 0 else 0.0

        avg_feat = feat_time_total / n
        avg_infer = infer_time_total / n
        avg_total = total_time / n

        # 3. Phân tích kết quả từng hạt và tổng hợp mẻ
        items = []
        class_counts = {c: 0 for c in self.CLASSES}

        for i in range(n):
            probs = probs_matrix[i]
            pred_idx = int(np.argmax(probs))
            pred_class = self.CLASSES[pred_idx]
            conf = float(probs[pred_idx])
            class_counts[pred_class] += 1

            prob_dict = {cls_name: round(float(probs[c]) * 100.0, 1) for c, cls_name in enumerate(self.CLASSES)}
            items.append({
                "index": i + 1,
                "predicted_class": pred_class,
                "confidence": round(conf * 100.0, 1),
                "probabilities": prob_dict
            })

        class_percentages = {c: round((class_counts[c] / n) * 100.0, 1) for c in self.CLASSES}
        dominant_class = max(class_counts, key=class_counts.get)

        dep_key = f"{model_name}_{feature_type}"
        expected_acc = "N/A"
        if dep_key in self.metadata.get("deployments", {}):
            val_acc = self.metadata["deployments"][dep_key].get("paper_val_accuracy")
            if val_acc is not None:
                expected_acc = f"{val_acc:.2f}%"

        return {
            "success": True,
            "is_batch": True,
            "total_images": n,
            "model_name": model_name,
            "model_full_name": self.AVAILABLE_MODELS.get(model_name, {}).get("name", model_name),
            "feature_type": self.AVAILABLE_FEATURES.get(feature_type, {}).get("name", feature_type),
            "feature_key": feature_type,
            "feature_dim": len(feat_vecs[0]),
            "expected_accuracy": expected_acc,
            "summary": {
                "class_counts": class_counts,
                "class_percentages": class_percentages,
                "dominant_class": dominant_class,
                "dominant_percentage": class_percentages[dominant_class]
            },
            "latency": {
                "feature_extraction_ms": round(feat_time_total, 2),
                "inference_ms": round(infer_time_total, 2),
                "total_ms": round(total_time, 2),
                "fps": round(fps, 0),
                "avg_feature_extraction_ms": round(avg_feat, 2),
                "avg_inference_ms": round(avg_infer, 2),
                "avg_total_ms": round(avg_total, 2)
            },
            "items": items
        }
