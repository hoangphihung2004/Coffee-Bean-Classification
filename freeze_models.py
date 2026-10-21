r"""
freeze_models.py
================
Tập lệnh đóng gói và đóng băng toàn bộ 36 mô hình Machine Learning (6 models x 6 feature sets)
dựa trên đúng siêu tham số tối ưu (Best_Params) và đặc trưng chọn lọc từ:
  - D:\New_Ideas\Coffee\data\results.csv
  - D:\New_Ideas\Coffee\data\results_fs_draft.csv

QUY TẮC BẢO TOÀN (ZERO-RETRAINING DEPLOYMENT):
1. Không chạy lại Optuna hay tìm kiếm tham số.
2. Trích xuất chính xác siêu tham số tối ưu của từng mô hình và đặc trưng.
3. Đóng gói thành:
   - Mô hình suy luận đơn (Production Pipeline fit trên toàn bộ dữ liệu): weights/{model}_{feature}.joblib
   - Gói 10-Fold CV Pipelines để phục vụ evaluate_preservation.py (đánh giá đối chiếu không cần train lại): weights/{model}_{feature}_cv_bundle.joblib
   - Metadata tổng hợp (kết quả bài báo, số chiều đặc trưng, danh sách cột): weights/models_metadata.json
"""

import os
import ast
import json
import time
import copy
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
COFFEE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(COFFEE_DIR, "data")
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")

os.makedirs(WEIGHTS_DIR, exist_ok=True)

FEATURES_CSV = os.path.join(DATA_DIR, "extracted_features_unsegmented.csv")
RESULTS_CSV = os.path.join(DATA_DIR, "results.csv")
RESULTS_FS_CSV = os.path.join(DATA_DIR, "results_fs_draft.csv")

MODEL_CLS_MAP = {
    "LR": LogisticRegression,
    "KNN": KNeighborsClassifier,
    "SVM": SVC,
    "RF": RandomForestClassifier,
    "ET": ExtraTreesClassifier,
    "EXTree": ExtraTreesClassifier,
    "LGBM": LGBMClassifier
}

FEATURE_CONFIGS = {
    "LBP": {
        "name": "Local Binary Patterns (LBP)",
        "dim": 59,
        "csv_source": "results.csv",
        "csv_feat_name": "LBP",
        "cols": [f"LBP_{i}" for i in range(59)]
    },
    "GLCM": {
        "name": "Gray-Level Co-occurrence Matrix (GLCM)",
        "dim": 16,
        "csv_source": "results.csv",
        "csv_feat_name": "GLCM",
        "cols": [f"GLCM_{i}" for i in range(16)]
    },
    "ELA": {
        "name": "Error Level Analysis (ELA)",
        "dim": 48,
        "csv_source": "results.csv",
        "csv_feat_name": "ELA",
        "cols": [f"ELA_{i}" for i in range(48)]
    },
    "Hist_HSV_64": {
        "name": "Color Histogram HSV (HSV-CH)",
        "dim": 192,
        "csv_source": "results.csv",
        "csv_feat_name": "Hist_HSV_64",
        "cols": [f"Hist_{c}_64_{i}" for c in ["H", "S", "V"] for i in range(64)]
    },
    "Top_20_Hist_HSV_64": {
        "name": "Top 20% Color Histogram HSV (Feature Selection)",
        "dim": 39,
        "csv_source": "results_fs_draft.csv",
        "csv_feat_name": "Hist_HSV_64",
        "fs_percent": 20,
        "cols": None # loaded per model from Selected_Features_Per_Fold
    },
    "ELA + Hist_HSV_64": {
        "name": "Fused ELA + Histogram HSV",
        "dim": 240,
        "csv_source": "results.csv",
        "csv_feat_name": "ELA + Hist_HSV_64",
        "cols": [f"ELA_{i}" for i in range(48)] + [f"Hist_{c}_64_{i}" for c in ["H", "S", "V"] for i in range(64)]
    }
}

def get_model(cls, p, num_class=3):
    params = copy.deepcopy(p)
    if cls == LGBMClassifier:
        params["objective"] = "multiclass"
        params["force_col_wise"] = True
        params["verbose"] = -1
        params["random_state"] = 42
        params["num_class"] = num_class
    elif cls == ExtraTreesClassifier:
        params["verbose"] = 0
        params["random_state"] = 42
    elif cls == RandomForestClassifier:
        params["random_state"] = 42
    elif cls == LogisticRegression:
        params["random_state"] = 42
    elif cls == SVC:
        params["random_state"] = 42
        params["probability"] = True
    elif cls == KNeighborsClassifier:
        pass
    return cls(**params)

def main():
    print("=" * 75)
    print("  COFFEE BEAN EDGE AI - FREEZING PRE-TRAINED MODELS & WEIGHTS")
    print("=" * 75)
    
    start_time = time.time()
    
    # 1. Load dataset
    print(f"[*] Loading feature dataset from: {FEATURES_CSV}")
    df_data = pd.read_csv(FEATURES_CSV)
    label_mapping = {"Expresso": 0, "Kenya": 1, "Starbucks": 2}
    y_all = df_data["label"].map(label_mapping).values
    print(f"    Total samples: {len(df_data)} across {len(label_mapping)} classes.")
    
    # 2. Load results CSVs
    print(f"[*] Reading baseline parameters from:")
    print(f"    - {RESULTS_CSV}")
    print(f"    - {RESULTS_FS_CSV}")
    df_results = pd.read_csv(RESULTS_CSV)
    df_results_fs = pd.read_csv(RESULTS_FS_CSV)
    
    models = ["LR", "KNN", "SVM", "RF", "ET", "LGBM"]
    metadata = {
        "models": {
            "LR": "Logistic Regression (LR)",
            "KNN": "K-Nearest Neighbors (KNN)",
            "SVM": "Support Vector Machine (SVM)",
            "RF": "Random Forest (RF)",
            "ET": "Extra Trees (ET)",
            "LGBM": "Light Gradient Boosting Machine (LGBM)"
        },
        "classes": ["Expresso", "Kenya", "Starbucks"],
        "features": {},
        "deployments": {}
    }
    
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    splits = list(outer_cv.split(df_data, y_all))
    
    total_saved = 0
    
    for feat_key, feat_cfg in FEATURE_CONFIGS.items():
        print(f"\n[-] Processing feature: {feat_key} ({feat_cfg['name']}, Dim={feat_cfg['dim']})")
        metadata["features"][feat_key] = {
            "name": feat_cfg["name"],
            "dim": feat_cfg["dim"]
        }
        
        for m_key in models:
            m_lookup = "EXTree" if m_key == "ET" else m_key
            model_cls = MODEL_CLS_MAP[m_key]
            
            # Retrieve hyperparameters and results from respective CSV
            if feat_cfg["csv_source"] == "results_fs_draft.csv":
                sub_df = df_results_fs[
                    (df_results_fs["Model"] == m_lookup) & 
                    (df_results_fs["Feature Name"] == feat_cfg["csv_feat_name"]) &
                    (df_results_fs["FS_Percent"] == feat_cfg["fs_percent"])
                ]
            else:
                sub_df = df_results[
                    (df_results["Model"] == m_lookup) & 
                    (df_results["Feature Name"] == feat_cfg["csv_feat_name"])
                ]
                
            if sub_df.empty:
                print(f"    [!] Warning: No entry found for {m_key} + {feat_key}")
                continue
                
            row = sub_df.iloc[0]
            best_params_list = ast.literal_eval(row["Best_Params"])
            val_acc_list = ast.literal_eval(row["Validation_Fold_Accuracy"])
            
            # Selected features
            if feat_cfg["cols"] is not None:
                feat_cols = feat_cfg["cols"]
                cols_per_fold = [feat_cols] * 10
            else:
                cols_per_fold = ast.literal_eval(row["Selected_Features_Per_Fold"])
                
            # Best fold index based on highest validation accuracy (standard Nested CV practice)
            best_fold_idx = int(np.argmax(val_acc_list))
            prod_params = best_params_list[best_fold_idx]
            prod_cols = cols_per_fold[best_fold_idx]
            
            # --- 1. FIT & SAVE PRODUCTION MODEL (on full dataset) ---
            X_full = df_data[prod_cols].values
            prod_clf = get_model(model_cls, prod_params)
            prod_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", prod_clf)
            ])
            prod_pipeline.fit(X_full, y_all)
            
            prod_save_name = f"{m_key}_{feat_key}.joblib"
            prod_save_path = os.path.join(WEIGHTS_DIR, prod_save_name)
            joblib.dump(prod_pipeline, prod_save_path, compress=3)
            
            # --- 2. FIT & SAVE 10-FOLD CV BUNDLE (for instant evaluate_preservation.py without retraining) ---
            fold_pipelines = []
            for fold_i, (train_ix, test_ix) in enumerate(splits):
                f_cols = cols_per_fold[fold_i]
                X_tr = df_data.iloc[train_ix][f_cols].values
                y_tr = y_all[train_ix]
                
                f_params = best_params_list[fold_i]
                f_clf = get_model(model_cls, f_params)
                f_pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", f_clf)
                ])
                f_pipeline.fit(X_tr, y_tr)
                fold_pipelines.append({
                    "fold": fold_i,
                    "pipeline": f_pipeline,
                    "cols": f_cols,
                    "train_ix": train_ix,
                    "test_ix": test_ix
                })
                
            bundle_name = f"{m_key}_{feat_key}_cv_bundle.joblib"
            bundle_path = os.path.join(WEIGHTS_DIR, bundle_name)
            joblib.dump(fold_pipelines, bundle_path, compress=3)
            
            total_saved += 1
            
            # Record metadata
            dep_key = f"{m_key}_{feat_key}"
            metadata["deployments"][dep_key] = {
                "model_key": m_key,
                "feature_key": feat_key,
                "model_name": metadata["models"][m_key],
                "feature_name": feat_cfg["name"],
                "dim": len(prod_cols),
                "prod_weight_file": prod_save_name,
                "cv_bundle_file": bundle_name,
                "selected_cols": prod_cols,
                "paper_val_accuracy": float(row["Validation_Accuracy"]) * 100.0,
                "paper_val_f1": float(row["Validation_F1"]) * 100.0,
                "paper_test_accuracy": float(row["Test_Accuracy"]) * 100.0,
                "paper_test_f1": float(row["Test_F1"]) * 100.0,
                "best_fold_idx": best_fold_idx,
                "best_params": prod_params
            }
            
            print(f"    + {m_key:5s} | Best Val Fold: {best_fold_idx} ({val_acc_list[best_fold_idx]*100:.2f}%) | Paper Val Acc: {float(row['Validation_Accuracy'])*100:.2f}% | Saved: {prod_save_name}")

    # Save metadata JSON
    meta_path = os.path.join(WEIGHTS_DIR, "models_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
        
    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"  FREEZING COMPLETED SUCCESSFULLY!")
    print(f"  Total models frozen: {total_saved} (Production weights + CV evaluation bundles)")
    print(f"  Destination directory: {WEIGHTS_DIR}")
    print(f"  Metadata saved to:     {meta_path}")
    print(f"  Total execution time:  {elapsed:.2f} seconds")
    print("=" * 75)

if __name__ == "__main__":
    main()
