# Coffee Bean Edge AI Classifier ☕🔬

Production-grade Edge AI deployment and benchmarking suite for 3-class coffee bean classification (*Expresso*, *Kenya*, *Starbucks*) on **NVIDIA Jetson Orin Nano** (ARM Cortex-A78AE).

Supports **36 model-feature combinations** (6 Machine Learning models $\times$ 6 feature sets) with **Zero-Retraining Accuracy Preservation ($\Delta = 0.00\%$)** directly evaluated from raw JPEG images on hardware.

---

## 1. Project Directory Structure
```
coffee_edge_deployment/
├── app/
│   ├── features.py              # Canonical feature extractor (LBP, GLCM, ELA, HSV-CH)
│   ├── pipeline.py              # Frozen production model manager & single-sample inference
│   ├── main.py                  # FastAPI application & system telemetry API
│   ├── templates/
│   │   └── index.html           # Edge AI Web Dashboard (Tailwind CSS, responsive)
│   └── static/
│       └── samples/             # Sample coffee bean images (Expresso, Kenya, Starbucks)
├── data/                        # Dataset for full scientific reproduction & benchmarking
│   ├── extracted_features_unsegmented.csv # Pre-extracted feature matrix (48.9 MB)
│   ├── results.csv              # Research tuning and validation log
│   └── Expresso/, Kenya/, Starbucks/      # Raw JPEG images (1,554 images)
├── weights/
│   ├── *.joblib                 # 36 Production pipelines (StandardScaler + Model fitted on full dataset)
│   ├── *_cv_bundle.joblib       # 36 10-Fold CV evaluation bundles (Zero-retraining preservation)
│   └── models_metadata.json     # Hyperparameters, column indices, and baseline results
├── benchmarking/
│   ├── evaluate_end_to_end.py   # True End-to-End benchmark directly from raw JPEG images
│   ├── evaluate_preservation.py # Out-of-fold accuracy preservation benchmark
│   ├── benchmark_latency.py     # Pure hardware latency & throughput (FPS) breakdown
│   ├── run_all_benchmarks.py    # Unified CLI benchmark runner
│   └── results/                 # Publication CSV reports, LaTeX tables, confusion matrices (Jetson Orin)
├── freeze_models.py             # Reproducibility script: train & freeze 36 models from features
├── run_server.py                # Server entrypoint (Uvicorn / FastAPI)
├── Dockerfile                   # Multi-arch container definition (ARM64 / x86_64)
├── docker-compose.yml           # Jetson production container deployment
├── docker_run.sh                # Helper script for container lifecycle
├── requirements.txt             # Python dependencies
└── COMMANDS.md                  # Comprehensive operational cheat-sheet
```

---

## 2. Supported Models & Feature Representations

### Machine Learning Models (6 Classifiers)
1. **LGBM**: Light Gradient Boosting Machine (Multi-class GBDT Engine)
2. **RF**: Optimized Random Forest Ensemble
3. **ET**: Extremely Randomized Trees Ensemble
4. **SVM**: Support Vector Classifier (RBF Kernel & Probability estimation)
5. **KNN**: K-Nearest Neighbors Classifier
6. **LR**: Multinomial Logistic Regression (L-BFGS)

### Feature Sets (6 Representations)
1. **LBP (59-D)**: Local Binary Patterns (`nri_uniform`, $P=8, R=1$, density-normalized)
2. **GLCM (16-D)**: Gray-Level Co-occurrence Matrix ($d=3$, 4 angles, 4 properties)
3. **ELA (48-D)**: Error Level Analysis (JPEG $Q=90$, 3 channels $\times$ [5 statistics + 11 bins])
4. **Hist_HSV_64 (192-D)**: HSV Color Histogram (64 bins $\times$ 3 channels, normalized)
5. **Top_20_Hist_HSV_64 (39-D)**: Top 39 salient HSV features via Feature Selection
6. **ELA + Hist_HSV_64 (240-D)**: Fused Error Level Analysis (48) and HSV Histogram (192)

---

## 3. Verified Jetson Hardware Benchmarks (NVIDIA Jetson Orin Nano)

Evaluated directly on 1,554 raw JPEG images with single-sample execution (batch size = 1):

| Feature Representation | Dim | Extraction ($T_{\text{feat}}$) | Inference ($T_{\text{infer}}$) | Total Latency ($T_{\text{total}}$) | Throughput (FPS) | Paper Acc | Deployed Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Top 20% HSV-CH (LGBM)** | **39** | **1.585 ms** | **1.714 ms** | **3.299 ms** | **303.1 FPS** | **95.50%** | **95.50 ± 1.82%** |
| HSV-CH (LGBM) | 192 | 1.584 ms | 1.721 ms | 3.305 ms | 302.6 FPS | 95.50% | 95.50 ± 1.82% |
| LBP (LGBM) | 59 | 18.197 ms | 1.781 ms | 19.978 ms | 50.1 FPS | 77.28% | 77.28 ± 2.65% |
| GLCM (LGBM) | 16 | 25.900 ms | 1.729 ms | 27.629 ms | 36.2 FPS | 80.82% | 80.82 ± 3.49% |
| ELA (LGBM) | 48 | 29.985 ms | 1.793 ms | 31.778 ms | 31.5 FPS | 87.07% | 87.07 ± 2.19% |
| ELA + HSV-CH (LGBM) | 240 | 31.523 ms | 1.703 ms | 33.226 ms | 30.1 FPS | 94.66% | 94.66 ± 1.53% |

*Accuracies reflect 10-fold cross-validation out-of-fold testing, matched to paper down to 0.01% ($\Delta = 0.00\%$).*

---

## 4. Quick Start

### A. Run Locally (PC / Development)
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start Web Dashboard:
   ```bash
   python run_server.py
   ```
   Open browser at: `http://localhost:8000` (API docs: `http://localhost:8000/docs`).

### B. Run on NVIDIA Jetson via Docker
1. Start the container:
   ```bash
   docker start coffee_edge_ai_container
   ```
   Access Web Dashboard at `http://192.168.55.1:8000` (via USB Type-C) or `http://<jetson-ip>:8000`.

2. Run End-to-End Evaluation from Raw Images:
   ```bash
   docker exec -it coffee_edge_ai_container python benchmarking/evaluate_end_to_end.py --mode cv
   ```

3. Run Unified Benchmark Suite:
   ```bash
   docker exec -it coffee_edge_ai_container python benchmarking/run_all_benchmarks.py --e2e
   ```

---

## 5. Full Scientific Reproducibility Pipeline

To reproduce all models, evaluation metrics, and paper figures from scratch:

1. **Retrain & Freeze 36 Models from Extracted Features**:
   ```bash
   python freeze_models.py
   ```
   *Generates all 36 `.joblib` production pipelines, 36 `_cv_bundle.joblib` bundles, and `models_metadata.json`.*

2. **Verify 10-Fold Cross-Validation Accuracy Preservation ($\Delta = 0.00\%$)**:
   ```bash
   python benchmarking/evaluate_preservation.py
   ```

3. **Verify End-to-End Image Inference Accuracy directly on 1,554 Raw JPEGs**:
   ```bash
   python benchmarking/evaluate_end_to_end.py --mode cv
   ```

4. **Benchmark Real-Time Latency & FPS Breakdown**:
   ```bash
   python benchmarking/benchmark_latency.py
   ```

