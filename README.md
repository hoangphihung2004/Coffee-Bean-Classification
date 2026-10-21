# Coffee Bean Edge AI Classifier ☕🔬

Production-grade Edge AI deployment and benchmarking suite for 3-class coffee bean classification (*Expresso*, *Kenya*, *Starbucks*) on **NVIDIA Jetson Orin Nano** (ARM Cortex-A78AE).

Supports **36 model-feature combinations** (6 Machine Learning models $\times$ 6 feature sets) with **Zero-Retraining Accuracy Preservation ($\Delta = 0.00\%$)** directly evaluated from raw JPEG images on hardware.

---

## Project Directory Structure
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
