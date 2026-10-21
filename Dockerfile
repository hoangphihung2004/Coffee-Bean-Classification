# ==============================================================================
# Dockerfile for Coffee Bean Edge AI Deployment (NVIDIA Jetson / ARM64 & x86_64)
# ==============================================================================

FROM python:3.10-slim

# Ngăn chặn tạo file .pyc và bật unbuffered output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV, LightGBM và phần cứng Jetson
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy toàn bộ mã nguồn, weights đóng băng và bộ test benchmark
COPY . /workspace

# Mở cổng 8000 cho Web Dashboard
EXPOSE 8000

# Lệnh mặc định: Khởi động Web Dashboard
CMD ["python", "run_server.py"]
