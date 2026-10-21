import os
import sys
import base64
import platform
import warnings
import cv2
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

warnings.filterwarnings("ignore")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")
STATIC_DIR = os.path.join(APP_DIR, "static")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")

app = FastAPI(title="Coffee Bean Classifier", version="2.0.0")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

from app.pipeline import CoffeeClassifierManager
manager = CoffeeClassifierManager()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "models": manager.AVAILABLE_MODELS,
            "features": manager.AVAILABLE_FEATURES,
            "classes": manager.CLASSES
        }
    )

@app.get("/api/options")
async def get_options():
    return {
        "models": manager.AVAILABLE_MODELS,
        "features": manager.AVAILABLE_FEATURES,
        "classes": manager.CLASSES,
        "baseline_accuracies": manager.baseline_accuracies
    }

@app.get("/api/samples")
async def get_samples():
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for fname in sorted(os.listdir(SAMPLES_DIR)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                label = "Unknown"
                if "expresso" in fname.lower(): label = "Expresso"
                elif "kenya" in fname.lower(): label = "Kenya"
                elif "starbucks" in fname.lower(): label = "Starbucks"
                samples.append({
                    "id": fname,
                    "label": label,
                    "url": f"/static/samples/{fname}"
                })
    return {"samples": samples}

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB maximum upload size

@app.post("/api/predict")
async def predict_endpoint(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    sample_id: str = Form(None),
    model_name: str = Form("LGBM"),
    feature_type: str = Form("Hist_HSV_64")
):
    valid_files = []
    if files:
        valid_files.extend([f for f in files if f and f.filename])
    if file and file.filename and file not in valid_files:
        valid_files.append(file)

    # 1. Trường hợp chọn hạt mẫu sẵn (1 hạt)
    if not valid_files and sample_id:
        img_path = os.path.join(SAMPLES_DIR, sample_id)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                res = manager.predict(img, model_name=model_name, feature_type=feature_type)
                res["preview_image"] = f"/static/samples/{sample_id}"
                res["filename"] = sample_id
                return res
        raise HTTPException(status_code=400, detail="Không tìm thấy ảnh mẫu đã chọn")

    if not valid_files:
        raise HTTPException(status_code=400, detail="Vui lòng tải lên ít nhất một ảnh hạt cà phê")

    # 2. Đọc và giải mã toàn bộ ảnh được tải lên
    decoded_images = []
    filenames = []
    thumbnails = []

    for f in valid_files:
        contents = await f.read()
        if len(contents) > MAX_UPLOAD_SIZE:
            continue
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            decoded_images.append(img)
            filenames.append(f.filename)

            # Tạo ảnh thumbnail gọn nhẹ (max 160px) cho preview gallery
            h, w = img.shape[:2]
            scale = min(160.0 / max(h, w), 1.0)
            thumb = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img
            _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_str = base64.b64encode(buf).decode("utf-8")
            thumbnails.append(f"data:image/jpeg;base64,{b64_str}")

    if not decoded_images:
        raise HTTPException(status_code=400, detail="Không thể giải mã định dạng ảnh nào trong danh sách tải lên.")

    # TRƯỜNG HỢP A: Người dùng tải lên 1 ẢNH DUY NHẤT
    if len(decoded_images) == 1:
        res = manager.predict(decoded_images[0], model_name=model_name, feature_type=feature_type)
        res["preview_image"] = thumbnails[0]
        res["filename"] = filenames[0]
        return res

    # TRƯỜNG HỢP B: Người dùng tải lên NHIỀU ẢNH (MẺ HẠT CÀ PHÊ BẤT KỲ)
    res = manager.predict_batch(decoded_images, model_name=model_name, feature_type=feature_type)
    for i, item in enumerate(res["items"]):
        item["filename"] = filenames[i]
        item["preview_image"] = thumbnails[i]
    return res

import datetime
import psutil

def get_device_hardware_info():
    uname = platform.uname()
    is_jetson = False
    device_name = f"Host System ({platform.system()} {platform.machine()})"
    processor_name = platform.processor() or uname.processor or platform.machine()
    power_mode = "Standard"

    # Check genuine Jetson device tree or L4T release
    if os.path.exists("/proc/device-tree/model"):
        try:
            with open("/proc/device-tree/model", "r") as f:
                model_board = f.read().strip().replace("\x00", "")
                if model_board:
                    device_name = model_board
                    is_jetson = True
        except Exception:
            pass
    elif os.path.isfile("/etc/nv_tegra_release"):
        try:
            with open("/etc/nv_tegra_release", "r") as f:
                content = f.read()
                if "R36" in content or "nvidia" in content.lower() or "tegra" in content.lower():
                    is_jetson = True
                    device_name = "NVIDIA Jetson Orin Nano"
        except Exception:
            pass

    if is_jetson:
        processor_name = "6-Core ARM Cortex-A78AE"
        power_mode = "15W MAXN"

    mem = psutil.virtual_memory()
    ram_used = round(mem.used / (1024 ** 3), 2)
    ram_total = round(mem.total / (1024 ** 3), 2)
    ram_percent = mem.percent

    cpu_usage = psutil.cpu_percent(interval=None)

    # Temperature
    temp_c = None
    thermal_zones = [
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
        "/sys/devices/virtual/thermal/thermal_zone1/temp"
    ]
    for tz in thermal_zones:
        if os.path.exists(tz):
            try:
                with open(tz, "r") as f:
                    val = float(f.read().strip())
                    temp_c = round(val / 1000.0, 1) if val > 1000 else round(val, 1)
                    break
            except Exception:
                pass

    boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
    uptime_delta = datetime.datetime.now() - boot_time
    hours, remainder = divmod(int(uptime_delta.total_seconds()), 3600)
    minutes, _ = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes:02d}m"
    current_time_str = datetime.datetime.now().strftime("%H:%M:%S")

    # Power draw detection (Jetson INA3221 sensor or dynamic estimation)
    power_w = None
    power_simulated = False
    hwmon_dirs = ["/sys/bus/i2c/drivers/ina3221/1-0040/hwmon", "/sys/class/hwmon"]
    for hdir in hwmon_dirs:
        if os.path.exists(hdir):
            try:
                for root, dirs, files in os.walk(hdir):
                    if "power1_input" in files:
                        with open(os.path.join(root, "power1_input"), "r") as pf:
                            pval = float(pf.read().strip())
                            power_w = round(pval / 1000.0, 1) if pval > 1000 else round(pval, 1)
                            break
                if power_w is not None:
                    break
            except Exception:
                pass

    if power_w is None and is_jetson:
        # Dynamic estimation based on SoC load when hardware I2C node is unexposed
        power_w = round(4.8 + (cpu_usage / 100.0) * 3.6, 1)
        power_simulated = True

    power_limit_w = 15.0 if is_jetson else None
    power_percent = min(100, round((power_w / power_limit_w) * 100, 1)) if (power_w and power_limit_w) else None

    device_name = device_name.replace(" (L4T Tegra)", "")

    return {
        "device_name": device_name,
        "is_jetson": is_jetson,
        "cpu_percent": round(cpu_usage, 1),
        "ram_used_gb": ram_used,
        "ram_total_gb": ram_total,
        "ram_percent": ram_percent,
        "temperature_c": temp_c,
        "uptime": uptime_str,
        "current_time": current_time_str,
        "power_w": power_w,
        "power_simulated": power_simulated,
        "power_limit_w": power_limit_w,
        "power_percent": power_percent,
        "power_mode": power_mode,
        "processor": processor_name
    }

@app.get("/api/system-status")
async def system_status_endpoint():
    return get_device_hardware_info()

@app.get("/api/health")
@app.get("/health")
async def health_check():
    hw_info = get_device_hardware_info()
    return {
        "status": "healthy",
        "service": "Coffee Bean Edge AI Classifier",
        "version": "2.0.0",
        "total_models": len(manager.baseline_accuracies),
        "device": hw_info.get("device_name", "Unknown"),
        "is_jetson": hw_info.get("is_jetson", False)
    }


