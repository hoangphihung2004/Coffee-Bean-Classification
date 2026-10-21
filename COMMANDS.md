# HƯỚNG DẪN CÁC LỆNH VẬN HÀNH THỦ CÔNG (CHEF CHEAT-SHEET)

Tài liệu này lưu trữ toàn bộ các lệnh chạy thủ công để quản lý hệ thống phân loại hạt cà phê trên **NVIDIA Jetson** (qua Docker) và trên **máy tính cá nhân (PC)**.

---

## I. THAO TÁC TRÊN THIẾT BỊ JETSON (QUA DOCKER)

Tên container cố định: **`coffee_edge_ai_container`**

### 1. Kết nối vào Jetson từ máy tính (qua cáp Type-C)
Mở PowerShell hoặc Command Prompt trên PC và gõ:
```bash
ssh -i ~/.ssh/id_jetson fptdanang@192.168.55.1
```

---

### 2. Quản lý Web Dashboard (Bật / Tắt để mở Web)

* **BẬT Web Dashboard (Chạy ngầm):**
  ```bash
  docker start coffee_edge_ai_container
  ```
  > Khi bật xong, mở trình duyệt trên máy tính/điện thoại vào:
  > 👉 **`http://192.168.55.1:8000`** *(nếu cắm cáp Type-C)*
  > 👉 **`http://<IP_Wifi_Jetson>:8000`** *(nếu dùng Wi-Fi chung mạng)*

* **TẮT Web Dashboard:**
  ```bash
  docker stop coffee_edge_ai_container
  ```

* **Kiểm tra trạng thái container đang chạy:**
  ```bash
  docker ps
  ```

* **Xem log trực tiếp của Web (khi upload ảnh hoặc debug):**
  ```bash
  docker logs -f coffee_edge_ai_container
  ```

---

### 3. Chạy Kết quả Riêng biệt dưới Terminal (Khi container đang chạy)

Dùng lệnh `docker exec` để gọi trực tiếp các kịch bản đo đạc mà không làm gián đoạn Web:

* **Chạy ĐÁNH GIÁ TOÀN TRÌNH TRỰC TIẾP TỪ ẢNH GỐC (End-to-End Raw Images - Khuyên Dùng):**
  > Kịch bản này nạp trực tiếp từng file ảnh thật (`.jpg`), tự động trích xuất đặc trưng thuần túy trên CPU Jetson ARM64, đưa vào pipeline tiền xử lý và suy luận, đo chính xác cả thời gian trích xuất $T_{\text{feature}}$, thời gian suy luận $T_{\text{infer}}$, tổng độ trễ $T_{\text{total}}$, FPS và đối chiếu độ chính xác:
  ```bash
  # Chạy 10-Fold CV toàn trình từ ảnh thô (Đối chiếu chính xác 100% với Paper):
  docker exec -it coffee_edge_ai_container python benchmarking/evaluate_end_to_end.py --mode cv

  # Chạy riêng 1 mô hình & đặc trưng tối ưu (Ví dụ: LGBM + Top 20% HSV):
  docker exec -it coffee_edge_ai_container python benchmarking/evaluate_end_to_end.py --model LGBM --feature Top_20_Hist_HSV_64 --mode cv

  # Chạy trên mô hình Production (Fit full dataset):
  docker exec -it coffee_edge_ai_container python benchmarking/evaluate_end_to_end.py --mode prod
  ```

* **Chạy kịch bản KIỂM CHỨNG BẢO TOÀN ĐỘ CHÍNH XÁC (Tập đặc trưng nén sẵn):**
  ```bash
  docker exec -it coffee_edge_ai_container python benchmarking/evaluate_preservation.py
  ```

* **Chỉ chạy kịch bản ĐO ĐỘ TRỄ SUY LUẬN & FPS TRÊN JETSON:**
  ```bash
  docker exec -it coffee_edge_ai_container python benchmarking/benchmark_latency.py
  ```

* **Lọc đo riêng 1 mô hình hoặc 1 đặc trưng cụ thể:**
  ```bash
  # Ví dụ chỉ đo mô hình LGBM trên đặc trưng Hist_HSV_64:
  docker exec -it coffee_edge_ai_container python benchmarking/evaluate_preservation.py --model LGBM --feature Hist_HSV_64
  ```

---

### 4. Khóa Xung & Bật Hiệu Năng Tối Đa (Jetson Clock Locking & Performance Tuning)

Để mô hình đạt tốc độ suy luận nhanh nhất và FPS cao nhất, bạn chạy 2 lệnh sau trước khi benchmark:

* **Bước 1: Bật chế độ nguồn tối đa (MAXN Mode - Bung toàn bộ 15W công suất):**
  ```bash
  sudo nvpmodel -m 0
  ```

* **Bước 2: Khóa cứng xung nhịp CPU, GPU và RAM ở mức tối đa (Lock Clocks):**
  ```bash
  sudo jetson_clocks
  ```
  *(Lệnh này sẽ đẩy xung CPU/GPU lên tần số kịch trần và bật quạt tản nhiệt quay tối đa để tránh bị hạ xung / throttling do nhiệt).*

* **Xem thông số xung nhịp hiện tại:**
  ```bash
  sudo jetson_clocks --show
  ```

* **Khôi phục lại chế độ tự động điều tiết xung (khi không test nữa):**
  ```bash
  sudo jetson_clocks --restore
  ```

---

### 5. Build lại Docker (Nếu có chỉnh sửa mã nguồn)
```bash
cd ~/coffee_edge_deployment
docker compose build
docker compose up -d
```

---

## II. THAO TÁC TRÊN MÁY TÍNH CÁ NHÂN (LOCAL PC - KHÔNG DÙNG JETSON)

Mở PowerShell tại thư mục: `D:\New_Ideas\Coffee\coffee_edge_deployment`

### 1. Khởi động Web Dashboard trên PC:
```powershell
python run_server.py
```
> Truy cập trên trình duyệt: **`http://localhost:8000`**

### 2. Chạy toàn bộ Benchmark trên Terminal PC:
```powershell
python benchmarking/run_all_benchmarks.py
```

### 3. Chỉ chạy kiểm chứng bảo toàn độ chính xác trên PC:
```powershell
python benchmarking/evaluate_preservation.py
```

### 4. Chỉ chạy đo tốc độ suy luận & FPS trên PC:
```powershell
python benchmarking/benchmark_latency.py
```

---

## III. VỊ TRÍ CÁC FILE KẾT QUẢ ĐỂ ĐƯA VÀO BÀI BÁO (PAPER)

Tất cả các file kết quả đo đạc chính thức từ Jetson được lưu tại:
📁 `D:\New_Ideas\Coffee\coffee_edge_deployment\benchmarking\results\jetson_hardware_results\`

* `paper_preservation_table.tex`: Mã nguồn bảng LaTeX đối chiếu độ chính xác.
* `preservation_report.csv`: Báo cáo chi tiết độ chính xác 10-fold CV.
* `latency_breakdown.csv`: Bảng số liệu chi tiết mili-giây từng khâu ($T_{\text{feature}}$, $T_{\text{infer}}$, $T_{\text{total}}$, FPS).
* `latency_summary.json`: Tóm tắt cấu hình phần cứng Jetson và kết quả tốc độ.
* `confusion_matrix_best.png`: Ma trận nhầm lẫn chuẩn in ấn 300 DPI của mô hình tốt nhất.
