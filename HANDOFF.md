# HANDOFF – Dự án "Mesh Studio" (Tạo model 3D low-poly cho game nông trại Unity)

## Trạng thái hiện tại: HOÀN TẤT BẢN MVP DESKTOP (.EXE LAUNCHER) & ĐÃ KIỂM CHỨNG

1. **Đã tải đầy đủ 100% model AI về máy** (`Lykon/dreamshaper-8` fp16 + `stabilityai/TripoSR`):
   - Đã gỡ `hf-xet` và cập nhật [`backend/download_models.py`](file:///d:/My_Software/Mesh/backend/download_models.py) dùng `curl.exe` gốc của Windows để không bị kẹt kết nối ở mốc 10 MB.
   - Đã dọn 3.2 GB file `.incomplete` thừa trong cache Hugging Face.

2. **Đã sửa lỗi tương thích & chạy thành công pipeline Text → 2D → 3D Low-Poly Unity**:
   - Cố định bộ thư viện tương thích trong `.venv`: `transformers==4.45.2`, `diffusers==0.32.2`, `accelerate==1.2.1`, `huggingface-hub==0.36.2`.
   - Sửa cấu hình `DPMSolverMultistepScheduler` (`algorithm_type="dpmsolver++"`) trong [`backend/mesh_core/text2img.py`](file:///d:/My_Software/Mesh/backend/mesh_core/text2img.py).
   - **Kết quả đo thực tế trên RTX 3050 Laptop (4 GB VRAM)**:
     - Bước SD 1.5 (Text → Image): **peak 2.63 GB VRAM**, ~15–25s.
     - Bước TripoSR (Image → 3D): **peak 2.33 GB VRAM**, ~13.3s.
     - Bước tối ưu Low-Poly + Palette + Xuất GLB/OBJ: **0.76s**.
     - Model đầu tiên đã sinh tại: `data/outputs/20261019_155619_a_white_and_black_dairy_cow/`.

3. **Đã hoàn thiện Backend FastAPI (`backend/server.py`) & tính năng chỉnh thông số tức thì**:
   - Lưu kèm `dense.ply` trong mỗi thư mục output và thêm hàm `reprocess_asset()` trong [`backend/mesh_core/pipeline.py`](file:///d:/My_Software/Mesh/backend/mesh_core/pipeline.py) để đổi số tam giác / số màu / chiều cao / góc xoay trong ~0.8s mà không cần chạy lại AI.
   - Các API: `/api/presets`, `/api/status`, `/api/generate/images`, `/api/generate/asset`, `/api/generate/upload`, `/api/library`, `/api/library/{id}/reprocess`, `/api/library/{id}/open-folder`.

4. **Đã hoàn thiện App Desktop Electron + React + Three.js (`frontend/`)**:
   - Đã cài đặt dependencies và build sẵn tại `frontend/dist/`.
   - Khởi chạy bằng 1 cú nhấp đúp vào [`Start-Mesh-Studio.bat`](file:///d:/My_Software/Mesh/Start-Mesh-Studio.bat) (mở cửa sổ app desktop `electron.exe` và tự động bật/tắt Python backend ngầm ở cổng `8765`).

## Các bước nâng cấp tiếp theo (nếu muốn mở rộng)
- **Bảng màu cố định toàn game (Custom Palette)**: cho phép nạp 1 file ảnh palette chung của dự án Unity để mọi asset đều dùng chung đúng bảng màu đó.
- **Đóng gói bộ cài đặt độc lập (`electron-builder`)**: nếu muốn mang sang máy khác cài đặt.
- **Giai đoạn hoạt hình (Rigging & Animation)**: tích hợp `bpy` và khung xương mẫu cho nhân vật / con vật khi bắt đầu làm animation.
