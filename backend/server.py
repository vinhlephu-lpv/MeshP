"""FastAPI server for the Mesh desktop/web app."""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from mesh_core.config import DEVICE, OUTPUT_DIR
from mesh_core.model_manager import manager
from mesh_core.pipeline import image_to_asset, reprocess_asset, text_to_images
from mesh_core.postprocess import PostprocessOptions
from mesh_core.presets import PRESETS, build_prompt

app = FastAPI(title="Mesh – Farm Game 3D Asset Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


class ProgressState:
    def __init__(self) -> None:
        self.busy: bool = False
        self.stage: str = "Sẵn sàng"
        self.percent: float = 0.0
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def update(self, stage: str, pct: float) -> None:
        with self._lock:
            self.stage = stage
            self.percent = round(float(pct), 1)
            print(f"[{self.percent:5.1f}%] {self.stage}", flush=True)

    def start(self, stage: str = "Bắt đầu...") -> None:
        with self._lock:
            self.busy = True
            self.stage = stage
            self.percent = 0.0
            self.error = None

    def finish(self, error: Optional[str] = None) -> None:
        with self._lock:
            self.busy = False
            if error:
                self.error = error
                self.stage = f"Lỗi: {error}"
            else:
                self.stage = "Hoàn tất"
                self.percent = 100.0


state = ProgressState()
_gpu_lock = threading.Lock()


def _pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _data_url_to_pil(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _enrich_meta(meta: dict) -> dict:
    out_dir = Path(meta.get("dir", OUTPUT_DIR / meta.get("id", "")))
    asset_id = out_dir.name
    meta["id"] = asset_id
    slug = meta.get("slug", "asset")
    meta["urls"] = {
        "glb": f"/outputs/{asset_id}/{slug}.glb",
        "obj": f"/outputs/{asset_id}/obj/{slug}.obj",
        "mtl": f"/outputs/{asset_id}/obj/{slug}.mtl",
        "texture": f"/outputs/{asset_id}/obj/{slug}_palette.png",
        "thumbnail": f"/outputs/{asset_id}/thumbnail.png",
        "turnaround": f"/outputs/{asset_id}/turnaround.png",
        "source": f"/outputs/{asset_id}/source.png",
        "prepared": f"/outputs/{asset_id}/prepared.png",
    }
    meta["has_dense"] = (out_dir / "dense.ply").exists()
    return meta


@app.get("/api/presets")
def get_presets():
    return {
        "device": DEVICE,
        "vram": manager.vram_report(),
        "presets": [asdict(p) for p in PRESETS.values()],
    }


@app.get("/api/status")
def get_status():
    return {
        "busy": state.busy,
        "stage": state.stage,
        "percent": state.percent,
        "error": state.error,
        "vram": manager.vram_report(),
    }


class GenerateImagesReq(BaseModel):
    subject: str
    preset: str = "animal"
    num_images: int = 2
    seed: Optional[int] = None


@app.post("/api/generate/images")
def api_generate_images(req: GenerateImagesReq):
    if req.preset not in PRESETS:
        raise HTTPException(400, f"Unknown preset: {req.preset}")
    if not _gpu_lock.acquire(blocking=False):
        raise HTTPException(409, "GPU đang xử lý tác vụ khác, vui lòng đợi.")
    try:
        num = max(1, min(4, req.num_images))
        state.start(f"Đang vẽ {num} ảnh 2D từ mô tả...")
        imgs = text_to_images(req.subject, req.preset, num_images=num, seed=req.seed)
        state.finish()
        return {
            "prompt": build_prompt(req.subject, req.preset),
            "images": [_pil_to_data_url(im) for im in imgs],
            "vram": manager.vram_report(),
        }
    except Exception as e:
        state.finish(str(e))
        raise HTTPException(500, str(e))
    finally:
        _gpu_lock.release()


class GenerateAssetFromDataReq(BaseModel):
    image_data_url: str
    name: str
    preset: str = "animal"
    target_faces: Optional[int] = None
    palette_colors: Optional[int] = None
    target_height_m: Optional[float] = None
    front_yaw_deg: float = 0.0


@app.post("/api/generate/asset")
def api_generate_asset(req: GenerateAssetFromDataReq):
    if req.preset not in PRESETS:
        raise HTTPException(400, f"Unknown preset: {req.preset}")
    if not _gpu_lock.acquire(blocking=False):
        raise HTTPException(409, "GPU đang xử lý tác vụ khác, vui lòng đợi.")
    try:
        state.start("Chuẩn bị dựng 3D...")
        preset = PRESETS[req.preset]
        opts = PostprocessOptions(
            target_faces=req.target_faces or preset.target_faces,
            palette_colors=req.palette_colors or preset.palette_colors,
            target_height_m=req.target_height_m or preset.target_height_m,
            front_yaw_deg=req.front_yaw_deg,
        )
        img = _data_url_to_pil(req.image_data_url)
        meta = image_to_asset(img, req.name or req.preset, req.preset, opts, progress=state.update)
        state.finish()
        return _enrich_meta(meta)
    except Exception as e:
        state.finish(str(e))
        raise HTTPException(500, str(e))
    finally:
        _gpu_lock.release()


@app.post("/api/generate/upload")
def api_generate_upload(
    file: UploadFile = File(...),
    name: str = Form("uploaded_asset"),
    preset: str = Form("prop"),
    target_faces: Optional[int] = Form(None),
    palette_colors: Optional[int] = Form(None),
    target_height_m: Optional[float] = Form(None),
):
    if preset not in PRESETS:
        raise HTTPException(400, f"Unknown preset: {preset}")
    raw = file.file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    if not _gpu_lock.acquire(blocking=False):
        raise HTTPException(409, "GPU đang xử lý tác vụ khác, vui lòng đợi.")
    try:
        state.start("Chuẩn bị dựng 3D từ ảnh tải lên...")
        p = PRESETS[preset]
        opts = PostprocessOptions(
            target_faces=target_faces or p.target_faces,
            palette_colors=palette_colors or p.palette_colors,
            target_height_m=target_height_m or p.target_height_m,
        )
        meta = image_to_asset(img, name or Path(file.filename or "asset").stem, preset, opts, progress=state.update)
        state.finish()
        return _enrich_meta(meta)
    except Exception as e:
        state.finish(str(e))
        raise HTTPException(500, str(e))
    finally:
        _gpu_lock.release()


class ReprocessReq(BaseModel):
    target_faces: int = 2000
    palette_colors: int = 12
    target_height_m: float = 1.0
    saturation_boost: float = 1.15
    front_yaw_deg: float = 0.0


@app.post("/api/library/{asset_id}/reprocess")
def api_reprocess(asset_id: str, req: ReprocessReq):
    if not _gpu_lock.acquire(blocking=False):
        raise HTTPException(409, "Đang bận xử lý tác vụ khác.")
    try:
        state.start("Tối ưu lại lưới & bảng màu...")
        opts = PostprocessOptions(
            target_faces=req.target_faces,
            palette_colors=req.palette_colors,
            target_height_m=req.target_height_m,
            saturation_boost=req.saturation_boost,
            front_yaw_deg=req.front_yaw_deg,
        )
        meta = reprocess_asset(asset_id, opts, progress=state.update)
        state.finish()
        return _enrich_meta(meta)
    except FileNotFoundError as e:
        state.finish(str(e))
        raise HTTPException(404, str(e))
    except Exception as e:
        state.finish(str(e))
        raise HTTPException(500, str(e))
    finally:
        _gpu_lock.release()


@app.get("/api/library")
def api_library():
    items = []
    if OUTPUT_DIR.exists():
        for d in sorted(OUTPUT_DIR.iterdir(), reverse=True):
            meta_file = d / "meta.json"
            if d.is_dir() and meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text("utf-8"))
                    items.append(_enrich_meta(meta))
                except Exception:
                    pass
    return {"items": items}


@app.delete("/api/library/{asset_id}")
def api_delete_asset(asset_id: str):
    target = (OUTPUT_DIR / asset_id).resolve()
    if target.parent != OUTPUT_DIR.resolve() or not target.exists():
        raise HTTPException(404, "Không tìm thấy asset")
    shutil.rmtree(target)
    return {"deleted": asset_id}


@app.post("/api/library/{asset_id}/open-folder")
def api_open_folder(asset_id: str):
    target = (OUTPUT_DIR / asset_id).resolve()
    if target.parent != OUTPUT_DIR.resolve() or not target.exists():
        raise HTTPException(404, "Không tìm thấy thư mục")
    if os.name == "nt":
        subprocess.Popen(["explorer", str(target)])
    return {"opened": str(target)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("MESH_PORT", "8765"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
