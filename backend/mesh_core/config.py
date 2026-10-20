"""Global configuration for the Mesh backend."""
from __future__ import annotations

import os
from pathlib import Path

import torch

# Force all AI model caches onto Drive C: (where >50 GB is free)
USER_CACHE_DIR = Path("C:/Users/Admin/.cache")
os.environ.setdefault("HF_HOME", str(USER_CACHE_DIR / "huggingface"))
os.environ.setdefault("U2NET_HOME", str(Path("C:/Users/Admin/.u2net")))
os.environ.setdefault("TORCH_HOME", str(USER_CACHE_DIR / "torch"))

ROOT_DIR = Path(__file__).resolve().parents[2]          # d:/My_Software/Mesh
BACKEND_DIR = ROOT_DIR / "backend"
THIRD_PARTY_DIR = BACKEND_DIR / "third_party"
TRIPOSR_DIR = THIRD_PARTY_DIR / "TripoSR"
HUNYUAN3D_DIR = THIRD_PARTY_DIR / "Hunyuan3D-2"

DATA_DIR = Path(os.environ.get("MESH_DATA_DIR", ROOT_DIR / "data"))
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model ids (all cached on Drive C:) --------------------------------------
TEXT2IMG_MODEL_ID = os.environ.get("MESH_T2I_MODEL", "Lykon/dreamshaper-8")
TRIPOSR_MODEL_ID = "stabilityai/TripoSR"
HUNYUAN3D_MODEL_ID = os.environ.get("MESH_HY3D_MODEL", "tencent/Hunyuan3D-2mini")
HUNYUAN3D_SUBFOLDER = os.environ.get("MESH_HY3D_SUBFOLDER", "hunyuan3d-dit-v2-mini-turbo")
HUNYUAN3D_FULL_MODEL_ID = "tencent/Hunyuan3D-2"
HUNYUAN3D_FULL_SUBFOLDER = "hunyuan3d-dit-v2-0-turbo"

MESH_3D_ENGINE = os.environ.get("MESH_3D_ENGINE", "hunyuan3d_char")

ENGINES_3D = [
    {
        "key": "hunyuan3d_char",
        "label": "Hunyuan3D-2.0 Full 1.1B • Chuyên Nhân Vật & Chi Tiết (3072 Latents)",
        "badge": "Đẹp nhất • Nhân vật / Người",
        "desc": "Dùng 3,072 Latents (gấp 6 lần chi tiết) + tách nền IS-Net + Lưới 384³ (~40s). Tối ưu cho người, nhân vật, con vật.",
    },
    {
        "key": "hunyuan3d_mini_hq",
        "label": "Hunyuan3D-2 Mini Ultra (0.6B • 3072 Latents • 384³)",
        "badge": "Nét cao & Nhẹ • 3.2GB VRAM",
        "desc": "Bản 0.6B nhưng mở khóa 3,072 Latents gốc + Lưới 384³ (~25s). Đẹp cho cả nhân vật lẫn đồ vật.",
    },
    {
        "key": "hunyuan3d_mini",
        "label": "Hunyuan3D-2 Mini Turbo (0.6B • 512 Latents)",
        "badge": "Nhanh • Trái cây / Đồ vật",
        "desc": "Nhanh (~14s, 2.4GB VRAM, Lưới 256³). Phù hợp cho trái cây, rau củ, đồ vật đơn giản.",
    },
    {
        "key": "triposr",
        "label": "TripoSR (Dự phòng siêu tốc)",
        "badge": "Siêu tốc ~5s",
        "desc": "Model NeRF nhẹ dự phòng (~5s).",
    },
]

# --- Low-VRAM knobs ----------------------------------------------------------
TRIPOSR_CHUNK_SIZE = int(os.environ.get("MESH_TRIPOSR_CHUNK", 4096))
TRIPOSR_MC_RESOLUTION = int(os.environ.get("MESH_MC_RES", 256))
HUNYUAN3D_STEPS = int(os.environ.get("MESH_HY3D_STEPS", 10))
HUNYUAN3D_OCTREE_RES = int(os.environ.get("MESH_HY3D_OCTREE_RES", 256))
