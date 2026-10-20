"""Global configuration for the Mesh backend."""
from __future__ import annotations

import os
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]          # d:/My_Software/Mesh
BACKEND_DIR = ROOT_DIR / "backend"
THIRD_PARTY_DIR = BACKEND_DIR / "third_party"
TRIPOSR_DIR = THIRD_PARTY_DIR / "TripoSR"
HUNYUAN3D_DIR = THIRD_PARTY_DIR / "Hunyuan3D-2"

DATA_DIR = Path(os.environ.get("MESH_DATA_DIR", ROOT_DIR / "data"))
OUTPUT_DIR = DATA_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model ids (all free, downloaded once from Hugging Face) -----------------
TEXT2IMG_MODEL_ID = os.environ.get("MESH_T2I_MODEL", "Lykon/dreamshaper-8")
TRIPOSR_MODEL_ID = "stabilityai/TripoSR"
HUNYUAN3D_MODEL_ID = os.environ.get("MESH_HY3D_MODEL", "tencent/Hunyuan3D-2mini")
HUNYUAN3D_SUBFOLDER = os.environ.get("MESH_HY3D_SUBFOLDER", "hunyuan3d-dit-v2-mini-turbo")
HUNYUAN3D_FULL_MODEL_ID = "tencent/Hunyuan3D-2"
HUNYUAN3D_FULL_SUBFOLDER = "hunyuan3d-dit-v2-0-turbo"

MESH_3D_ENGINE = os.environ.get("MESH_3D_ENGINE", "hunyuan3d_mini")

ENGINES_3D = [
    {
        "key": "hunyuan3d_mini",
        "label": "Hunyuan3D-2 Mini Turbo (0.6B)",
        "badge": "Khuyên dùng • 2.4GB VRAM",
        "desc": "Nhanh (~15s), mượt, giữ nguyên tốc độ hiện tại (Lưới 256³, 10 bước).",
    },
    {
        "key": "hunyuan3d_mini_hq",
        "label": "Hunyuan3D-2 Mini Ultra (0.6B • Lưới 384³)",
        "badge": "Nét cao • 3.2GB VRAM",
        "desc": "Dùng sẵn model 0.6B nhưng tăng độ phân giải khối lên 384³ & 18 bước (~28s).",
    },
    {
        "key": "hunyuan3d_full",
        "label": "Hunyuan3D-2.0 Full Turbo (1.1B • Mạnh nhất)",
        "badge": "Siêu nét • 4.5–6GB VRAM",
        "desc": "Model 1.1 tỷ tham số lớn gấp đôi + Lưới 384³ & 16 bước (~35s).",
    },
    {
        "key": "triposr",
        "label": "TripoSR (Dự phòng siêu tốc)",
        "badge": "Nhẹ • 1.8GB VRAM",
        "desc": "Model NeRF nhẹ dự phòng (~5s).",
    },
]

# --- Low-VRAM knobs ----------------------------------------------------------
TRIPOSR_CHUNK_SIZE = int(os.environ.get("MESH_TRIPOSR_CHUNK", 4096))
TRIPOSR_MC_RESOLUTION = int(os.environ.get("MESH_MC_RES", 256))
HUNYUAN3D_STEPS = int(os.environ.get("MESH_HY3D_STEPS", 10))
HUNYUAN3D_OCTREE_RES = int(os.environ.get("MESH_HY3D_OCTREE_RES", 256))
