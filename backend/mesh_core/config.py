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
MESH_3D_ENGINE = os.environ.get("MESH_3D_ENGINE", "hunyuan3d")  # "hunyuan3d" or "triposr"

# --- Low-VRAM knobs (tuned for 4 GB GPUs) -------------------------------------
TRIPOSR_CHUNK_SIZE = int(os.environ.get("MESH_TRIPOSR_CHUNK", 4096))
TRIPOSR_MC_RESOLUTION = int(os.environ.get("MESH_MC_RES", 256))
HUNYUAN3D_STEPS = int(os.environ.get("MESH_HY3D_STEPS", 10))
HUNYUAN3D_OCTREE_RES = int(os.environ.get("MESH_HY3D_OCTREE_RES", 256))

