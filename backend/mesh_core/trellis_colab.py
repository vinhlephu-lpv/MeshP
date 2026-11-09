"""Standalone runner for Microsoft TRELLIS (microsoft/TRELLIS-image-large) on Google Colab T4."""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Mandatory environment variables for Tesla T4 (Turing sm_75) before importing trellis
os.environ["MPLBACKEND"] = "Agg"
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"
os.environ["MODEL_CHECKSUM_DISABLED"] = "1"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")

if "/content/TRELLIS" not in sys.path:
    sys.path.insert(0, "/content/TRELLIS")

import cv2
import imageio
import numpy as np
import pyvista as pv
import torch
import trimesh
import trimesh.visual
import xatlas
from huggingface_hub import hf_hub_download
from PIL import Image
from pymeshfix import _meshfix
from scipy.spatial import cKDTree

# Bypass GitHub web 504 timeout inside torch.hub.load('facebookresearch/dinov2', ...)
_orig_hub_load = torch.hub.load


def _safe_hub_load(repo_or_dir, model, *args, **kwargs):
    if repo_or_dir == "facebookresearch/dinov2":
        local_dino = "/content/dinov2"
        if not os.path.exists(f"{local_dino}/hubconf.py"):
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/facebookresearch/dinov2.git", local_dino],
                check=True,
            )
        kwargs["source"] = "local"
        repo_or_dir = local_dino
    return _orig_hub_load(repo_or_dir, model, *args, **kwargs)


torch.hub.load = _safe_hub_load

SH_C0 = 0.28209479177387814


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return (s or "asset")[:40]


def ensure_u2net() -> None:
    """Download u2net.onnx from HuggingFace mirror so rembg never fails with GitHub 504."""
    cached = hf_hub_download(repo_id="tomjackson2023/rembg", filename="u2net.onnx")
    for d in {os.environ.get("U2NET_HOME", "/root/.u2net"), os.path.expanduser("~/.u2net")}:
        Path(d).mkdir(parents=True, exist_ok=True)
        dst = Path(d) / "u2net.onnx"
        if not dst.exists() or dst.stat().st_size < 10_000_000:
            shutil.copyfile(cached, dst)


def render_gaussian_views_cuda(
    g_xyz: np.ndarray,
    g_rgb: np.ndarray,
    g_opa: np.ndarray,
    yaws_rad: list[float],
    pitch_rad: float = 0.28,
    size: int = 512,
    bg_val: float = 0.95,
) -> list[np.ndarray]:
    """Pure-PyTorch CUDA 3D Gaussian Splatting renderer (works on all GPUs including Tesla T4 sm_75)."""
    xyz_t = torch.from_numpy(g_xyz).float().cuda()
    rgb_t = torch.from_numpy(g_rgb).float().cuda()
    opa_t = torch.from_numpy(g_opa).float().cuda()

    center = 0.5 * (xyz_t.min(dim=0).values + xyz_t.max(dim=0).values)
    span = float((xyz_t.max(dim=0).values - xyz_t.min(dim=0).values).max().item())
    xyz_c = (xyz_t - center) / max(span, 1e-6)

    vis = opa_t > 0.12
    xyz_c = xyz_c[vis]
    rgb_c = rgb_t[vis]

    cp, sp = float(np.cos(pitch_rad)), float(np.sin(pitch_rad))
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], device="cuda")

    frames = []
    for yaw in yaws_rad:
        cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
        ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], device="cuda")
        rot = rx @ ry
        v = xyz_c @ rot.T

        # Sort back-to-front along camera Z
        order = torch.argsort(v[:, 2], descending=False)
        vs = v[order]
        cs = rgb_c[order]

        px = ((vs[:, 0] * 0.80 + 0.5) * (size - 1)).round().long().clamp(2, size - 3)
        py = ((-vs[:, 1] * 0.80 + 0.5) * (size - 1)).round().long().clamp(2, size - 3)

        canvas = torch.full((size, size, 3), bg_val, dtype=torch.float32, device="cuda")
        # Two-pass splat: outer 5x5 soft halo then inner 3x3 crisp core
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                if abs(dx) == 2 or abs(dy) == 2:
                    canvas[py + dy, px + dx] = 0.5 * canvas[py + dy, px + dx] + 0.5 * cs
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                canvas[py + dy, px + dx] = cs

        frame = np.clip(canvas.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def bake_trellis_glb_t4(
    gs,
    mesh_res,
    simplify_ratio: float = 0.90,
    texture_size: int = 2048,
) -> tuple[trimesh.Trimesh, Image.Image, np.ndarray, np.ndarray, np.ndarray]:
    """Bake 360° 3D Gaussian appearance directly onto UV-unwrapped mesh in 3D space (T4-safe)."""
    # 1. Extract 3D Gaussian attributes and rotate from Z-up to Y-up
    rot_z2y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    g_xyz = gs.get_xyz.detach().cpu().numpy().astype(np.float32) @ rot_z2y
    g_rgb = (
        torch.clamp(gs._features_dc.squeeze(1) * SH_C0 + 0.5, 0.0, 1.0)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    g_opa = gs.get_opacity.squeeze(-1).detach().cpu().numpy().astype(np.float32)
    g_scale = gs.get_scaling.mean(dim=-1).detach().cpu().numpy().astype(np.float32)

    # 2. Extract and decimate mesh
    verts = mesh_res.vertices.detach().cpu().numpy().astype(np.float32) @ rot_z2y
    faces = mesh_res.faces.detach().cpu().numpy().astype(np.int32)

    if 0.0 < simplify_ratio < 1.0 and len(faces) > 10000:
        pv_faces = np.hstack([np.full((len(faces), 1), 3, dtype=np.int32), faces])
        poly = pv.PolyData(verts, pv_faces).decimate(simplify_ratio)
        verts = np.asarray(poly.points, dtype=np.float32)
        faces = np.asarray(poly.faces, dtype=np.int32).reshape(-1, 4)[:, 1:]

    try:
        mf = _meshfix.PyTMesh()
        mf.load_array(verts.astype(np.float64), faces.astype(np.int32))
        mf.fill_small_boundaries(nbe=32, refine=True)
        v_fix, f_fix = mf.return_arrays()
        if len(v_fix) > 100 and len(f_fix) > 100:
            verts, faces = v_fix.astype(np.float32), f_fix.astype(np.int32)
    except Exception:
        pass

    # 3. UV unwrap via xatlas
    vmapping, indices, uvs = xatlas.parametrize(verts, faces.astype(np.uint32))
    verts = verts[vmapping]
    faces = indices.astype(np.int32)
    uvs = uvs.astype(np.float32)

    # 4. Rasterize 3D surface coordinates into UV space on GPU and query 3D Gaussians via cKDTree
    H = W = int(texture_size)
    v_t = torch.from_numpy(verts).cuda()
    f_t = torch.from_numpy(faces).long().cuda()
    uv_t = torch.from_numpy(uvs).cuda()
    v0, v1, v2 = v_t[f_t[:, 0]], v_t[f_t[:, 1]], v_t[f_t[:, 2]]
    uv0, uv1, uv2 = uv_t[f_t[:, 0]], uv_t[f_t[:, 1]], uv_t[f_t[:, 2]]

    u_grid = torch.linspace(0.0, 1.0, 14, device="cuda")
    uu, vv = torch.meshgrid(u_grid, u_grid, indexing="ij")
    mb = (uu + vv) <= 1.0
    b0, b1 = uu[mb], vv[mb]
    b2 = 1.0 - b0 - b1

    pos_sum = torch.zeros((H * W, 3), dtype=torch.float32, device="cuda")
    cnt = torch.zeros((H * W,), dtype=torch.float32, device="cuda")
    F_num = len(faces)
    for s in range(0, F_num, 8000):
        e = min(s + 8000, F_num)
        p_uv = (
            b0[None, :, None] * uv0[s:e, None, :]
            + b1[None, :, None] * uv1[s:e, None, :]
            + b2[None, :, None] * uv2[s:e, None, :]
        )
        p_xyz = (
            b0[None, :, None] * v0[s:e, None, :]
            + b1[None, :, None] * v1[s:e, None, :]
            + b2[None, :, None] * v2[s:e, None, :]
        ).reshape(-1, 3)
        px = (p_uv[..., 0].reshape(-1) * (W - 1)).round().long().clamp(0, W - 1)
        py = ((1.0 - p_uv[..., 1].reshape(-1)) * (H - 1)).round().long().clamp(0, H - 1)
        idx = py * W + px
        pos_sum.scatter_add_(0, idx[:, None].expand(-1, 3), p_xyz)
        cnt.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

    mask_1d = (cnt > 0).cpu().numpy()
    valid_xyz = (pos_sum[cnt > 0] / cnt[cnt > 0, None]).cpu().numpy()
    del pos_sum, cnt, v_t, f_t, uv_t
    torch.cuda.empty_cache()

    tree = cKDTree(g_xyz)
    dists, idxs = tree.query(valid_xyz, k=12, workers=-1)
    sigma = np.maximum(g_scale[idxs], 0.004)
    w = g_opa[idxs] * np.exp(-0.5 * (dists / sigma) ** 2) + 1e-6
    w /= w.sum(axis=1, keepdims=True)
    valid_rgb = (g_rgb[idxs] * w[..., None]).sum(axis=1)

    tex = np.zeros((H * W, 3), dtype=np.uint8)
    tex[mask_1d] = np.clip(valid_rgb * 255.0, 0, 255).astype(np.uint8)
    tex = tex.reshape(H, W, 3)
    inpaint_mask = (~mask_1d.reshape(H, W)).astype(np.uint8)
    tex = cv2.inpaint(tex, inpaint_mask, 3, cv2.INPAINT_TELEA)
    tex_img = Image.fromarray(tex)

    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=0.85,
        metallicFactor=0.0,
        baseColorTexture=tex_img,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
    )
    out_mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
        process=False,
    )
    return out_mesh, tex_img, g_xyz, g_rgb, g_opa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/content/inputs")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--simplify", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_base = Path(
        args.output_root
        or os.environ.get("MESH_DATA_DIR", "/content/drive/MyDrive/MeshStudio/data")
    ) / "outputs"
    out_base.mkdir(parents=True, exist_ok=True)

    print("[1/4] Kiểm tra bộ xoá nền u2net (HuggingFace mirror)...", flush=True)
    ensure_u2net()

    from trellis.pipelines import TrellisImageTo3DPipeline

    print("[2/4] Đang tải Microsoft TRELLIS (microsoft/TRELLIS-image-large)...", flush=True)
    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    if "slat_decoder_rf" in pipe.models:
        del pipe.models["slat_decoder_rf"]
    gc.collect()
    torch.cuda.empty_cache()

    paths = [
        p for p in sorted(glob.glob(f"{args.input_dir}/*"))
        if Path(p).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    ]
    if not paths:
        print(f"⚠️ Không tìm thấy ảnh nào trong {args.input_dir}!")
        return

    for idx, path in enumerate(paths, 1):
        name = Path(path).stem
        slug = _slug(name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = out_base / f"{stamp}_{slug}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== [{idx}/{len(paths)}] Đang tạo 3D bằng Microsoft TRELLIS: {name} ===", flush=True)
        t0 = time.perf_counter()
        img = Image.open(path)
        img.save(out_dir / "source.png")

        pipe.cuda()
        outputs = pipe.run(
            img,
            seed=args.seed,
            formats=["gaussian", "mesh"],
            preprocess_image=True,
        )
        gs = outputs["gaussian"][0]
        mesh_res = outputs["mesh"][0]

        pipe.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[3/4] Đang nướng UV Texture 360° ({args.texture_size}x{args.texture_size}) & xuất GLB/OBJ...", flush=True)
        glb_mesh, tex_img, g_xyz, g_rgb, g_opa = bake_trellis_glb_t4(
            gs,
            mesh_res,
            simplify_ratio=args.simplify,
            texture_size=args.texture_size,
        )
        glb_path = out_dir / f"{slug}.glb"
        obj_path = out_dir / f"{slug}.obj"
        glb_mesh.export(glb_path)
        glb_mesh.export(obj_path)
        tex_img.save(out_dir / f"{slug}_texture.png")

        print("[4/4] Kết xuất 4 góc nhìn 360° & video xoay...", flush=True)
        snap = render_gaussian_views_cuda(
            g_xyz,
            g_rgb,
            g_opa,
            yaws_rad=[0.0, float(np.pi / 2), float(np.pi), float(3 * np.pi / 2)],
            size=512,
        )
        turnaround = Image.fromarray(np.concatenate(snap, axis=1))
        turnaround.save(out_dir / "turnaround.png")
        Image.fromarray(snap[0]).save(out_dir / "thumbnail.png")

        vid_yaws = np.linspace(0.0, float(2 * np.pi), 72, endpoint=False).tolist()
        vid_frames = render_gaussian_views_cuda(g_xyz, g_rgb, g_opa, yaws_rad=vid_yaws, size=512)
        imageio.mimsave(out_dir / "preview_360.mp4", vid_frames, fps=20)

        elapsed = round(time.perf_counter() - t0, 1)
        stats = {
            "dense_faces": int(mesh_res.faces.shape[0]),
            "final_faces": int(len(glb_mesh.faces)),
            "final_vertices": int(len(glb_mesh.vertices)),
            "texture_size": args.texture_size,
            "engine": "microsoft/TRELLIS-image-large",
        }
        meta = {
            "id": out_dir.name,
            "name": name,
            "slug": slug,
            "engine": "trellis_large",
            "created": stamp,
            "stats": stats,
            "timings": {"total_s": elapsed},
            "files": {
                "glb": str(glb_path),
                "obj": str(obj_path),
                "turnaround": str(out_dir / "turnaround.png"),
                "video": str(out_dir / "preview_360.mp4"),
            },
            "dir": str(out_dir),
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
        print(f"✅ Hoàn tất {name} trong {elapsed}s! Lưu tại: {out_dir}", flush=True)

        del outputs, gs, mesh_res, glb_mesh
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
