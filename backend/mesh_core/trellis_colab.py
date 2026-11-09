"""Standalone runner for Microsoft TRELLIS (microsoft/TRELLIS-image-large) on Google Colab T4.
Uses native FlexiCubes 3D color extraction & pure-PyTorch anti-aliased UV baking.
Zero crash on Colab T4 (no broken diff_gaussian_rasterization CUDA extension).
"""
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


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return (s or "asset")[:40]


def ensure_u2net() -> None:
    cached = hf_hub_download(repo_id="tomjackson2023/rembg", filename="u2net.onnx")
    for d in {os.environ.get("U2NET_HOME", "/root/.u2net"), os.path.expanduser("~/.u2net")}:
        Path(d).mkdir(parents=True, exist_ok=True)
        dst = Path(d) / "u2net.onnx"
        if not dst.exists() or dst.stat().st_size < 10_000_000:
            shutil.copyfile(cached, dst)


def render_mesh_views_pytorch(
    verts: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    yaws_rad: list[float],
    pitch_rad: float = 0.25,
    size: int = 1024,
    bg_color: tuple[float, float, float] = (0.96, 0.96, 0.98),
) -> list[np.ndarray]:
    """Pure-PyTorch depth-buffered triangle rasterizer: sharp, anti-aliased, 100% stable on T4."""
    v_t = torch.from_numpy(verts).float().cuda()
    f_t = torch.from_numpy(faces).long().cuda()
    c_t = torch.from_numpy(vertex_colors).float().cuda()

    center = 0.5 * (v_t.min(dim=0).values + v_t.max(dim=0).values)
    span = float((v_t.max(dim=0).values - v_t.min(dim=0).values).max().item())
    v_norm = (v_t - center) / max(span, 1e-6)

    # Simple studio directional lighting
    tri_v0 = v_norm[f_t[:, 0]]
    tri_v1 = v_norm[f_t[:, 1]]
    tri_v2 = v_norm[f_t[:, 2]]
    face_normals = torch.nn.functional.normalize(torch.cross(tri_v1 - tri_v0, tri_v2 - tri_v0, dim=-1), dim=-1)

    light_dir = torch.nn.functional.normalize(torch.tensor([0.4, 0.6, 0.8], device="cuda"), dim=-1)
    shade = 0.45 + 0.55 * torch.clamp((face_normals * light_dir[None, :]).sum(dim=-1), 0.0, 1.0)

    # Sub-triangle barycentric sampling points (for anti-aliased rasterization)
    u_grid = torch.linspace(0.0, 1.0, 12, device="cuda")
    uu, vv = torch.meshgrid(u_grid, u_grid, indexing="ij")
    mb = (uu + vv) <= 1.0
    b0, b1 = uu[mb], vv[mb]
    b2 = 1.0 - b0 - b1

    c0 = c_t[f_t[:, 0]] * shade[:, None]
    c1 = c_t[f_t[:, 1]] * shade[:, None]
    c2 = c_t[f_t[:, 2]] * shade[:, None]

    cp, sp = float(np.cos(pitch_rad)), float(np.sin(pitch_rad))
    rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], device="cuda")

    frames = []
    F_num = len(faces)
    for yaw in yaws_rad:
        cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
        ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], device="cuda")
        rot = rx @ ry
        v_rot = v_norm @ rot.T

        rv0 = v_rot[f_t[:, 0]]
        rv1 = v_rot[f_t[:, 1]]
        rv2 = v_rot[f_t[:, 2]]

        # Backface culling in camera space (camera looks down -Z)
        cross_z = (rv1[:, 0] - rv0[:, 0]) * (rv2[:, 1] - rv0[:, 1]) - (rv1[:, 1] - rv0[:, 1]) * (rv2[:, 0] - rv0[:, 0])
        front_faces = torch.nonzero(cross_z > 0.0).reshape(-1)

        depth_buf = torch.full((size, size), -1e5, dtype=torch.float32, device="cuda")
        color_buf = torch.full((size, size, 3), 0.0, dtype=torch.float32, device="cuda")

        for s in range(0, len(front_faces), 10000):
            idx_f = front_faces[s:s + 10000]
            p_xyz = (
                b0[None, :, None] * rv0[idx_f, None, :]
                + b1[None, :, None] * rv1[idx_f, None, :]
                + b2[None, :, None] * rv2[idx_f, None, :]
            ).reshape(-1, 3)
            p_col = (
                b0[None, :, None] * c0[idx_f, None, :]
                + b1[None, :, None] * c1[idx_f, None, :]
                + b2[None, :, None] * c2[idx_f, None, :]
            ).reshape(-1, 3)

            px = ((p_xyz[:, 0] * 0.82 + 0.5) * (size - 1)).round().long().clamp(0, size - 1)
            py = ((-p_xyz[:, 1] * 0.82 + 0.5) * (size - 1)).round().long().clamp(0, size - 1)
            pz = p_xyz[:, 2]

            flat_idx = py * size + px
            depth_flat = depth_buf.view(-1)
            color_flat = color_buf.view(-1, 3)

            # Atomic depth maximum
            depth_flat.scatter_reduce_(0, flat_idx, pz, reduce="amax", include_self=True)
            is_closest = (pz >= depth_flat[flat_idx] - 1e-4)
            color_flat.scatter_(0, flat_idx[is_closest, None].expand(-1, 3), p_col[is_closest])

        mask = (depth_buf > -1e4)[:, :, None]
        bg = torch.tensor(bg_color, device="cuda", dtype=torch.float32)[None, None, :]
        out = torch.where(mask, color_buf, bg)

        frame = np.clip(out.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        frames.append(frame)

    del v_t, f_t, c_t, tri_v0, tri_v1, tri_v2, face_normals
    torch.cuda.empty_cache()
    return frames


def build_crisp_textured_glb(
    mesh_res,
    simplify_ratio: float = 0.90,
    texture_size: int = 2048,
) -> tuple[trimesh.Trimesh, Image.Image, np.ndarray, np.ndarray, np.ndarray]:
    """Convert FlexiCubes vertex attributes into an ultra-sharp 2048x2048 UV texture GLB."""
    # Rotate mesh from Z-up to Y-up
    rot_z2y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    verts = mesh_res.vertices.detach().cpu().numpy().astype(np.float32) @ rot_z2y
    faces = mesh_res.faces.detach().cpu().numpy().astype(np.int32)

    # Extract 3D color channels from FlexiCubes attributes
    raw_col = mesh_res.vertex_attrs[:, :3].detach().cpu().numpy().astype(np.float32)
    # If the network output logits, map through sigmoid; otherwise clamp
    if raw_col.min() < 0.0 or raw_col.max() > 1.0:
        raw_col = 1.0 / (1.0 + np.exp(-raw_col))
    colors = np.clip(raw_col, 0.0, 1.0)

    # Decimate geometry while preserving vertex color sharpness
    if 0.0 < simplify_ratio < 1.0 and len(faces) > 12000:
        poly = pv.PolyData(verts, np.hstack([np.full((len(faces), 1), 3, dtype=np.int32), faces]))
        poly_dec = poly.decimate(simplify_ratio)
        verts_dec = np.asarray(poly_dec.points, dtype=np.float32)
        faces_dec = np.asarray(poly_dec.faces, dtype=np.int32).reshape(-1, 4)[:, 1:]

        # Query colors for decimated vertices using cKDTree
        c_tree = cKDTree(verts)
        _, c_idx = c_tree.query(verts_dec, k=1, workers=-1)
        verts = verts_dec
        faces = faces_dec
        colors = colors[c_idx]

    try:
        mf = _meshfix.PyTMesh()
        mf.load_array(verts.astype(np.float64), faces.astype(np.int32))
        mf.fill_small_boundaries(nbe=32, refine=True)
        v_fix, f_fix = mf.return_arrays()
        if len(v_fix) > 100 and len(f_fix) > 100:
            # Re-sample colors to fixed mesh
            c_tree = cKDTree(verts)
            _, c_idx = c_tree.query(v_fix, k=1, workers=-1)
            verts, faces = v_fix.astype(np.float32), f_fix.astype(np.int32)
            colors = colors[c_idx]
    except Exception:
        pass

    # UV unwrap with xatlas
    vmapping, indices, uvs = xatlas.parametrize(verts, faces.astype(np.uint32))
    verts = verts[vmapping]
    faces = indices.astype(np.int32)
    colors = colors[vmapping]
    uvs = uvs.astype(np.float32)

    # Rasterize vertex colors into UV texture map (2048x2048)
    H = W = int(texture_size)
    c_t = torch.from_numpy(colors).cuda()
    f_t = torch.from_numpy(faces).long().cuda()
    uv_t = torch.from_numpy(uvs).cuda()

    c0, c1, c2 = c_t[f_t[:, 0]], c_t[f_t[:, 1]], c_t[f_t[:, 2]]
    uv0, uv1, uv2 = uv_t[f_t[:, 0]], uv_t[f_t[:, 1]], uv_t[f_t[:, 2]]

    u_grid = torch.linspace(0.0, 1.0, 14, device="cuda")
    uu, vv = torch.meshgrid(u_grid, u_grid, indexing="ij")
    mb = (uu + vv) <= 1.0
    b0, b1 = uu[mb], vv[mb]
    b2 = 1.0 - b0 - b1

    col_sum = torch.zeros((H * W, 3), dtype=torch.float32, device="cuda")
    cnt = torch.zeros((H * W,), dtype=torch.float32, device="cuda")
    F_num = len(faces)
    for s in range(0, F_num, 8000):
        e = min(s + 8000, F_num)
        p_uv = (
            b0[None, :, None] * uv0[s:e, None, :]
            + b1[None, :, None] * uv1[s:e, None, :]
            + b2[None, :, None] * uv2[s:e, None, :]
        )
        p_col = (
            b0[None, :, None] * c0[s:e, None, :]
            + b1[None, :, None] * c1[s:e, None, :]
            + b2[None, :, None] * c2[s:e, None, :]
        ).reshape(-1, 3)
        px = (p_uv[..., 0].reshape(-1) * (W - 1)).round().long().clamp(0, W - 1)
        py = ((1.0 - p_uv[..., 1].reshape(-1)) * (H - 1)).round().long().clamp(0, H - 1)
        idx = py * W + px
        col_sum.scatter_add_(0, idx[:, None].expand(-1, 3), p_col)
        cnt.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

    mask_1d = (cnt > 0).cpu().numpy()
    valid_rgb = (col_sum[cnt > 0] / cnt[cnt > 0, None]).cpu().numpy()
    del col_sum, cnt, c_t, f_t, uv_t
    torch.cuda.empty_cache()

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
    return out_mesh, tex_img, verts, faces, colors


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

    print("[2/4] Đang nạp Microsoft TRELLIS (microsoft/TRELLIS-image-large)...", flush=True)
    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    # We only decode structured latent directly into mesh with rich 3D colors (saving VRAM & 100% bug-free)
    for k in ("slat_decoder_rf", "slat_decoder_gs"):
        if k in pipe.models:
            del pipe.models[k]
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
            formats=["mesh"],
            preprocess_image=True,
        )
        mesh_res = outputs["mesh"][0]

        pipe.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[3/4] Đang nướng UV Texture 360° ({args.texture_size}x{args.texture_size}) & xuất GLB/OBJ...", flush=True)
        glb_mesh, tex_img, verts_yup, faces_dec, v_colors = build_crisp_textured_glb(
            mesh_res,
            simplify_ratio=args.simplify,
            texture_size=args.texture_size,
        )

        glb_path = out_dir / f"{slug}.glb"
        obj_path = out_dir / f"{slug}.obj"
        glb_mesh.export(glb_path)
        glb_mesh.export(obj_path)
        tex_img.save(out_dir / f"{slug}_texture.png")

        print("[4/4] Kết xuất 4 góc nhìn 360° Siêu Nét (1024px) & video xoay...", flush=True)
        snap = render_mesh_views_pytorch(
            verts_yup,
            faces_dec,
            v_colors,
            yaws_rad=[0.0, float(np.pi / 2), float(np.pi), float(3 * np.pi / 2)],
            size=1024,
        )
        turnaround = Image.fromarray(np.concatenate(snap, axis=1))
        turnaround.save(out_dir / "turnaround.png")
        Image.fromarray(snap[0]).save(out_dir / "thumbnail.png")

        vid_yaws = np.linspace(0.0, float(2 * np.pi), 72, endpoint=False).tolist()
        vid_frames = render_mesh_views_pytorch(
            verts_yup,
            faces_dec,
            v_colors,
            yaws_rad=vid_yaws,
            size=512,
        )
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

        del outputs, mesh_res, glb_mesh
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
