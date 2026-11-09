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
import utils3d
import xatlas
from huggingface_hub import hf_hub_download
from PIL import Image
from pymeshfix import _meshfix

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


def bake_multiview_gaussian_to_glb(
    gs,
    mesh_res,
    render_utils,
    simplify_ratio: float = 0.90,
    texture_size: int = 2048,
) -> tuple[trimesh.Trimesh, Image.Image]:
    """Multi-view projection of 1024x1024 3D Gaussian Splatting renders onto 2048x2048 UV texture."""
    verts = mesh_res.vertices.detach().cpu().numpy().astype(np.float32)
    faces = mesh_res.faces.detach().cpu().numpy().astype(np.int32)

    if 0.0 < simplify_ratio < 1.0 and len(faces) > 12000:
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

    vmapping, indices, uvs = xatlas.parametrize(verts, faces.astype(np.uint32))
    verts = verts[vmapping]
    faces = indices.astype(np.int32)
    uvs = uvs.astype(np.float32)

    # Compute vertex normals in Z-up space before rotating to Y-up
    tmp_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    trimesh.repair.fix_normals(tmp_mesh)
    v_normals = tmp_mesh.vertex_normals.astype(np.float32)

    # Render 36 views at 1024x1024 using true anisotropic 3D Gaussian Splatting (r=2, fov=40)
    observations, extrinsics, intrinsics = render_utils.render_multiview(
        gs, resolution=1024, nviews=36
    )

    # Dense barycentric sampling of 3D positions & normals into 2048x2048 UV space
    H = W = int(texture_size)
    v_t = torch.from_numpy(verts).cuda()
    n_t = torch.from_numpy(v_normals).cuda()
    f_t = torch.from_numpy(faces).long().cuda()
    uv_t = torch.from_numpy(uvs).cuda()

    v0, v1, v2 = v_t[f_t[:, 0]], v_t[f_t[:, 1]], v_t[f_t[:, 2]]
    n0, n1, n2 = n_t[f_t[:, 0]], n_t[f_t[:, 1]], n_t[f_t[:, 2]]
    uv0, uv1, uv2 = uv_t[f_t[:, 0]], uv_t[f_t[:, 1]], uv_t[f_t[:, 2]]

    u_grid = torch.linspace(0.0, 1.0, 20, device="cuda")
    uu, vv = torch.meshgrid(u_grid, u_grid, indexing="ij")
    mb = (uu + vv) <= 1.0
    b0, b1 = uu[mb], vv[mb]
    b2 = 1.0 - b0 - b1

    pos_sum = torch.zeros((H * W, 3), dtype=torch.float32, device="cuda")
    nrm_sum = torch.zeros((H * W, 3), dtype=torch.float32, device="cuda")
    cnt = torch.zeros((H * W,), dtype=torch.float32, device="cuda")
    F_num = len(faces)
    for s in range(0, F_num, 6000):
        e = min(s + 6000, F_num)
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
        p_nrm = (
            b0[None, :, None] * n0[s:e, None, :]
            + b1[None, :, None] * n1[s:e, None, :]
            + b2[None, :, None] * n2[s:e, None, :]
        ).reshape(-1, 3)
        px = (p_uv[..., 0].reshape(-1) * (W - 1)).round().long().clamp(0, W - 1)
        py = ((1.0 - p_uv[..., 1].reshape(-1)) * (H - 1)).round().long().clamp(0, H - 1)
        idx = py * W + px
        pos_sum.scatter_add_(0, idx[:, None].expand(-1, 3), p_xyz)
        nrm_sum.scatter_add_(0, idx[:, None].expand(-1, 3), p_nrm)
        cnt.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

    valid_mask = cnt > 0
    pts_xyz = pos_sum[valid_mask] / cnt[valid_mask, None]
    pts_nrm = torch.nn.functional.normalize(nrm_sum[valid_mask], dim=-1)
    del pos_sum, nrm_sum, cnt

    # Project 3D UV texels into each of the 36 rendered 1024x1024 Gaussian views
    M = pts_xyz.shape[0]
    pts_homo = torch.cat([pts_xyz, torch.ones((M, 1), device="cuda")], dim=-1)
    col_acc = torch.zeros((M, 3), dtype=torch.float32, device="cuda")
    w_acc = torch.zeros((M, 1), dtype=torch.float32, device="cuda")

    for obs_np, extr, intr in zip(observations, extrinsics, intrinsics):
        obs_t = torch.from_numpy(obs_np).float().cuda() / 255.0  # (1024, 1024, 3)
        oh, ow = obs_t.shape[:2]
        extr_t = extr.float().cuda()
        intr_t = intr.float().cuda()

        cam_pos = torch.inverse(extr_t)[:3, 3]
        view_dir = torch.nn.functional.normalize(cam_pos[None, :] - pts_xyz, dim=-1)
        cos_sim = (pts_nrm * view_dir).sum(dim=-1, keepdim=True)
        front = cos_sim > 0.15

        cam_pts = pts_homo @ extr_t.T  # (M, 4)
        z_cam = cam_pts[:, 2:3].clamp(min=1e-4)
        u_proj = (cam_pts[:, 0:1] / z_cam) * intr_t[0, 0] + intr_t[0, 2]
        v_proj = (cam_pts[:, 1:2] / z_cam) * intr_t[1, 1] + intr_t[1, 2]

        in_bounds = front & (u_proj >= 0.0) & (u_proj <= 1.0) & (v_proj >= 0.0) & (v_proj <= 1.0)
        grid = torch.stack([u_proj * 2.0 - 1.0, v_proj * 2.0 - 1.0], dim=-1).view(1, M, 1, 2)
        sampled = torch.nn.functional.grid_sample(
            obs_t.permute(2, 0, 1).unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).view(3, M).T  # (M, 3)

        non_black = sampled.max(dim=-1, keepdim=True).values > 0.02
        w = torch.where(in_bounds & non_black, (cos_sim.clamp(min=0.0) ** 6), torch.zeros_like(cos_sim))
        col_acc += sampled * w
        w_acc += w

    valid_rgb = (col_acc / w_acc.clamp(min=1e-6)).cpu().numpy()
    mask_1d = valid_mask.cpu().numpy()

    tex = np.zeros((H * W, 3), dtype=np.uint8)
    tex[mask_1d] = np.clip(valid_rgb * 255.0, 0, 255).astype(np.uint8)
    tex = tex.reshape(H, W, 3)
    inpaint_mask = (~mask_1d.reshape(H, W)).astype(np.uint8)
    tex = cv2.inpaint(tex, inpaint_mask, 5, cv2.INPAINT_TELEA)

    # Rotate mesh from Z-up to Y-up for Unity / Blender / Showcase
    rot_z2y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    verts_yup = verts @ rot_z2y
    tex_img = Image.fromarray(tex)
    material = trimesh.visual.material.PBRMaterial(
        roughnessFactor=0.85,
        metallicFactor=0.0,
        baseColorTexture=tex_img,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
    )
    out_mesh = trimesh.Trimesh(
        vertices=verts_yup,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uvs, material=material),
        process=False,
    )
    return out_mesh, tex_img


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
    from trellis.utils import postprocessing_utils, render_utils

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

        print("[3/4] Kết xuất 4 góc nhìn 360° Siêu Nét (1024px) & video xoay...", flush=True)
        # Crucial fix: pass r=2, fov=40 (default r=10, fov=8 in render_snapshot causes tile overflow!)
        snap = render_utils.render_snapshot(
            gs, resolution=1024, bg_color=(0.96, 0.96, 0.98), r=2, fov=40
        )["color"]
        turnaround = Image.fromarray(np.concatenate(snap, axis=1))
        turnaround.save(out_dir / "turnaround.png")
        Image.fromarray(snap[0]).save(out_dir / "thumbnail.png")

        vid_color = render_utils.render_video(
            gs, resolution=512, bg_color=(0.96, 0.96, 0.98), num_frames=80, r=2, fov=40
        )["color"]
        imageio.mimsave(out_dir / "preview_360.mp4", vid_color, fps=20)

        print(f"[4/4] Đang nướng UV Texture 360° ({args.texture_size}x{args.texture_size}) & xuất GLB/OBJ...", flush=True)
        try:
            glb_mesh = postprocessing_utils.to_glb(
                gs,
                mesh_res,
                simplify=args.simplify,
                texture_size=args.texture_size,
                verbose=True,
            )
            tex_img = glb_mesh.visual.material.baseColorTexture
        except Exception as e:
            print(f"[T4 Fallback] Chuyển sang bộ nướng Multi-View 36 góc T4 ({e})...", flush=True)
            torch.cuda.empty_cache()
            glb_mesh, tex_img = bake_multiview_gaussian_to_glb(
                gs,
                mesh_res,
                render_utils,
                simplify_ratio=args.simplify,
                texture_size=args.texture_size,
            )

        glb_path = out_dir / f"{slug}.glb"
        obj_path = out_dir / f"{slug}.obj"
        glb_mesh.export(glb_path)
        glb_mesh.export(obj_path)
        if tex_img is not None:
            tex_img.save(out_dir / f"{slug}_texture.png")

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
