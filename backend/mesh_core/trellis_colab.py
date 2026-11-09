"""Standalone runner for Microsoft TRELLIS (microsoft/TRELLIS-image-large) on Google Colab T4."""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import re
import shutil
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

import imageio
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image


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
    # Remove unused radiance_field decoder to save RAM & VRAM on Colab T4
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

        # Offload diffusion models to CPU before 2048x2048 UV texture baking so T4 15GB VRAM never OOMs
        pipe.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        print("[3/4] Kết xuất 4 góc nhìn 360° & video xoay...", flush=True)
        snap = render_utils.render_snapshot(
            gs, resolution=512, bg_color=(0.95, 0.95, 0.97)
        )["color"]
        turnaround = Image.fromarray(np.concatenate(snap, axis=1))
        turnaround.save(out_dir / "turnaround.png")
        Image.fromarray(snap[0]).save(out_dir / "thumbnail.png")

        vid_color = render_utils.render_video(gs, resolution=512, num_frames=80)["color"]
        vid_norm = render_utils.render_video(mesh_res, resolution=512, num_frames=80)["normal"]
        vid = [np.concatenate([c, n], axis=1) for c, n in zip(vid_color, vid_norm)]
        imageio.mimsave(out_dir / "preview_360.mp4", vid, fps=20)

        print(f"[4/4] Đang nướng UV Texture 360° ({args.texture_size}x{args.texture_size}) & xuất GLB/OBJ...", flush=True)
        glb_mesh = postprocessing_utils.to_glb(
            gs,
            mesh_res,
            simplify=args.simplify,
            texture_size=args.texture_size,
            verbose=True,
        )
        glb_path = out_dir / f"{slug}.glb"
        obj_path = out_dir / f"{slug}.obj"
        glb_mesh.export(glb_path)
        glb_mesh.export(obj_path)

        try:
            tex_img = glb_mesh.visual.material.baseColorTexture
            if tex_img is not None:
                tex_img.save(out_dir / f"{slug}_texture.png")
        except Exception:
            pass

        elapsed = round(time.perf_counter() - t0, 1)
        stats = {
            "dense_faces": int(mesh_res.faces.shape[0]),
            "final_faces": int( len(glb_mesh.faces) ),
            "final_vertices": int( len(glb_mesh.vertices) ),
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
