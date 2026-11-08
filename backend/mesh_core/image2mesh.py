"""Image -> 3D mesh supporting character-optimized 3072-latent engines & IS-Net segmentation:
- hunyuan3d_char:    Tencent Hunyuan3D-2.0 Full 1.1B (3,072 latents, octree 384, 18 steps)
- hunyuan3d_mini_hq: Tencent Hunyuan3D-2mini 0.6B    (3,072 latents via replace_vae=False, octree 384, 18 steps)
- hunyuan3d_mini:    Tencent Hunyuan3D-2mini 0.6B    (512 latents fast mode, octree 256, 10 steps)
- triposr:           StabilityAI TripoSR             (Fast fallback)
"""
from __future__ import annotations

import gc
import sys
import types

import numpy as np
import rembg
import torch
import trimesh
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, median_filter

from .config import (
    DEVICE,
    HUNYUAN3D_DIR,
    HUNYUAN3D_FULL_MODEL_ID,
    HUNYUAN3D_FULL_SUBFOLDER,
    HUNYUAN3D_MODEL_ID,
    HUNYUAN3D_OCTREE_RES,
    HUNYUAN3D_STEPS,
    HUNYUAN3D_SUBFOLDER,
    HUNYUAN3D_ULTRA_SUBFOLDER,
    MESH_3D_ENGINE,
    TRIPOSR_CHUNK_SIZE,
    TRIPOSR_DIR,
    TRIPOSR_MC_RESOLUTION,
    TRIPOSR_MODEL_ID,
)
from .model_manager import manager


# ---------------------------------------------------------------------------
# `torchmcubes` shim for TripoSR fallback on Windows without CUDA compiler
# ---------------------------------------------------------------------------
def _install_torchmcubes_shim() -> None:
    if "torchmcubes" in sys.modules:
        return
    try:
        import torchmcubes  # noqa: F401
        return
    except ImportError:
        pass

    from skimage import measure

    def marching_cubes(vol: torch.Tensor, thresh: float):
        v = vol.detach().float().cpu().numpy()
        try:
            verts, faces, _, _ = measure.marching_cubes(v, level=thresh)
        except (ValueError, RuntimeError):
            return torch.zeros((0, 3)), torch.zeros((0, 3), dtype=torch.long)
        verts = np.ascontiguousarray(verts[:, ::-1])
        faces = np.ascontiguousarray(faces[:, ::-1])
        return (
            torch.from_numpy(verts.astype(np.float32)),
            torch.from_numpy(faces.astype(np.int64)),
        )

    shim = types.ModuleType("torchmcubes")
    shim.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = shim


_install_torchmcubes_shim()
for _p in (HUNYUAN3D_DIR, TRIPOSR_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tsr.utils import remove_background, resize_foreground  # noqa: E402


# ---------------------------------------------------------------------------
# Model loaders registered with ModelManager
# ---------------------------------------------------------------------------
def _load_hunyuan3d_mini_fast():
    """0.6B model with 512-latent turbo VAE (fastest, best for simple fruits/props)."""
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        HUNYUAN3D_MODEL_ID,
        subfolder=HUNYUAN3D_SUBFOLDER,
        use_safetensors=True,
        device="cpu",
        dtype=dtype,
    )
    pipe.enable_flashvdm(mc_algo="mc", replace_vae=True)
    return pipe


def _load_hunyuan3d_mini_3072():
    """0.6B model keeping bundled 3,072-latent VAE (6x more detail for characters/limbs)."""
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        HUNYUAN3D_MODEL_ID,
        subfolder=HUNYUAN3D_SUBFOLDER,
        use_safetensors=True,
        device="cpu",
        dtype=dtype,
    )
    pipe.enable_flashvdm(mc_algo="mc", replace_vae=False)
    return pipe


def _load_hunyuan3d_full():
    """1.1B Full model with 3,072-latent turbo VAE (highest anatomical & geometric quality)."""
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        HUNYUAN3D_FULL_MODEL_ID,
        subfolder=HUNYUAN3D_FULL_SUBFOLDER,
        use_safetensors=True,
        device="cpu",
        dtype=dtype,
    )
    pipe.enable_flashvdm(mc_algo="mc", replace_vae=True)
    return pipe


def _load_hunyuan3d_ultra():
    """Non-distilled 1.1B model (50 steps + classifier-free guidance). Needs ~12+ GB VRAM (Colab T4)."""
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        HUNYUAN3D_FULL_MODEL_ID,
        subfolder=HUNYUAN3D_ULTRA_SUBFOLDER,
        use_safetensors=True,
        device=DEVICE,
        dtype=dtype,
    )
    pipe.enable_flashvdm(mc_algo="mc", replace_vae=False)
    return pipe


def _load_triposr():
    from tsr.system import TSR

    model = TSR.from_pretrained(
        TRIPOSR_MODEL_ID, config_name="config.yaml", weight_name="model.ckpt"
    )
    model.renderer.set_chunk_size(TRIPOSR_CHUNK_SIZE)
    model.to(DEVICE)
    model.eval()
    return model


manager.register("hunyuan3d", _load_hunyuan3d_mini_fast)
manager.register("hunyuan3d_mini", _load_hunyuan3d_mini_fast)
manager.register("hunyuan3d_mini_hq", _load_hunyuan3d_mini_3072)
manager.register("hunyuan3d_full", _load_hunyuan3d_full)
manager.register("hunyuan3d_char", _load_hunyuan3d_full)
manager.register("hunyuan3d_ultra", _load_hunyuan3d_ultra)
manager.register("triposr", _load_triposr)

_rembg_session = None


def _ensure_rembg_model(model_name: str = "isnet-general-use") -> None:
    """Pre-download rembg ONNX weights from HuggingFace mirror when GitHub Releases times out (504)."""
    import os
    import shutil
    from pathlib import Path

    os.environ.setdefault("MODEL_CHECKSUM_DISABLED", "1")
    u2net_dir = Path(
        os.path.expanduser(os.getenv("U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net")))
    )
    u2net_dir.mkdir(parents=True, exist_ok=True)
    target = u2net_dir / f"{model_name}.onnx"
    if target.exists() and target.stat().st_size > 10_000_000:
        return
    if target.exists():
        target.unlink(missing_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(repo_id="tomjackson2023/rembg", filename=f"{model_name}.onnx")
        shutil.copyfile(cached, target)
    except Exception:
        pass


def prepare_image(image: Image.Image, foreground_ratio: float = 0.85) -> Image.Image:
    """Remove background using high-precision IS-Net (cached on C:) for clean character limbs/hair."""
    global _rembg_session
    if _rembg_session is None:
        try:
            _ensure_rembg_model("isnet-general-use")
            _rembg_session = rembg.new_session("isnet-general-use", providers=["CPUExecutionProvider"])
        except Exception:
            _ensure_rembg_model("u2net")
            _rembg_session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
    img = remove_background(image.convert("RGB"), _rembg_session)
    img = resize_foreground(img, foreground_ratio)
    return img.convert("RGBA")


def _rgba_to_grey_rgb(image: Image.Image) -> Image.Image:
    """Composite RGBA onto neutral 50% grey background (for TripoSR fallback)."""
    arr = np.asarray(image.convert("RGBA")).astype(np.float32) / 255.0
    rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * 0.5
    return Image.fromarray((rgb * 255.0).astype(np.uint8))


def colorize_mesh_from_rgba(mesh: trimesh.Trimesh, prepared_rgba: Image.Image) -> trimesh.Trimesh:
    """Project 2D colors onto the 3D mesh with 1:1 silhouette alignment so character limbs & face align."""
    rgba = np.asarray(prepared_rgba.convert("RGBA"))
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3].astype(np.float32) / 255.0

    # Erode valid color mask by 2px so thin character fingers/limbs retain their colors while stripping edge halo
    fg_color = binary_erosion((alpha > 0.80) & (rgb.max(axis=-1) > 0.05), iterations=2)
    if not fg_color.any():
        fg_color = alpha > 0.5
    if not fg_color.any():
        fg_color = np.ones_like(alpha, dtype=bool)

    # Fill exterior (and gaps between arms/legs) with nearest valid interior foreground pixel
    _, (iy, ix) = distance_transform_edt(~fg_color, return_indices=True)
    rgb_filled = rgb[iy, ix]

    # Soften harsh white specular glare while keeping facial/clothing detail crisp
    max_c = rgb_filled.max(axis=-1, keepdims=True)
    min_c = rgb_filled.min(axis=-1, keepdims=True)
    sat = (max_c - min_c) / np.maximum(max_c, 1e-4)
    glare = np.clip((max_c - 0.86) / 0.14, 0.0, 1.0) * np.clip((0.28 - sat) / 0.28, 0.0, 1.0)
    if glare.max() > 0:
        blurred = gaussian_filter(rgb_filled, sigma=(8.0, 8.0, 0.0))
        rgb_filled = rgb_filled * (1.0 - 0.50 * glare) + blurred * (0.50 * glare)

    rgb_filled = np.clip(gaussian_filter(rgb_filled, sigma=(0.5, 0.5, 0.0)), 0.0, 1.0)

    # Bounding box of geometric silhouette
    fg_box = alpha > 0.5
    if fg_box.any():
        ys, xs = np.nonzero(fg_box)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    else:
        y0, y1, x0, x1 = 0, rgb.shape[0] - 1, 0, rgb.shape[1] - 1

    trimesh.repair.fix_normals(mesh)
    v = mesh.vertices
    vn = mesh.vertex_normals
    vmin, vmax = v.min(axis=0), v.max(axis=0)

    u = (v[:, 0] - vmin[0]) / max(float(vmax[0] - vmin[0]), 1e-6)
    w = 1.0 - (v[:, 1] - vmin[1]) / max(float(vmax[1] - vmin[1]), 1e-6)
    z_norm = (v[:, 2] - vmin[2]) / max(float(vmax[2] - vmin[2]), 1e-6)

    # Local normal nudge for front-facing 1:1 silhouette alignment
    side_factor = 1.0 - np.abs(vn[:, 2])
    u_in = np.clip(u - vn[:, 0] * 0.015 * side_factor, 0.005, 0.995)
    w_in = np.clip(w + vn[:, 1] * 0.015 * side_factor, 0.005, 0.995)

    px_front = np.clip((x0 + u_in * (x1 - x0)).astype(int), 0, rgb.shape[1] - 1)
    py_front = np.clip((y0 + w_in * (y1 - y0)).astype(int), 0, rgb.shape[0] - 1)
    col_front = rgb_filled[py_front, px_front]

    # Triplanar sampling for sides (±X) and back (-Z) from the clean interior body region
    # so side views (90°/270°) never stretch 1px edge lines horizontally, and the back (180°)
    # stays crisp without duplicating small top/front protrusions (like a watermelon stem).
    w_body = np.where(w_in > 0.16, 0.22 + 0.68 * w_in, w_in)
    py_body = np.clip((y0 + w_body * (y1 - y0)).astype(int), 0, rgb.shape[0] - 1)

    u_side = 0.18 + 0.64 * z_norm
    px_side = np.clip((x0 + u_side * (x1 - x0)).astype(int), 0, rgb.shape[1] - 1)
    col_side = rgb_filled[py_body, px_side]

    u_back = 0.18 + 0.64 * (1.0 - u_in)
    px_back = np.clip((x0 + u_back * (x1 - x0)).astype(int), 0, rgb.shape[1] - 1)
    col_back = rgb_filled[py_body, px_back]

    # Smooth triplanar weights based on surface normal direction
    w_fz = np.clip(vn[:, 2], 0.0, 1.0) ** 2
    w_bz = np.clip(-vn[:, 2], 0.0, 1.0) ** 2
    w_sx = (vn[:, 0] ** 2) + 0.35 * (vn[:, 1] ** 2) * (1.0 - w_fz)
    w_sum = np.maximum(w_fz + w_bz + w_sx, 1e-6)

    blended = (
        col_front * (w_fz / w_sum)[:, None]
        + col_side * (w_sx / w_sum)[:, None]
        + col_back * (w_bz / w_sum)[:, None]
    )

    vcol = (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vcol)
    return mesh


def _hunyuan_image_to_mesh(
    prepared_rgba: Image.Image,
    model_key: str = "hunyuan3d_char",
    steps: int = 18,
    octree_resolution: int = 384,
    max_facenum: int = 250000,
    seed: int = 42,
) -> trimesh.Trimesh:
    """Run Hunyuan3D (Full 1.1B 3072-latent or Mini 0.6B) with sequential GPU staging."""
    from hy3dgen.shapegen.pipelines import retrieve_timesteps

    try:
        pipe = manager.get(model_key)
    except Exception as e:
        print(f"[MeshStudio] Chuyển từ {model_key} sang hunyuan3d_mini_hq (3072 latents): {e}", flush=True)
        pipe = manager.get("hunyuan3d_mini_hq")

    dev = torch.device(DEVICE)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    if prepared_rgba.mode != "RGBA":
        prepared_rgba = prepare_image(prepared_rgba)

    with torch.inference_mode():
        # Stage 1: Image conditioning (DINOv2-giant)
        cond_in = pipe.prepare_image(prepared_rgba)
        im_t = cond_in.pop("image").to(dev, dtype=dtype)
        pipe.conditioner.to(dev, dtype=dtype)
        cond = pipe.encode_cond(
            image=im_t,
            additional_cond_inputs=cond_in,
            do_classifier_free_guidance=False,
            dual_guidance=False,
        )
        pipe.conditioner.to("cpu")
        del im_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stage 2: DiT Flow Matching diffusion
        pipe.model.to(dev, dtype=dtype)
        pipe.device = dev
        sigmas = np.linspace(0, 1, steps)
        timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, dev, sigmas=sigmas)
        generator = torch.Generator(device=dev).manual_seed(seed)
        latents = pipe.prepare_latents(1, dtype, dev, generator)
        guidance = torch.tensor([5.5], device=dev, dtype=dtype)
        for t in timesteps:
            timestep = t.expand(1).to(dtype) / pipe.scheduler.config.num_train_timesteps
            v = pipe.model(latents, timestep, cond, guidance=guidance)
            latents = pipe.scheduler.step(v, t, latents).prev_sample
        pipe.model.to("cpu")
        del cond
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stage 3: FlashVDM hierarchical octree volume decoding + Marching Cubes
        pipe.vae.to(dev, dtype=dtype)
        mesh = pipe._export(
            latents,
            "trimesh",
            1.01,
            0.0,
            max_facenum,
            octree_resolution,
            "mc",
            enable_pbar=False,
        )[0]
        pipe.vae.to("cpu")
        del latents
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mesh = colorize_mesh_from_rgba(mesh, prepared_rgba)
    return mesh


def _hunyuan_ultra_image_to_mesh(
    prepared_rgba: Image.Image,
    steps: int = 50,
    guidance: float = 5.0,
    octree_resolution: int = 512,
    seed: int = 42,
) -> trimesh.Trimesh:
    """Highest-quality path: full 1.1B model, 50 steps with CFG, 512^3 grid. Falls back to 384^3 on OOM."""
    pipe = manager.get("hunyuan3d_ultra")
    if prepared_rgba.mode != "RGBA":
        prepared_rgba = prepare_image(prepared_rgba)

    def run(res: int):
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        return pipe(
            image=prepared_rgba,
            num_inference_steps=steps,
            guidance_scale=guidance,
            dual_guidance=False,
            generator=gen,
            octree_resolution=res,
            num_chunks=20000,
            output_type="trimesh",
            enable_pbar=False,
        )[0]

    try:
        mesh = run(octree_resolution)
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        print("[MeshStudio] Het VRAM o luoi 512^3, thu lai 384^3", flush=True)
        mesh = run(384)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return colorize_mesh_from_rgba(mesh, prepared_rgba)


def _triposr_image_to_mesh(prepared_rgba: Image.Image, resolution: int = TRIPOSR_MC_RESOLUTION) -> trimesh.Trimesh:
    """Fallback TripoSR generator converted to Y-up, front = +Z."""
    model = manager.get("triposr")
    grey_rgb = _rgba_to_grey_rgb(prepared_rgba)
    with torch.no_grad():
        scene_codes = model([grey_rgb], device=DEVICE)
        mesh = model.extract_mesh(scene_codes, True, resolution=resolution)[0]
    del scene_codes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
    return mesh


def image_to_mesh(
    prepared: Image.Image,
    resolution: int = HUNYUAN3D_OCTREE_RES,
    engine: str = MESH_3D_ENGINE,
) -> trimesh.Trimesh:
    """Returns a dense, vertex-coloured mesh in Unity frame (Y-up, front = +Z)."""
    eng = (engine or MESH_3D_ENGINE).lower()
    if eng == "triposr":
        return _triposr_image_to_mesh(prepared, resolution=resolution)
    if eng == "hunyuan3d_ultra":
        return _hunyuan_ultra_image_to_mesh(prepared)
    if eng in ("hunyuan3d_char", "hunyuan3d_full"):
        return _hunyuan_image_to_mesh(
            prepared,
            model_key="hunyuan3d_full",
            steps=18,
            octree_resolution=384,
            max_facenum=250000,
        )
    if eng == "hunyuan3d_mini_hq":
        return _hunyuan_image_to_mesh(
            prepared,
            model_key="hunyuan3d_mini_hq",
            steps=18,
            octree_resolution=384,
            max_facenum=250000,
        )
    return _hunyuan_image_to_mesh(
        prepared,
        model_key="hunyuan3d_mini",
        steps=HUNYUAN3D_STEPS,
        octree_resolution=resolution,
        max_facenum=250000,
    )
