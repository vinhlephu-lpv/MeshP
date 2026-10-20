"""End-to-end pipeline: text/image -> game-ready asset folder."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image

from .config import MESH_3D_ENGINE, OUTPUT_DIR
from .export import export_asset, render_thumbnail, render_turnaround
from .model_manager import manager
from .postprocess import PostprocessOptions, make_game_asset
from .presets import PRESETS, build_prompt

ProgressFn = Callable[[str, float], None]


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return (s or "asset")[:40]


def _noop(stage: str, pct: float) -> None:
    try:
        print(f"[{pct:5.1f}%] {stage}", flush=True)
    except UnicodeEncodeError:
        print(f"[{pct:5.1f}%] {stage.encode('ascii', 'replace').decode('ascii')}", flush=True)



def text_to_images(subject: str, preset_key: str, num_images: int = 4,
                   seed: Optional[int] = None) -> list[Image.Image]:
    from .text2img import generate_images

    prompt = build_prompt(subject, preset_key)
    return generate_images(prompt, num_images=num_images, seed=seed)


def image_to_asset(image: Image.Image, name: str, preset_key: str,
                   opts: Optional[PostprocessOptions] = None,
                   progress: ProgressFn = _noop,
                   engine: str = MESH_3D_ENGINE) -> dict:
    from .image2mesh import image_to_mesh, prepare_image

    preset = PRESETS[preset_key]
    if opts is None:
        opts = PostprocessOptions(
            target_faces=preset.target_faces,
            palette_colors=preset.palette_colors,
            target_height_m=preset.target_height_m,
        )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slug(name)
    out_dir = OUTPUT_DIR / f"{stamp}_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = {}

    t = time.perf_counter()
    progress("Xoá nền ảnh", 5)
    image.save(out_dir / "source.png")
    prepared = prepare_image(image)
    prepared.save(out_dir / "prepared.png")
    timings["prepare_s"] = round(time.perf_counter() - t, 2)

    engine_labels = {
        "hunyuan3d_mini": "Hunyuan3D-2 Mini Turbo (0.6B)",
        "hunyuan3d_mini_hq": "Hunyuan3D-2 Mini Ultra (384³)",
        "hunyuan3d_full": "Hunyuan3D-2.0 Full Turbo (1.1B)",
        "triposr": "TripoSR",
    }
    eng_label = engine_labels.get(engine, engine)

    t = time.perf_counter()
    progress(f"Dựng hình 3D ({eng_label})", 20)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    dense = image_to_mesh(prepared, engine=engine)
    dense.export(out_dir / "dense.ply")
    timings["hunyuan3d_s"] = round(time.perf_counter() - t, 2)
    vram = manager.vram_report()

    t = time.perf_counter()
    progress("Tối ưu low-poly & tô màu", 70)
    final, texture, stats = make_game_asset(dense, opts, slug, legacy_triposr=False)
    timings["postprocess_s"] = round(time.perf_counter() - t, 2)

    progress("Xuất file", 90)
    files = export_asset(final, texture, out_dir, slug)
    render_thumbnail(final).save(out_dir / "thumbnail.png")
    render_turnaround(final).save(out_dir / "turnaround.png")

    meta = {
        "id": out_dir.name,
        "name": name, "slug": slug, "preset": preset_key, "created": stamp,
        "engine": engine,
        "options": asdict(opts), "stats": stats, "timings": timings, "vram": vram,
        "files": files, "dir": str(out_dir),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    progress("Hoàn tất", 100)
    return meta


def reprocess_asset(asset_id: str, opts: PostprocessOptions,
                    progress: ProgressFn = _noop) -> dict:
    """Re-run low-poly decimation, palette quantization and export from saved dense.ply."""
    import trimesh

    out_dir = OUTPUT_DIR / asset_id
    meta_path = out_dir / "meta.json"
    dense_path = out_dir / "dense.ply"
    if not meta_path.exists() or not dense_path.exists():
        raise FileNotFoundError(f"Asset {asset_id} missing meta.json or dense.ply")

    meta = json.loads(meta_path.read_text("utf-8"))
    slug = meta.get("slug", "asset")
    eng = str(meta.get("engine", ""))
    is_legacy = not eng.startswith("hunyuan3d") and eng != "triposr" and "hunyuan3d_s" not in meta.get("timings", {})

    t = time.perf_counter()
    progress("Đọc lưới gốc & tối ưu lại", 20)
    dense = trimesh.load(dense_path, process=False)
    final, texture, stats = make_game_asset(dense, opts, slug, legacy_triposr=is_legacy)
    meta.setdefault("timings", {})["postprocess_s"] = round(time.perf_counter() - t, 2)

    progress("Xuất lại file GLB / OBJ", 80)
    files = export_asset(final, texture, out_dir, slug)
    render_thumbnail(final).save(out_dir / "thumbnail.png")
    render_turnaround(final).save(out_dir / "turnaround.png")

    meta["options"] = asdict(opts)
    meta["stats"] = stats
    meta["files"] = files
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
    progress("Hoàn tất", 100)
    return meta

