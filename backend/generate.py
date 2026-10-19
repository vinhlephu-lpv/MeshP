"""CLI to test the pipeline.

Examples:
  python backend/generate.py --text "a cow" --preset animal
  python backend/generate.py --image my_carrot.png --preset crop --name carrot
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from PIL import Image  # noqa: E402

from mesh_core.config import OUTPUT_DIR  # noqa: E402
from mesh_core.model_manager import manager  # noqa: E402
from mesh_core.presets import PRESETS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Mô tả vật thể, tiếng Anh, vd: 'a cow'")
    src.add_argument("--image", help="Đường dẫn ảnh")
    ap.add_argument("--preset", default="prop", choices=list(PRESETS))
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--faces", type=int, default=None, help="Ghi đè số tam giác")
    ap.add_argument("--colors", type=int, default=None, help="Ghi đè số màu")
    args = ap.parse_args()

    from mesh_core.pipeline import image_to_asset, text_to_images
    from mesh_core.postprocess import PostprocessOptions

    t0 = time.perf_counter()
    if args.text:
        print(f"Tạo ảnh từ text: {args.text!r} ...")
        image = text_to_images(args.text, args.preset, num_images=1, seed=args.seed)[0]
        print(f"  xong ({time.perf_counter() - t0:.1f}s) | {manager.vram_report()}")
        name = args.name or args.text
    else:
        image = Image.open(args.image)
        name = args.name or Path(args.image).stem

    preset = PRESETS[args.preset]
    opts = PostprocessOptions(
        target_faces=args.faces or preset.target_faces,
        palette_colors=args.colors or preset.palette_colors,
        target_height_m=preset.target_height_m,
    )
    meta = image_to_asset(image, name, args.preset, opts)
    print("\n=== KẾT QUẢ ===")
    for k in ("stats", "timings", "vram", "files"):
        print(f"{k}: {meta[k]}")
    print(f"Tổng thời gian: {time.perf_counter() - t0:.1f}s")
    print(f"Thư mục: {meta['dir']}")


if __name__ == "__main__":
    main()
