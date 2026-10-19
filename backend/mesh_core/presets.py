"""Asset presets for a farm game (Unity, metres, Y-up)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetPreset:
    key: str
    label: str
    target_height_m: float      # real-world height in Unity units (metres)
    target_faces: int           # triangle budget for the low-poly result
    palette_colors: int         # number of flat colours
    prompt_hint: str            # extra words added to the text-to-image prompt


PRESETS: dict[str, AssetPreset] = {p.key: p for p in [
    AssetPreset("crop",     "Cây trồng / rau củ", 0.4, 800,  12, "single stylized vegetable crop, plump clean shape"),
    AssetPreset("fruit",    "Trái cây / nông sản", 0.25, 800, 12, "single fresh stylized fruit, plump rounded shape"),
    AssetPreset("tree",     "Cây lớn",            4.0, 2000, 14, "single stylized low-poly farm tree, thick trunk, chunky foliage"),
    AssetPreset("animal",   "Con vật",            1.2, 3000, 16, "single cute stylized farm animal, full body, standing, four legs visible"),
    AssetPreset("building", "Nhà / chuồng trại",  4.0, 3000, 16, "single cute miniature farm building, isometric front-three-quarter view"),
    AssetPreset("tool",     "Dụng cụ",            1.0, 1200, 12, "single stylized farm tool prop, thick sturdy proportions"),
    AssetPreset("prop",     "Đồ vật trang trí",   1.0, 1500, 14, "single stylized farm prop object, clean solid silhouette"),
    AssetPreset("character","Nhân vật",           1.7, 4000, 16, "single cute chibi farm character, full body, standing, A-pose"),
]}

BASE_PROMPT = (
    "({subject}:1.3), {hint}, stylized 3d farm game asset, blender 3d icon render, "
    "clean solid geometry, smooth matte hand-painted finish, vibrant saturated colors, "
    "front three-quarter view, centered, whole object fully visible, "
    "isolated on plain pure white background, soft even studio lighting, no harsh gloss"
)

NEGATIVE_PROMPT = (
    "multiple objects, duplicate, crowd, cropped, cut off, out of frame, text, watermark, "
    "signature, logo, ground, floor, grass, pedestal, base, scene, landscape, background clutter, "
    "cast shadow, harsh specular highlight, mirror reflection, blurry, noisy, lowres, "
    "photorealistic, messy wires, thin fragile strands, deformed, ugly"
)


def build_prompt(subject: str, preset_key: str) -> str:
    preset = PRESETS.get(preset_key) or PRESETS["prop"]
    return BASE_PROMPT.format(subject=subject.strip(), hint=preset.prompt_hint)

