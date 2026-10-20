"""Asset presets for Unity (metres, Y-up) with user-customizable preset storage."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

from .config import DATA_DIR

CUSTOM_PRESETS_FILE = DATA_DIR / "custom_presets.json"


@dataclass(frozen=True)
class AssetPreset:
    key: str
    label: str
    target_height_m: float      # real-world height in Unity units (metres)
    target_faces: int           # triangle budget for the result
    palette_colors: int         # number of colours
    prompt_hint: str            # extra words added to the text-to-image prompt
    custom: bool = False


DEFAULT_PRESET = AssetPreset(
    key="default",
    label="Mặc định",
    target_height_m=1.0,
    target_faces=10000,
    palette_colors=128,
    prompt_hint="single stylized 3d game model, clean well-defined proportions",
    custom=False,
)

# Built-in fallback lookup so legacy keys never error
PRESETS: dict[str, AssetPreset] = {
    "default": DEFAULT_PRESET,
}


def load_all_presets() -> dict[str, AssetPreset]:
    out: dict[str, AssetPreset] = {"default": DEFAULT_PRESET}
    if CUSTOM_PRESETS_FILE.exists():
        try:
            items = json.loads(CUSTOM_PRESETS_FILE.read_text("utf-8"))
            for item in items:
                p = AssetPreset(
                    key=str(item["key"]),
                    label=str(item["label"]),
                    target_height_m=float(item.get("target_height_m", 1.0)),
                    target_faces=int(item.get("target_faces", 10000)),
                    palette_colors=int(item.get("palette_colors", 128)),
                    prompt_hint=str(item.get("prompt_hint", "single 3d game asset")),
                    custom=True,
                )
                out[p.key] = p
        except Exception:
            pass
    PRESETS.clear()
    PRESETS.update(out)
    return out


def save_custom_preset(
    label: str,
    target_height_m: float = 1.0,
    target_faces: int = 10000,
    palette_colors: int = 128,
    prompt_hint: str = "single stylized 3d game asset",
) -> list[dict]:
    all_p = load_all_presets()
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.lower()).strip("_") or "custom"
    import time
    key = f"custom_{slug}_{int(time.time() * 1000) % 10_000_000}"
    new_p = AssetPreset(
        key=key,
        label=label.strip() or "Tùy chỉnh",
        target_height_m=float(target_height_m),
        target_faces=int(target_faces),
        palette_colors=int(palette_colors),
        prompt_hint=prompt_hint.strip() or "single stylized 3d game asset",
        custom=True,
    )
    all_p[key] = new_p
    custom_list = [asdict(p) for p in all_p.values() if p.custom]
    CUSTOM_PRESETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_PRESETS_FILE.write_text(json.dumps(custom_list, indent=2, ensure_ascii=False), "utf-8")
    load_all_presets()
    return [asdict(p) for p in PRESETS.values()]


def delete_custom_preset(key: str) -> list[dict]:
    all_p = load_all_presets()
    if key in all_p and all_p[key].custom:
        del all_p[key]
        custom_list = [asdict(p) for p in all_p.values() if p.custom]
        CUSTOM_PRESETS_FILE.write_text(json.dumps(custom_list, indent=2, ensure_ascii=False), "utf-8")
        load_all_presets()
    return [asdict(p) for p in PRESETS.values()]


load_all_presets()

BASE_PROMPT = (
    "({subject}:1.35), {hint}, high quality 3d character and asset render, "
    "clean well-defined anatomy and proportions, smooth hand-painted stylized 3d model, "
    "vibrant colors, front view, centered, whole subject fully visible from head to toe, "
    "separated limbs, isolated on plain pure white background, soft even studio lighting, no harsh gloss"
)

NEGATIVE_PROMPT = (
    "multiple objects, duplicate, crowd, cropped, cut off, out of frame, text, watermark, "
    "signature, logo, ground, floor, grass, pedestal, base, scene, landscape, background clutter, "
    "cast shadow, harsh specular highlight, mirror reflection, blurry, noisy, lowres, "
    "merged limbs, fused legs, extra limbs, missing limbs, deformed hands, mutated, ugly"
)


def get_preset(preset_key: str) -> AssetPreset:
    load_all_presets()
    return PRESETS.get(preset_key) or DEFAULT_PRESET


def build_prompt(subject: str, preset_key: str) -> str:
    preset = get_preset(preset_key)
    subj = subject.strip()
    lower = subj.lower()
    # Automatically add A-pose full-body hint when subject describes a person/character
    char_keywords = ("human", "person", "man", "woman", "boy", "girl", "character", "farmer", "worker", "warrior", "knight", "chibi", "villager", "người")
    extra = ""
    if any(k in lower for k in char_keywords):
        extra = "full body standing in A-pose, arms slightly apart, legs separated, clear face and hands, "
    return BASE_PROMPT.format(subject=subj, hint=extra + preset.prompt_hint)
