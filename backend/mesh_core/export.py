"""Export helpers (Unity-friendly) and a tiny software renderer for thumbnails."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def export_asset(mesh: trimesh.Trimesh, texture: Image.Image, out_dir: Path, name: str) -> dict:
    """Writes GLB (glTFast / web) and OBJ+MTL+PNG (native Unity import)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    glb_path = out_dir / f"{name}.glb"
    mesh.export(glb_path)
    files["glb"] = glb_path

    obj_dir = out_dir / "obj"
    obj_dir.mkdir(exist_ok=True)
    tex_name = f"{name}_palette.png"
    texture.save(obj_dir / tex_name)
    _write_obj(mesh, obj_dir / f"{name}.obj", obj_dir / f"{name}.mtl", tex_name, name)
    files["obj"] = obj_dir / f"{name}.obj"
    files["texture"] = obj_dir / tex_name
    return {k: str(v) for k, v in files.items()}


def _write_obj(mesh: trimesh.Trimesh, obj_path: Path, mtl_path: Path, tex_name: str, name: str):
    """Hand-written OBJ so the MTL/texture names are predictable for Unity."""
    v = mesh.vertices
    uv = np.asarray(mesh.visual.uv)
    n = mesh.vertex_normals
    f = mesh.faces + 1
    mat = f"{name}_mat"
    with open(mtl_path, "w", encoding="utf-8") as fh:
        fh.write(f"newmtl {mat}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nNs 0\nillum 1\nmap_Kd {tex_name}\n")
    lines = [f"mtllib {mtl_path.name}", f"o {name}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    lines += [f"vt {a:.6f} {b:.6f}" for a, b in uv]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in n]
    lines.append(f"usemtl {mat}")
    lines += [f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}" for a, b, c in f]
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Thumbnail renderer (painter's algorithm, no OpenGL needed)
# ---------------------------------------------------------------------------
def _face_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    vis = mesh.visual
    if isinstance(vis, trimesh.visual.TextureVisuals) and vis.uv is not None:
        tex = np.asarray(vis.material.baseColorTexture.convert("RGB")).astype(np.float32) / 255
        h, w = tex.shape[:2]
        uv = np.asarray(vis.uv)[mesh.faces].mean(axis=1)
        px = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
        py = np.clip(((1 - uv[:, 1]) * h).astype(int), 0, h - 1)
        return tex[py, px]
    try:
        return np.asarray(vis.face_colors)[:, :3].astype(np.float32) / 255
    except Exception:
        return np.full((len(mesh.faces), 3), 0.8, dtype=np.float32)


def render_thumbnail(mesh: trimesh.Trimesh, size: int = 512, yaw_deg: float = 30.0,
                     pitch_deg: float = 20.0) -> Image.Image:
    """Y-up mesh, camera looks at the front (+Z side) rotated by yaw/pitch."""
    cols = _face_colors(mesh)
    v = mesh.vertices - mesh.bounds.mean(axis=0)
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    rot = rx @ ry
    v = v @ rot.T
    normals = mesh.face_normals @ rot.T
    scale = 0.8 * size / max(np.ptp(v[:, 0]), np.ptp(v[:, 1]), 1e-6)
    pts = np.stack([v[:, 0] * scale + size / 2, -v[:, 1] * scale + size / 2], axis=1)

    light = np.array([0.4, 0.6, 0.7])
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.clip(normals @ light, 0, 1)
    visible = normals[:, 2] > 0                      # camera looks down -Z, sees +Z faces
    depth = v[mesh.faces][:, :, 2].mean(axis=1)
    order = np.argsort(depth)                          # far -> near

    img = Image.new("RGB", (size, size), (235, 238, 242))
    draw = ImageDraw.Draw(img)
    tri = pts[mesh.faces]
    for i in order:
        if not visible[i]:
            continue
        c = tuple(int(x) for x in np.clip(cols[i] * shade[i] * 255, 0, 255))
        draw.polygon([tuple(p) for p in tri[i]], fill=c, outline=c)
    return img


def render_turnaround(mesh: trimesh.Trimesh, size: int = 256) -> Image.Image:
    """Front / right / back / left views side by side (for quick checks)."""
    views = [render_thumbnail(mesh, size, yaw, 15) for yaw in (0, 90, 180, 270)]
    out = Image.new("RGB", (size * 4, size))
    for i, im in enumerate(views):
        out.paste(im, (i * size, 0))
    return out
