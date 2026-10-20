"""Turns a dense AI mesh into a clean, low-poly, game-ready asset for Unity.

Pipeline: orient (Y-up, front = +Z) -> clean -> decimate -> resample colours ->
quantise to a small palette -> flat-shaded mesh + palette texture -> scale & pivot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.spatial import cKDTree

PALETTE_CELL = 8          # pixels per colour cell in the palette texture
PALETTE_GRID = 32         # 32x32 = up to 1024 colours -> 256x256 px texture


@dataclass
class PostprocessOptions:
    target_faces: int = 3500
    palette_colors: int = 36
    target_height_m: float = 1.0
    saturation_boost: float = 1.15
    min_component_ratio: float = 0.02   # drop floating bits smaller than 2% of faces
    front_yaw_deg: float = 0.0          # extra rotation if the object faces the wrong way


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------
def orient_for_unity(
    mesh: trimesh.Trimesh,
    extra_yaw_deg: float = 0.0,
    legacy_triposr: bool = False,
) -> trimesh.Trimesh:
    """Ensure Y-up, front facing +Z. Only rotate Z-up->Y-up for legacy TripoSR dense.ply."""
    m = mesh.copy()
    if legacy_triposr:
        m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
    if extra_yaw_deg:
        m.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(extra_yaw_deg), [0, 1, 0])
        )
    return m



# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_mesh(mesh: trimesh.Trimesh, min_component_ratio: float) -> trimesh.Trimesh:
    m = mesh.copy()
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()

    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        total = sum(len(p.faces) for p in parts)
        keep = [p for p in parts if len(p.faces) >= total * min_component_ratio]
        if not keep:
            keep = [max(parts, key=lambda p: len(p.faces))]
        m = trimesh.util.concatenate(keep)

    # Gentle Taubin smoothing: removes marching-cubes stair-steps (faces/limbs) without shrinking
    if len(m.faces) >= 5000:
        try:
            trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=3)
        except Exception:
            pass

    trimesh.repair.fill_holes(m)
    trimesh.repair.fix_normals(m)
    return m


# ---------------------------------------------------------------------------
# Decimation
# ---------------------------------------------------------------------------
def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh.copy()
    import fast_simplification

    reduction = 1.0 - target_faces / len(mesh.faces)
    pts, faces = fast_simplification.simplify(
        mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32), target_reduction=reduction
    )
    out = trimesh.Trimesh(vertices=pts, faces=faces, process=True)
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(out)
    return out


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------
def _vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    try:
        vc = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
        if len(vc) == len(mesh.vertices):
            return vc
    except Exception:
        pass
    return np.full((len(mesh.vertices), 3), 0.8, dtype=np.float32)


def sample_face_colors(low: trimesh.Trimesh, dense: trimesh.Trimesh) -> np.ndarray:
    """Average the dense mesh colours around points on every low-poly face."""
    dense_colors = _vertex_colors(dense)
    tree = cKDTree(dense.vertices)
    if len(low.faces) > 25000:
        # High-poly fast path: query face centroids directly
        pts = low.triangles_center
        dist, idx = tree.query(pts, k=3)
        w = 1.0 / np.maximum(dist, 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        face_cols = (dense_colors[idx] * w[..., None]).sum(axis=1).astype(np.float64)
    else:
        tri = low.triangles                                      # (F, 3, 3)
        bary = np.array([[1 / 3, 1 / 3, 1 / 3],
                         [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]], dtype=np.float64)
        pts = np.einsum("sk,fkd->fsd", bary, tri).reshape(-1, 3)  # (F*4, 3)
        dist, idx = tree.query(pts, k=6)
        w = 1.0 / np.maximum(dist, 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        col = (dense_colors[idx] * w[..., None]).sum(axis=1)
        face_cols = col.reshape(len(low.faces), len(bary), 3).mean(axis=1)

    # Smooth colours across adjacent faces (bilateral-like weight by colour similarity)
    try:
        adj = low.face_adjacency
        if len(adj) > 0:
            f0, f1 = adj[:, 0], adj[:, 1]
            iters = 1 if len(low.faces) > 40000 else 2
            for _ in range(iters):
                diff = np.linalg.norm(face_cols[f0] - face_cols[f1], axis=1, keepdims=True)
                sim = np.exp(-diff * 6.0)
                acc = face_cols.copy()
                cnt = np.ones((len(face_cols), 1), dtype=np.float64)
                np.add.at(acc, f0, face_cols[f1] * sim)
                np.add.at(cnt, f0, sim)
                np.add.at(acc, f1, face_cols[f0] * sim)
                np.add.at(cnt, f1, sim)
                face_cols = acc / cnt
    except Exception:
        pass
    return face_cols


def _boost_saturation(rgb: np.ndarray, amount: float) -> np.ndarray:
    grey = rgb.mean(axis=1, keepdims=True)
    return np.clip(grey + (rgb - grey) * amount, 0.0, 1.0)


def quantize_colors(face_colors: np.ndarray, areas: np.ndarray, k: int, seed: int = 0):
    """k-means on face colours + fast KDTree nearest-cluster lookup (supports up to 1024 colours & 200k faces)."""
    k = max(1, min(k, PALETTE_GRID * PALETTE_GRID, len(face_colors)))
    rng = np.random.default_rng(seed)
    probs = areas / np.maximum(areas.sum(), 1e-12)
    n_samples = min(50000, max(len(face_colors) * 2, k * 32))
    sample_idx = rng.choice(len(face_colors), size=n_samples, p=probs)
    sample = face_colors[sample_idx].astype(np.float64)
    init_method = "++" if k <= 128 else "points"
    try:
        palette, _ = kmeans2(sample, k, iter=10, minit=init_method, seed=seed)
    except Exception:
        palette, _ = kmeans2(sample, k, iter=8, minit="points", seed=seed)
    _, labels = cKDTree(palette).query(face_colors, k=1)
    labels = labels.astype(np.int32)
    used = np.unique(labels)                      # drop empty clusters
    remap = np.zeros(len(palette), dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return palette[used], remap[labels]


# ---------------------------------------------------------------------------
# Build final textured flat-shaded mesh
# ---------------------------------------------------------------------------
def build_palette_texture(palette: np.ndarray) -> Image.Image:
    size = PALETTE_GRID * PALETTE_CELL
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (palette[0] * 255).astype(np.uint8)
    for i, c in enumerate(palette):
        r, col = divmod(i, PALETTE_GRID)
        img[r * PALETTE_CELL:(r + 1) * PALETTE_CELL,
            col * PALETTE_CELL:(col + 1) * PALETTE_CELL] = (np.clip(c, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img, "RGB")


def _cell_uv(index: np.ndarray) -> np.ndarray:
    r, c = np.divmod(index, PALETTE_GRID)
    u = (c + 0.5) / PALETTE_GRID
    v = 1.0 - (r + 0.5) / PALETTE_GRID          # glTF/OBJ: v=0 at the bottom of the image
    return np.stack([u, v], axis=1)


def build_flat_mesh(mesh: trimesh.Trimesh, labels: np.ndarray, texture: Image.Image,
                    name: str = "asset") -> trimesh.Trimesh:
    """Unweld vertices so every face is flat-shaded and maps to one palette cell."""
    verts = mesh.vertices[mesh.faces].reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)
    uv = np.repeat(_cell_uv(labels), 3, axis=0)
    if len(mesh.faces) >= 3000:
        # Smooth vertex normals for detailed meshes (characters look sculpted, not faceted)
        normals = mesh.vertex_normals[mesh.faces].reshape(-1, 3)
    else:
        normals = np.repeat(mesh.face_normals, 3, axis=0)

    material = trimesh.visual.material.PBRMaterial(
        name=f"{name}_mat",
        baseColorTexture=texture,
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals,
                           visual=visual, process=False)


def normalize_transform(mesh: trimesh.Trimesh, target_height_m: float) -> trimesh.Trimesh:
    """Scale to real height, put pivot at bottom-centre (sits on the ground in Unity)."""
    m = mesh.copy()
    lo, hi = m.bounds
    height = max(hi[1] - lo[1], 1e-6)
    m.apply_scale(target_height_m / height)
    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    return m


def make_game_asset(
    dense: trimesh.Trimesh,
    opts: PostprocessOptions,
    name: str = "asset",
    legacy_triposr: bool = False,
):
    """Full post-processing. Returns (final_mesh, palette_texture, stats)."""
    oriented = orient_for_unity(dense, opts.front_yaw_deg, legacy_triposr=legacy_triposr)
    cleaned = clean_mesh(oriented, opts.min_component_ratio)
    # keep colours on the cleaned dense mesh for sampling
    low = decimate(cleaned, opts.target_faces)
    face_cols = sample_face_colors(low, oriented)
    face_cols = _boost_saturation(face_cols, opts.saturation_boost)
    palette, labels = quantize_colors(face_cols, low.area_faces, opts.palette_colors)
    texture = build_palette_texture(palette)
    flat = build_flat_mesh(low, labels, texture, name)
    final = normalize_transform(flat, opts.target_height_m)
    stats = {
        "dense_faces": int(len(dense.faces)),
        "final_faces": int(len(final.faces)),
        "palette_colors": int(len(palette)),
        "size_m": [round(float(x), 3) for x in final.extents],
    }
    return final, texture, stats

