"""Mesh loading, normalization and export."""

import os
import warnings

import numpy as np
import trimesh

IGL_EXTENSIONS = ('.obj', '.off', '.ply', '.stl', '.mesh', '.wrl')


def _load_with_trimesh(path, weld):
    mesh = trimesh.load(path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if weld:
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64) if hasattr(mesh, 'faces') else np.zeros((0, 3), np.int64)
    return vertices, faces


def _load_with_igl(path):
    import igl
    v, f = igl.read_triangle_mesh(path)
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)


def load_mesh(path, weld=False, loader='auto'):
    """Load a mesh as ``(vertices [V,3] float64, faces [F,3] int64)``.

    Args:
        path: ``.obj`` / ``.glb`` / ``.gltf`` / ``.ply`` / ``.off`` / ``.stl``.
        weld: merge vertices sharing a position, ignoring UV/normal splits.
            This renumbers vertices; faces are left untouched.
        loader: ``'auto'``, ``'igl'`` or ``'trimesh'``.

    ``'auto'`` reads with libigl where it can, so vertices come back in the order
    the file stores them; trimesh splits a vertex whenever it carries different
    UVs or normals across faces. ``.glb``/``.gltf`` scenes go through trimesh and
    are concatenated into a single mesh.
    """
    if loader not in ('auto', 'igl', 'trimesh'):
        raise ValueError(f"Unknown loader {loader!r} (expected 'auto', 'igl' or 'trimesh')")

    ext = os.path.splitext(str(path))[1].lower()
    if weld or loader == 'trimesh':
        vertices, faces = _load_with_trimesh(path, weld=bool(weld))
    elif loader == 'igl':
        vertices, faces = _load_with_igl(path)
    elif ext in IGL_EXTENSIONS:
        try:
            vertices, faces = _load_with_igl(path)
        except ImportError:
            warnings.warn(
                'libigl is not installed, falling back to trimesh; vertex order may not match '
                'the input file. Install with `pip install libigl`.',
                RuntimeWarning,
            )
            vertices, faces = _load_with_trimesh(path, weld=False)
    else:
        vertices, faces = _load_with_trimesh(path, weld=False)

    if vertices.shape[-1] > 3:
        vertices = vertices[:, :3]
    return vertices, faces


def normalization_transform(vertices):
    """Return ``(centre, scale)`` with ``normalized = (vertices - centre) / scale``."""
    v = np.asarray(vertices, dtype=np.float64)
    centre = 0.5 * (v.min(axis=0) + v.max(axis=0))
    scale = float(np.max(np.linalg.norm(v - centre, axis=1)))
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('Degenerate mesh: all vertices coincide, cannot normalize.')
    return centre, scale


def normalize_vertices(vertices):
    """Centre on the bounding box and scale to the unit sphere."""
    v = np.asarray(vertices, dtype=np.float64)
    if len(v) == 0:
        return v.copy()
    centre, scale = normalization_transform(v)
    return (v - centre) / scale


def denormalize_points(points, vertices):
    """Map normalized points back into the frame of ``vertices``."""
    centre, scale = normalization_transform(vertices)
    return np.asarray(points, dtype=np.float64) * scale + centre


def face_centroids(vertices, faces):
    return np.asarray(vertices)[np.asarray(faces)].mean(axis=1)


def save_colored_mesh(path, vertices, faces, face_colors=None, vertex_colors=None):
    """Write a mesh with per-face or per-vertex RGB (uint8 ``[N,3]``)."""
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    if face_colors is not None:
        rgba = np.full((len(mesh.faces), 4), 255, dtype=np.uint8)
        rgba[:, :3] = np.asarray(face_colors, dtype=np.uint8)
        mesh.visual.face_colors = rgba
    if vertex_colors is not None:
        rgba = np.full((len(mesh.vertices), 4), 255, dtype=np.uint8)
        rgba[:, :3] = np.asarray(vertex_colors, dtype=np.uint8)
        mesh.visual.vertex_colors = rgba
    mesh.export(path)


def save_colored_point_cloud(path, points, colors):
    """Write an RGB point cloud as ``.ply``."""
    cloud = trimesh.PointCloud(vertices=np.asarray(points), colors=np.asarray(colors, dtype=np.uint8))
    cloud.export(path)
