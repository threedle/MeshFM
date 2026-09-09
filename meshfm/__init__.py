"""MeshFM: feedforward 3D feature fields for meshes and point clouds."""

from .config import load_config
from .inference import MeshFM, MeshFeatures, default_device
from .mesh_io import (
    denormalize_points,
    face_centroids,
    load_mesh,
    normalization_transform,
    normalize_vertices,
    save_colored_mesh,
    save_colored_point_cloud,
)
from .sampling import sample_surface_points
from .viz import pca_colors

__version__ = '1.0.0'

__all__ = [
    'MeshFM',
    'MeshFeatures',
    'default_device',
    'load_config',
    'load_mesh',
    'normalize_vertices',
    'normalization_transform',
    'denormalize_points',
    'face_centroids',
    'save_colored_mesh',
    'save_colored_point_cloud',
    'sample_surface_points',
    'pca_colors',
]
