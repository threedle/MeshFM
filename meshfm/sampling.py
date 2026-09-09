"""Uniform surface point sampling."""

import numpy as np
import torch

# torch.multinomial cannot address more than 2**24 categories.
MAX_FACES = 2 ** 23


def _as_tensors(vertices, faces):
    if not torch.is_tensor(vertices):
        vertices = torch.from_numpy(np.asarray(vertices))
    if not torch.is_tensor(faces):
        faces = torch.from_numpy(np.asarray(faces))
    return vertices.float(), faces.long()


def _subsample_faces(faces, generator=None):
    if faces.shape[0] <= MAX_FACES:
        return faces
    keep = torch.randperm(faces.shape[0], generator=generator)[:MAX_FACES]
    return faces[keep]


def sample_surface_points(vertices, faces, num_points, backend='torch', generator=None):
    """Sample ``num_points`` points uniformly over the mesh surface.

    Args:
        vertices: ``[V,3]`` array or tensor, already normalized.
        faces: ``[F,3]`` array or tensor. If empty, the vertices are treated as
            a point cloud and subsampled.
        num_points: number of samples.
        backend: ``'torch'``, ``'kaolin'``, or ``'auto'`` to prefer kaolin when
            it is importable.
        generator: optional ``torch.Generator`` for reproducible sampling.

    Returns:
        ``[num_points, 3]`` float tensor on the CPU.
    """
    vertices, faces = _as_tensors(vertices, faces)
    if torch.isnan(vertices).any() or torch.isinf(vertices).any():
        vertices = torch.nan_to_num(vertices, nan=0.0, posinf=0.0, neginf=0.0)

    if faces.numel() == 0:
        if vertices.shape[0] > num_points:
            idx = torch.randperm(vertices.shape[0], generator=generator)[:num_points]
            return vertices[idx]
        return vertices

    if backend == 'auto':
        try:
            import kaolin  # noqa: F401
            backend = 'kaolin'
        except ImportError:
            backend = 'torch'

    faces = _subsample_faces(faces, generator=generator)

    if backend == 'kaolin':
        import kaolin as kal
        points, _ = kal.ops.mesh.sample_points(vertices.unsqueeze(0), faces, num_points)
        return points.squeeze(0)
    if backend != 'torch':
        raise ValueError(f"Unknown sampling backend {backend!r} (expected 'torch', 'kaolin' or 'auto')")

    tri = vertices[faces]
    ab, ac = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    areas = 0.5 * torch.linalg.cross(ab, ac, dim=-1).norm(dim=-1)
    if float(areas.sum()) <= 0:
        raise ValueError('Mesh has zero total surface area, cannot sample points.')

    face_idx = torch.multinomial(areas, num_points, replacement=True, generator=generator)
    uv = torch.rand(num_points, 2, generator=generator)
    su = uv[:, :1].sqrt()
    w1, w2 = su * (1.0 - uv[:, 1:]), su * uv[:, 1:]
    picked = tri[face_idx]
    return picked[:, 0] + w1 * (picked[:, 1] - picked[:, 0]) + w2 * (picked[:, 2] - picked[:, 0])
