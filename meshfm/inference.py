"""Feature extraction API."""

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .config import load_config
from .mesh_io import face_centroids, load_mesh, normalize_vertices
from .model.simple_triplane_model import SimpleTriplaneModel
from .sampling import sample_surface_points


@dataclass
class MeshFeatures:
    """Features read off the feature field of one shape."""

    features: np.ndarray                 # [N, C] at the query points
    query_points: np.ndarray             # [N, 3] queries, normalized frame
    vertices: np.ndarray                 # [V, 3] original vertices
    vertices_norm: np.ndarray            # [V, 3] normalized vertices
    faces: np.ndarray                    # [F, 3]
    mode: str
    mesh_path: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def feature_dim(self):
        return int(self.features.shape[1])


def default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MeshFM:
    """Feedforward triplane feature field over a 3D shape."""

    def __init__(self, model, cfg, device=None):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device) if device is not None else next(model.parameters()).device

    @classmethod
    def from_checkpoint(cls, checkpoint_path, config_path, device=None, apply_tanh=True, verbose=True):
        """Build the network from a config and load pretrained weights."""
        device = torch.device(device) if device is not None else default_device()
        cfg = load_config(config_path)

        model = SimpleTriplaneModel(cfg=cfg, apply_tanh=apply_tanh)
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if any(k.startswith('module.') for k in state):
            state = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}

        model.load_state_dict(state, strict=True)
        model.eval().to(device)

        if verbose:
            n_params = sum(p.numel() for p in model.parameters())
            print(f'[meshfm] loaded {checkpoint_path}')
            print(f'[meshfm] {n_params / 1e6:.1f}M params | triplane '
                  f'{cfg.pvcnn.z_triplane_resolution}^2 x {cfg.triplane_channels_high}ch | device {device}')
        return cls(model, cfg, device=device)

    @torch.no_grad()
    def encode(self, points):
        """Point cloud ``[N,3]`` or ``[B,N,3]`` in the normalized frame -> triplane ``[B,3,C,H,W]``."""
        if not torch.is_tensor(points):
            points = torch.from_numpy(np.asarray(points))
        points = points.float()
        if points.dim() == 2:
            points = points.unsqueeze(0)
        return self.model(points.to(self.device))

    @torch.no_grad()
    def query(self, triplane, query_points, chunk=200000):
        """Sample the triplane at ``[N,3]`` normalized positions -> ``[N,C]`` on the CPU."""
        if not torch.is_tensor(query_points):
            query_points = torch.from_numpy(np.asarray(query_points))
        query_points = query_points.float()
        if query_points.dim() == 3:
            if query_points.shape[0] != 1:
                raise ValueError('query() handles one shape at a time; pass [N,3].')
            query_points = query_points[0]

        out = []
        for i in range(0, query_points.shape[0], chunk):
            q = query_points[i:i + chunk].unsqueeze(0).to(self.device)
            q = torch.clamp(q, -1.0, 1.0)
            feats = self.model.sample_triplane_features_from_points(triplane, q)
            out.append(feats.squeeze(0).float().cpu())
        if not out:
            return torch.zeros(0, int(self.cfg.triplane_channels_high))
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def extract_mesh_features(
        self,
        mesh,
        mode='vertex',
        num_sample_points=100000,
        num_query_points=None,
        query_points=None,
        chunk=200000,
        sampler='torch',
        weld=False,
        loader='auto',
        normalize=True,
        seed=None,
    ):
        """Run the network on one shape and read features off the feature field.

        Args:
            mesh: path to a mesh file, or a ``(vertices, faces)`` pair.
            mode: ``'vertex'`` (per input vertex), ``'face'`` (per face centroid),
                ``'surface'`` (at the sampled surface points), or ``'custom'``
                with ``query_points`` supplied.
            num_sample_points: surface points fed to the encoder.
            num_query_points: ``'surface'`` mode only, keep this many of the
                sampled points. The encoder still sees all of them.
            query_points: ``[N,3]`` positions for ``mode='custom'``, in the
                normalized frame unless ``normalize=False``.
            chunk: query batch size.
            sampler: surface sampling backend.
            weld: merge coincident vertices on load.
            loader: mesh loader backend.
            normalize: apply the unit-sphere normalization.
            seed: seed for the surface sampler.

        Returns:
            :class:`MeshFeatures`.
        """
        mesh_path = None
        if isinstance(mesh, (str, os.PathLike)):
            mesh_path = str(mesh)
            vertices, faces = load_mesh(mesh_path, weld=weld, loader=loader)
        else:
            vertices, faces = mesh
            vertices = np.asarray(vertices, dtype=np.float64)
            faces = np.asarray(faces, dtype=np.int64)

        if len(vertices) == 0:
            raise ValueError(f'Mesh has no vertices: {mesh_path}')

        vertices_norm = normalize_vertices(vertices) if normalize else np.asarray(vertices, dtype=np.float64)

        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        points = sample_surface_points(
            vertices_norm, faces, num_sample_points, backend=sampler, generator=generator
        )

        triplane = self.encode(points)

        if mode == 'vertex':
            queries = vertices_norm
        elif mode == 'face':
            if len(faces) == 0:
                raise ValueError(f"mode='face' needs a mesh with faces: {mesh_path}")
            queries = face_centroids(vertices_norm, faces)
        elif mode == 'surface':
            queries = points.numpy().astype(np.float64)
            if num_query_points is not None and 0 < num_query_points < len(queries):
                keep = torch.randperm(len(queries), generator=generator)[:num_query_points]
                queries = queries[keep.numpy()]
        elif mode == 'custom':
            if query_points is None:
                raise ValueError("mode='custom' requires query_points")
            queries = np.asarray(query_points, dtype=np.float64)
        else:
            raise ValueError(
                f"Unknown mode {mode!r} (expected 'vertex', 'face', 'surface' or 'custom')")

        features = self.query(triplane, queries, chunk=chunk).numpy()

        del triplane
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return MeshFeatures(
            features=features,
            query_points=np.asarray(queries),
            vertices=np.asarray(vertices),
            vertices_norm=np.asarray(vertices_norm),
            faces=np.asarray(faces),
            mode=mode,
            mesh_path=mesh_path,
            meta={
                'num_sample_points': int(num_sample_points),
                'num_query_points': int(len(queries)),
                'sampler': sampler,
                'welded': bool(weld),
                'loader': loader,
                'normalized': bool(normalize),
                'seed': seed,
            },
        )
