#!/usr/bin/env python3
"""Extract MeshFM features for a mesh or a folder of meshes.

Examples:
    python scripts/extract_features.py \
        --checkpoint checkpoints/meshfm.pth \
        --config configs/meshfm_1536d.yaml \
        --input mesh.obj --output out/ --visualize

    python scripts/extract_features.py \
        --checkpoint checkpoints/meshfm.pth \
        --config configs/meshfm_1536d.yaml \
        --input meshes/ --glob '**/*.glb' \
        --output out/ --mode face
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meshfm import (  # noqa: E402
    MeshFM, denormalize_points, pca_colors, save_colored_mesh, save_colored_point_cloud,
)

MESH_EXTENSIONS = ('.obj', '.glb', '.gltf', '.ply', '.off', '.stl')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, help='pretrained weights (.pth)')
    p.add_argument('--config', required=True, help='model config YAML')
    p.add_argument('--input', required=True, help='mesh file or directory of meshes')
    p.add_argument('--output', required=True, help='output directory')
    p.add_argument('--glob', default='**/*', help="glob under --input when it is a directory")
    p.add_argument('--mode', default='vertex', choices=['vertex', 'face', 'surface'],
                   help='query at input vertices, face centroids, or sampled surface points')
    p.add_argument('--num_sample_points', type=int, default=100000,
                   help='surface points fed to the encoder')
    p.add_argument('--num_query_points', type=int, default=0,
                   help='--mode surface only: keep this many sampled points (0 = all)')
    p.add_argument('--chunk', type=int, default=200000, help='query points per forward chunk')
    p.add_argument('--sampler', default='torch', choices=['torch', 'kaolin', 'auto'],
                   help='surface sampling backend')
    p.add_argument('--loader', default='auto', choices=['auto', 'igl', 'trimesh'],
                   help='mesh loader backend')
    p.add_argument('--weld', action='store_true', help='merge coincident vertices on load')
    p.add_argument('--seed', type=int, default=0, help='sampling seed, -1 for nondeterministic')
    p.add_argument('--device', default=None, help="'cuda', 'cuda:1' or 'cpu'")
    p.add_argument('--fp16', action='store_true', help='store features as float16')
    p.add_argument('--visualize', action='store_true', help='also write a PCA-coloured .ply')
    p.add_argument('--skip_features', action='store_true', help='write only the visualization')
    p.add_argument('--overwrite', action='store_true', help='recompute meshes that already have outputs')
    p.add_argument('--limit', type=int, default=0, help='process at most N meshes (0 = all)')
    return p.parse_args()


def collect_meshes(input_path, pattern, limit=0):
    if os.path.isfile(input_path):
        return [input_path]
    import glob as globlib
    paths = sorted(
        p for p in globlib.glob(os.path.join(input_path, pattern), recursive=True)
        if p.lower().endswith(MESH_EXTENSIONS)
    )
    return paths[:limit] if limit > 0 else paths


def write_visualization(out_stem, result):
    colors = pca_colors(result.features)
    if result.mode == 'surface':
        points = denormalize_points(result.query_points, result.vertices)
        save_colored_point_cloud(out_stem + '_pca_points.ply', points, colors)
    elif result.mode == 'face':
        save_colored_mesh(out_stem + '_pca.ply', result.vertices, result.faces, face_colors=colors)
    elif len(result.faces):
        save_colored_mesh(out_stem + '_pca.ply', result.vertices, result.faces, vertex_colors=colors)
    else:
        save_colored_point_cloud(out_stem + '_pca.ply', result.vertices, colors)


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    mesh_paths = collect_meshes(args.input, args.glob, args.limit)
    if not mesh_paths:
        print(f'[error] no meshes found under {args.input}', file=sys.stderr)
        return 1
    print(f'[main] {len(mesh_paths)} mesh(es) to process')

    model = MeshFM.from_checkpoint(args.checkpoint, args.config, device=args.device)
    seed = None if args.seed < 0 else args.seed

    n_ok = n_skip = n_err = 0
    for i, mesh_path in enumerate(mesh_paths, 1):
        stem = os.path.splitext(os.path.basename(mesh_path))[0]
        out_stem = os.path.join(args.output, stem)
        if os.path.exists(out_stem + '.json') and not args.overwrite:
            print(f'[{i}/{len(mesh_paths)}] skip {stem} (exists)')
            n_skip += 1
            continue

        t0 = time.time()
        try:
            result = model.extract_mesh_features(
                mesh_path,
                mode=args.mode,
                num_sample_points=args.num_sample_points,
                num_query_points=args.num_query_points or None,
                chunk=args.chunk,
                sampler=args.sampler,
                weld=args.weld,
                loader=args.loader,
                seed=seed,
            )
        except Exception:
            traceback.print_exc()
            print(f'[{i}/{len(mesh_paths)}] ERROR {stem}', file=sys.stderr)
            n_err += 1
            continue

        features = result.features.astype(np.float16 if args.fp16 else np.float32)
        if not args.skip_features:
            np.save(out_stem + '.npy', features)
        with open(out_stem + '.json', 'w') as f:
            json.dump({
                'mesh_path': os.path.abspath(mesh_path),
                'mode': result.mode,
                'num_vertices': int(len(result.vertices)),
                'num_faces': int(len(result.faces)),
                'feature_shape': list(features.shape),
                'dtype': str(features.dtype),
                **result.meta,
            }, f, indent=2)

        if args.visualize:
            write_visualization(out_stem, result)

        print(f'[{i}/{len(mesh_paths)}] {stem}: {features.shape} in {time.time() - t0:.1f}s')
        n_ok += 1

    print(f'[main] done. ok={n_ok} skipped={n_skip} errors={n_err}')
    return 1 if n_err and not n_ok else 0


if __name__ == '__main__':
    raise SystemExit(main())
