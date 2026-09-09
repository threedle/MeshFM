# MeshFM: 2D Features Are All You Need for 3D Shape Understanding

Jinfan Zhou*, Richard Liu*, Itai Lang, Rana Hanocka — University of Chicago
( equal contribution)

**[[Project page]](https://threedle.github.io/MeshFM/)** **[[arXiv]](https://arxiv.org/abs/2607.27592)** **[[PDF]](https://arxiv.org/pdf/2607.27592)**

MeshFM is a feedforward model that predicts rich, general-purpose features over
3D shapes. Given a mesh or point cloud, a single forward pass produces a
continuous feature field: query any 3D position on or near the surface and get a
1536-dimensional descriptor.

MeshFM distills 2D features from visual foundation models into 3D, so it needs
no 3D annotation, and unlike per-shape distillation methods it needs no
optimization at inference time. The features apply zero-shot to part
segmentation, dense correspondence and mesh deformation, and SO(3) rotation
augmentation during training makes them robust to shapes in arbitrary
orientations.

This repository contains the inference code for feature extraction.

---



## Contents

- [Install](#install)
- [Pretrained model](#pretrained-model)
- [Usage](#usage)
- [Citation](#citation)

---



## Install

Python 3.9+ and a CUDA build of PyTorch (developed with torch 2.4 / CUDA 12.4).

```bash
conda create -n meshfm python=3.10 -y
conda activate meshfm
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```



## Pretrained model

The pretrained model is available
[here](https://drive.google.com/file/d/1ONUMs3Ji_VlmyflO74No5rk5bAHM6qgc/view?usp=sharing).
Download it and place it at `checkpoints/meshfm.pth`, or fetch it directly:

```bash
mkdir -p checkpoints
pip install gdown
gdown 1ONUMs3Ji_VlmyflO74No5rk5bAHM6qgc -O checkpoints/meshfm.pth
```

The model is trained on Objaverse with SO(3) rotation augmentation, distilling
1536-d DINOv2 + SAM-refined features into the 3D feature field.


|                   |                     |
| ----------------- | ------------------- |
| Parameters        | 125.5 M             |
| Feature dimension | 1536                |
| Triplane          | 3 planes, 128 x 128 |
| Encoder input     | 100k surface points |




## Usage

Per-vertex features, with a PCA-coloured `.ply` for inspection in MeshLab or
Blender:

```bash
python scripts/extract_features.py \
    --checkpoint checkpoints/meshfm.pth \
    --config configs/meshfm_1536d.yaml \
    --input path/to/mesh.obj \
    --output out/ --visualize
```

Pass a directory to `--input` to process a folder (`--glob '**/*.glb'`,
`--mode face` for face-centroid features, `--fp16` to halve the file size).

To get a dense feature point cloud, sampled uniformly over the surface rather
than at the mesh vertices:

```bash
python scripts/extract_features.py \
    --checkpoint checkpoints/meshfm.pth \
    --config configs/meshfm_1536d.yaml \
    --input path/to/meshes/ --glob '*.glb' \
    --output out/ \
    --mode surface --num_query_points 50000 \
    --visualize --skip_features
```

`--mode surface` queries the points already fed to the encoder, so it costs no
extra forward pass. `--num_query_points` keeps a subset of them; the encoder
still sees all `--num_sample_points`, so the features are unchanged.
`--skip_features` writes only the `.ply`, which helps when batching, since
50k x 1536 features is around 300 MB per mesh.

With `--mode vertex`, row *i* of the feature array corresponds to vertex *i* of
the input file. `.obj`, `.off`, `.ply` and `.stl` are read with libigl to
preserve that order; `.glb` and `.gltf` scenes are concatenated by trimesh. Use
`--loader {auto,igl,trimesh}` to override.


## Citation

```bibtex
@InProceedings{zhou2026meshfm,
  author    = {Zhou, Jinfan and Liu, Richard and Lang, Itai and Hanocka, Rana},
  title     = {{MeshFM: 2D Features Are All You Need for 3D Shape Understanding}},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

