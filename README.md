# MeshFM Project Page

Project page for **"2D Features Are All You Need for 3D Shape Understanding"** (MeshFM, ECCV 2026).

Static single-page site adapted from the [threedle/bsb](https://github.com/threedle/bsb/tree/site) project page
(itself adapted from [Geometry in Style](https://threedle.github.io/geometry-in-style/)). No build step — just
`index.html`, `style.css`, and `assets/`.

## Preview locally

```bash
# from this directory
open index.html          # macOS, or just double-click the file
# or serve it (avoids any file:// quirks):
python3 -m http.server 8000   # then visit http://localhost:8000
```

## Deploy (GitHub Pages)

Push the contents of this `site/` folder to a branch (e.g. `gh-pages` or a `site` branch) and enable GitHub Pages
for that branch. The page is fully self-contained except for two CDN dependencies (Font Awesome, Academicons font).

## TODO before publishing

- **Nav links** — `Paper`, `arXiv`, and `Code` buttons currently point to `#`. Replace with real URLs.
- **Deformation figure** — `assets/applications/deformation_placeholder.svg` is a placeholder. Drop in the real figure
  and update the `<img src>` in the Applications section.
- **Image sizes** — the figures copied from the LaTeX source are large (the page is ~37 MB total). Downsize / re-encode
  (e.g. convert big PNGs to optimized JPG/WebP at web resolution) before publishing for faster loads.
- **Author links** — Jinfan Zhou and Richard Liu are plain text; add homepage links if available.
- **BibTeX** — update `@InProceedings{zhou2026meshfm, ...}` with final page numbers once available.

## Asset provenance

Figures were copied from `../ECCV_2026_MeshFM/figures/` and renamed:

| Page asset | Source figure |
|---|---|
| `teaser/teaser.png` | `teaser_v11.drawio.png` |
| `key_idea/feature_aliasing.png` | `sam_feature_bleeding_v2.png` |
| `method.png` | `meshfm_cam.png` |
| `gallery.png` | `seg_viz_hie_v4.drawio.png` |
| `robustness/rotation.png` | `seg_rot_v1.drawio.png` |
| `applications/correspondence.jpg` | `corr_viz.jpg` |
| `comparison/seg_comparison.png` | `seg_comp_v2.drawio.png` |
| `comparison/corr_comparison.png` | `corr_compare_v3.drawio.png` |
