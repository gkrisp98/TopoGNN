# TopoGNN

Reference implementation of **TopoGNN: Heterogeneous B-Rep Graph Learning with NURBS-Based Geometric Embeddings for Industrial Machining Feature Recognition** (Computer-Aided Design, 2026).

TopoGNN operates directly on STEP boundary representations and models each CAD part as a heterogeneous face–edge–vertex graph. Type-specific self-supervised variational autoencoders learn NURBS- and UV-based geometric embeddings, and a relation-aware GATv2 classifier produces per-face (segmentation) or per-graph (classification) predictions.

This repository reproduces the public-benchmark results from the paper:

| Benchmark | Task | Accuracy | macro F1 | mIoU |
|---|---|---|---|---|
| Fusion 360 Gallery | Per-face MFR (8 classes) | **0.9721** | **0.9292** | **0.8757** |
| TMCAD             | Part classification (10 classes) | **0.8850** | **0.8820** | – |

The paper additionally reports results on a proprietary industrial dataset; that dataset cannot be redistributed and is therefore not part of this repository.

## Repository layout

```
TopoGNN/
├── environment.yml             # conda environment (PyTorch + PyG + pythonocc-core)
├── fusion360/                  # Fusion 360 Gallery (per-face MFR)
│   ├── extract_nurbs.py        # NURBS / UV / boundary-curve extraction
│   ├── extract_handcrafted.py  # 17/8/8-D handcrafted descriptors
│   ├── train_face_vae.py       # Self-supervised face VAE
│   ├── train_edge_vae.py       # Self-supervised edge VAE
│   ├── train_vertex_vae.py     # Self-supervised vertex VAE
│   ├── merge_embeddings.py     # Merge per-entity embeddings into per-model files
│   ├── build_graphs.py         # Assemble the heterogeneous B-Rep graphs
│   ├── train_gatv2.py          # Relation-aware GATv2 classifier
│   ├── encoder_utils.py        # Shared VAE / encoder modules
│   ├── train_test.json         # Official Fusion 360 split
│   ├── utils/                  # OCC helpers + scaling utilities
│   ├── scripts/
│   │   ├── run_full_pipeline.sh   # Reproduces the Full model row
│   │   └── run_ablations.sh       # Reproduces Tables 8–11
│   └── results/                # Reference test metrics (results.json)
├── tmcad/                      # TMCAD (part-level classification)
│   ├── extract_nurbs.py
│   ├── extract_handcrafted.py
│   ├── train_face_vae.py / train_edge_vae.py / train_vertex_vae.py
│   ├── merge_embeddings.py
│   ├── build_graphs.py         # Graph-level labels, edge-based variant
│   ├── train_gatv2.py          # Graph-classification GATv2
│   ├── generate_split.py       # Stratified 70/15/15 split
│   ├── mechcad_split.json      # The split used in the paper
│   ├── visualize_embeddings.py # t-SNE + linear-probe diagnostics
│   ├── run_experiment.py       # Multi-seed runner (mean ± std)
│   ├── scripts/run_full_pipeline.sh
│   └── results/
└── docs/                       # (placeholder for figures, etc.)
```

## Installation

The pipeline depends on `pythonocc-core` for STEP parsing, on `torch_geometric`
for the heterogeneous GNN, and on `occwl` for UV grid sampling. The provided
conda environment installs everything needed for both benchmarks:

```bash
conda env create -f environment.yml
conda activate neuronurbs_fusion
pip install occwl    # UV-grid sampling helper
```

A CUDA-capable GPU is recommended; all numbers in the paper were obtained on
a single NVIDIA GPU. CPU-only runs are possible but the VAE pretraining is
slow.

## Datasets

| Dataset | Source | Notes |
|---|---|---|
| Fusion 360 Gallery (segmentation) | <https://github.com/AutodeskAILab/Fusion360GalleryDataset> | The `breps/` release ships `step/` files and per-face `seg/` labels. We use the official train / validation / test split, mirrored in `fusion360/train_test.json`. |
| TMCAD | The TMCAD release of Zou & Zhu, *CAD 2025* (BRT) | 10 part classes, 10,897 STEP files. We use a 70/15/15 stratified split (`tmcad/mechcad_split.json`). To regenerate from a different seed run `python tmcad/generate_split.py`. |

The proprietary industrial dataset reported in Tables 4–5 of the paper is **not** included.

## Reproducing the Fusion 360 result (Tables 4, 6)

After downloading the Fusion 360 Gallery release, run:

```bash
bash fusion360/scripts/run_full_pipeline.sh \
    /path/to/fusion360/breps/step \
    /path/to/fusion360/breps/seg  \
    /path/to/work_dir
```

The script performs, in order: NURBS / UV / boundary-curve extraction →
handcrafted descriptor extraction → self-supervised pretraining of the three
VAEs → embedding merge → heterogeneous-graph construction → GATv2 training.

The produced `models_topognn_full/results.json` should match
`fusion360/results/models_gatv2_newseed/results.json` (test accuracy 0.9721,
macro F1 0.9292, mIoU 0.8757).

> **Two GNN configurations appear in the paper.** The headline Fusion 360 row
> was trained with 4 layers, 8 heads and lr 5e-4 (what `run_full_pipeline.sh`
> uses). The ablation suite in Tables 8-11 was trained with 6 layers, 4 heads
> and lr 1e-3, so that every variant is compared against a common backbone;
> `run_ablations.sh` uses those. This is why `models_rhc_baseline` (0.9722 /
> 0.9259 / 0.8717) is close to but not identical to the headline row. The
> `config` block inside each committed `results.json` records the exact
> settings of that run.

### Ablations (Tables 8–11)

Once the features and embeddings exist (steps 1–4 above), reproduce the
ablation tables with:

```bash
bash fusion360/scripts/run_ablations.sh /path/to/work_dir $(pwd)/fusion360
```

The script emits one model directory per ablation (`models_rhc_*`); the
expected metrics are committed under `fusion360/results/`. Mapping to the paper:

| Variant | Directory | Paper table |
|---|---|---|
| Full model | `models_rhc_baseline` | 8/9/10/11 |
| No handcrafted features | `models_rhc_no_hc` | 8 |
| Handcrafted only | `models_rhc_hc_only` | 8 |
| Face-only graph | `models_rhc_face_only` | 9 |
| No edges | `models_rhc_no_edges` | 9 |
| No vertices | `models_rhc_no_vertices` | 9 |
| 5 relations (no loop / vertex-incidence) | `models_rhc_4relations` | 9 |
| UV only | `models_rhc_uv_only` | 10 |
| NURBS only | `models_rhc_nurbs_only` | 10 |
| Simplified model | `models_rhc_simplified` | 11 |
| No VAE pretraining | `models_rhc_no_vae` | 11 |

The `UV only` and `NURBS only` ablations require re-training the Face VAE
with the corresponding branch disabled, then re-running
`merge_embeddings.py` into `embeddings_merged_uv_only` and
`embeddings_merged_nurbs_only`.

The reduced handcrafted descriptor set used in the paper (Table 3) is
selected via the `--keep_*_hc_indices` flags, hard-coded at the top of
`scripts/run_ablations.sh`.

### Handcrafted descriptor sets

The full extractor outputs **17 face / 8 edge / 8 vertex** features. The
paper retains only the compact subset shown in Table 3:

| Entity | Dim | Indices kept | Description |
|---|---|---|---|
| Face   | 10 | 0–8, 16 | 7-class surface-type one-hot, area, aspect ratio, number of boundary loops |
| Edge   | 7  | 0–5, 7  | Absolute / relative length, dihedral (angle, sin, cos), convexity, boundary flag |
| Vertex | 6  | 0–4, 6  | Normalized xyz, distance to centre of gravity, valence, boundary flag |

`build_graphs.py --keep_*_hc_indices` performs this selection.

## Reproducing the TMCAD result (Tables 7, 12, 13)

```bash
bash tmcad/scripts/run_full_pipeline.sh /path/to/tmcad/step /path/to/work_dir
```

The script trains the full TopoGNN classifier (HC + VAE) and the
handcrafted-only ablation. Expected metrics:

| Configuration | Directory | Test acc | Test macro F1 |
|---|---|---|---|
| Handcrafted only | `models_gatv2_edge_handcrafted` | 0.8739 | 0.8710 |
| Full TopoGNN     | `models_gatv2_edge_vae`         | 0.8850 | 0.8820 |

Multi-seed statistics (used for the discussion in Section 6.5) can be
obtained with `python tmcad/run_experiment.py --n_runs 5 ...`.

## Citation

```bibtex
@article{topognn2026,
  title   = {TopoGNN: Heterogeneous B-Rep Graph Learning with NURBS-Based Geometric Embeddings for Industrial Machining Feature Recognition},
  author  = {...},
  journal = {Computer-Aided Design},
  year    = {2026}
}
```

## License

Released under the MIT License (`LICENSE`). The Fusion 360 Gallery and TMCAD
datasets retain the licenses of their respective providers.
