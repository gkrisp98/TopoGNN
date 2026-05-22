# Fusion 360 Gallery — TopoGNN

Per-face machining feature recognition on the public Fusion 360 Gallery
benchmark (8 classes). Reproduces Tables 4, 6, 8–11 from the paper.

## Data

Download the `breps/` release of the Fusion 360 Gallery dataset and lay it
out as

```
breps/
├── step/   # *.step
└── seg/    # *.seg per-face labels (same basename as the STEP file)
```

`train_test.json` (committed here) is the official Fusion 360 split.

## Five-stage pipeline

1. **`extract_nurbs.py`** — converts each STEP face to its NURBS control net
   (≤10×10×4 + knot vectors), samples a 16×16 trimmed UV grid, and stores up
   to 8 boundary curves per face. One pickle file per model.
2. **`extract_handcrafted.py`** — 17/8/8-dimensional handcrafted descriptors
   per face/edge/vertex, including surface type one-hot, dihedral encoding,
   normalized vertex positions, etc.
3. **`train_face_vae.py` / `train_edge_vae.py` / `train_vertex_vae.py`** —
   self-supervised VAEs trained on entities from the **train split only**.
   Each script writes its best checkpoint and a per-model embedding pickle.
4. **`merge_embeddings.py`** — combines the three per-model embedding files
   into the format consumed by the graph builder.
5. **`build_graphs.py`** — assembles the typed heterogeneous graph and
   selects the reduced handcrafted descriptor subset reported in Table 3
   (`--keep_*_hc_indices`).

`train_gatv2.py` finally trains the relation-aware GATv2 classifier
(focal loss, AdamW, cosine schedule, early stopping on val macro F1).

## End-to-end reproduction

```bash
bash scripts/run_full_pipeline.sh /data/fusion360/step /data/fusion360/seg /data/fusion360/work
```

Expected `models_topognn_full/results.json`:

```json
{
  "test_accuracy": 0.9721,
  "test_f1": 0.9292,
  "test_iou": 0.8757
}
```

## Ablations (Tables 8–11)

```bash
bash scripts/run_ablations.sh /data/fusion360/work $(pwd)
```

Reference numbers are committed under `results/models_rhc_*/results.json`
and are summarized in the top-level `README.md`.
