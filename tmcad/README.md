# TMCAD — TopoGNN

Part-level CAD classification on the TMCAD benchmark (10 mechanical-component
classes). Reproduces Tables 7, 12, 13 from the paper.

## Data

The TMCAD release ships one folder of STEP files per part class. Generate (or
re-use) the stratified 70/15/15 split:

```bash
python generate_split.py --root /data/tmcad/step --output mechcad_split.json --seed 42
```

`mechcad_split.json` (committed here) is the split used in the paper.

## Differences vs. the Fusion 360 pipeline

The geometry / VAE / merge stages are identical to `fusion360/`. Two things
change for graph-level classification:

* `build_graphs.py` writes a single `data.y` per graph instead of per-face
  labels, and uses the `face/edge/vertex` schema with the relations described
  in Section 3.2.
* `train_gatv2.py` replaces the per-face head with a multi-type
  mean+max readout followed by an MLP classifier (Eq. 20).

## End-to-end reproduction

```bash
bash scripts/run_full_pipeline.sh /data/tmcad/step /data/tmcad/work
```

Expected metrics:

| Configuration | Directory | Test acc | Test macro F1 |
|---|---|---|---|
| Handcrafted only (Table 12 row 1) | `models_gatv2_edge_handcrafted` | 0.8739 | 0.8710 |
| Full TopoGNN (HC + VAE, Table 12 row 2 / Table 7) | `models_gatv2_edge_vae` | 0.8850 | 0.8820 |

`results/models_gatv2_edge_vae/results.json` also contains the per-class F1
scores reported in Table 13.

## Multi-seed runs

For the discussion in Section 6.5 (mean ± std), use:

```bash
python run_experiment.py \
    --graphs_dir /data/tmcad/work/graphs_edge_vae \
    --output_dir ./experiment \
    --split_file ./mechcad_split.json \
    --n_runs 5 --device cuda
```

## Diagnostics

`visualize_embeddings.py` reproduces Figs. 10–11 (linear probe + t-SNE of the
VAE / GNN embeddings).
