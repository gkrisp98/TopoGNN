#!/bin/bash
# End-to-end TopoGNN pipeline on the TMCAD classification benchmark.
# Reproduces Table 7 (88.50% acc) and Tables 12-13 from the paper.
#
# Inputs:
#   $STEP_DIR  -- directory of TMCAD .step files (10 part classes)
#   $WORK      -- working directory for intermediate features
set -e

STEP_DIR=${1:?"missing step dir"}
WORK=${2:?"missing work dir"}
SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)

JOINT="$WORK/nurbs"
HC="$WORK/handcrafted"
FACE_OUT="$WORK/face_vae"
EDGE_OUT="$WORK/edge_vae"
VERTEX_OUT="$WORK/vertex_vae"
EMBED="$WORK/embeddings_merged"
GRAPHS_VAE="$WORK/graphs_edge_vae"
GRAPHS_HC="$WORK/graphs_edge_handcrafted"
SPLIT="$SCRIPT_DIR/mechcad_split.json"

mkdir -p "$WORK"

# (Optional) regenerate stratified split:
# python $SCRIPT_DIR/generate_split.py --root "$STEP_DIR" --output "$SPLIT" --seed 42

# 1. Feature extraction.
python $SCRIPT_DIR/extract_nurbs.py        --step_dir "$STEP_DIR" --output_dir "$JOINT" --num_workers 8
python $SCRIPT_DIR/extract_handcrafted.py  --step_dir "$STEP_DIR" --output_dir "$HC"    --num_workers 8

# 2. Self-supervised pretraining on TMCAD train split only.
python $SCRIPT_DIR/train_face_vae.py   --joint_nurbs_dir "$JOINT" --output_dir "$FACE_OUT"   --split_file "$SPLIT" --epochs 100 --batch_size 64 --device cuda
python $SCRIPT_DIR/train_edge_vae.py   --joint_nurbs_dir "$JOINT" --output_dir "$EDGE_OUT"   --split_file "$SPLIT" --epochs 100 --batch_size 64 --device cuda
python $SCRIPT_DIR/train_vertex_vae.py --handcrafted_dir "$HC"    --output_dir "$VERTEX_OUT" --split_file "$SPLIT" --epochs 100 --batch_size 64 --device cuda
python $SCRIPT_DIR/merge_embeddings.py \
    --face_dir "$FACE_OUT/embeddings" --edge_dir "$EDGE_OUT/embeddings" \
    --vertex_dir "$VERTEX_OUT/embeddings" --output_dir "$EMBED"

# 3. Build face-edge-vertex graphs (graph-level labels).
python $SCRIPT_DIR/build_graphs.py \
    --joint_nurbs_dir "$JOINT" --joint_embed_dir "$EMBED" \
    --handcrafted_dir "$HC" --output_dir "$GRAPHS_VAE" --device cuda

python $SCRIPT_DIR/build_graphs.py \
    --joint_nurbs_dir "$JOINT" --handcrafted_dir "$HC" \
    --output_dir "$GRAPHS_HC" --no_embeddings

# 4. Full TopoGNN (HC + VAE) -- Table 7 / Table 12 row 2.
python $SCRIPT_DIR/train_gatv2.py \
    --graphs_dir "$GRAPHS_VAE" --output_dir "$SCRIPT_DIR/models_gatv2_edge_vae" \
    --split_file "$SPLIT" \
    --hidden_dim 256 --num_heads 8 --num_layers 4 \
    --batch_size 64 --epochs 200 --device cuda --seed 42

# 5. Handcrafted-only ablation -- Table 12 row 1.
python $SCRIPT_DIR/train_gatv2.py \
    --graphs_dir "$GRAPHS_HC" --output_dir "$SCRIPT_DIR/models_gatv2_edge_handcrafted" \
    --split_file "$SPLIT" \
    --hidden_dim 256 --num_heads 8 --num_layers 4 \
    --batch_size 64 --epochs 200 --device cuda --seed 42

echo "Done. Compare against tmcad/results/models_gatv2_edge_*/results.json"
