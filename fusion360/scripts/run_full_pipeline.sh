#!/bin/bash
# End-to-end TopoGNN pipeline on Fusion 360 Gallery.
#
# Inputs:
#   $STEP_DIR  -- directory of .step files from the Fusion 360 Gallery release
#   $SEG_DIR   -- directory of .seg label files (same release)
#   $WORK      -- working directory for intermediate features
#
# Reproduces the Full model row in Tables 4-12 (test acc 0.9721, F1 0.9292,
# mIoU 0.8757).
#
# Usage:
#   bash run_full_pipeline.sh /path/to/step_files /path/to/seg_files /path/to/work
set -e

STEP_DIR=${1:?"missing step dir"}
SEG_DIR=${2:?"missing seg dir"}
WORK=${3:?"missing work dir"}
SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$WORK"

JOINT="$WORK/nurbs_10x10"
HC="$WORK/handcrafted"
FACE_OUT="$WORK/face_vae"
EDGE_OUT="$WORK/edge_vae"
VERTEX_OUT="$WORK/vertex_vae"
EMBED="$WORK/embeddings_merged"
GRAPHS="$WORK/graphs"
SPLIT="$SCRIPT_DIR/train_test.json"

# 1. Geometry extraction (NURBS + UV grid + boundary curves).
python $SCRIPT_DIR/extract_nurbs.py \
    --data_dir "$STEP_DIR" --output_dir "$JOINT" --seg_dir "$SEG_DIR"

# 2. Handcrafted descriptors (Table 3).
python $SCRIPT_DIR/extract_handcrafted.py \
    --data_dir "$STEP_DIR" --output_dir "$HC"

# 3. Self-supervised entity-specific VAEs (Section 4.3).
python $SCRIPT_DIR/train_face_vae.py   --joint_nurbs_dir "$JOINT" --output_dir "$FACE_OUT"   --split_file "$SPLIT" --epochs 100 --batch_size 64 --device cuda
python $SCRIPT_DIR/train_edge_vae.py   --joint_nurbs_dir "$JOINT" --output_dir "$EDGE_OUT"   --split_file "$SPLIT" --epochs 100 --batch_size 64 --device cuda
python $SCRIPT_DIR/train_vertex_vae.py --handcrafted_dir "$HC"   --output_dir "$VERTEX_OUT"  --split_file "$SPLIT" --epochs 100 --batch_size 64 --device cuda

# 4. Merge per-entity embeddings into a single per-model file.
python $SCRIPT_DIR/merge_embeddings.py \
    --face_dir   "$FACE_OUT/embeddings" \
    --edge_dir   "$EDGE_OUT/embeddings" \
    --vertex_dir "$VERTEX_OUT/embeddings" \
    --output_dir "$EMBED"

# 5. Build heterogeneous graphs with reduced handcrafted descriptors (Table 3).
FACE_KEEP="0,1,2,3,4,5,6,7,8,16"
EDGE_KEEP="0,1,2,3,4,5,7"
VERTEX_KEEP="0,1,2,3,4,6"
python $SCRIPT_DIR/build_graphs.py \
    --joint_nurbs_dir "$JOINT" --embed_dir "$EMBED" \
    --handcrafted_dir "$HC" --seg_dir "$SEG_DIR" \
    --output_dir "$GRAPHS" \
    --use_face --use_edge --use_vertex \
    --keep_face_hc_indices $FACE_KEEP \
    --keep_edge_hc_indices $EDGE_KEEP \
    --keep_vertex_hc_indices $VERTEX_KEEP

# 6. Train the heterogeneous GATv2 classifier.
python $SCRIPT_DIR/train_gatv2.py \
    --graphs_dir "$GRAPHS" --output_dir "$SCRIPT_DIR/models_topognn_full" \
    --split_file "$SPLIT" \
    --hidden_dim 128 --num_layers 4 --num_heads 8 \
    --epochs 200 --lr 5e-4 --batch_size 128 --patience 30 \
    --seed 42 --device cuda

echo "Done. Results: $SCRIPT_DIR/models_topognn_full/results.json"
