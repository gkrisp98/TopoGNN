#!/bin/bash
# Reproduce all Fusion 360 ablation results from the TopoGNN paper (Tables 8-11).
#
# Pre-requisites:
#   1. NURBS + UV features extracted to $JOINT  (see extract_nurbs.py)
#   2. Handcrafted features extracted to $HC     (see extract_handcrafted.py)
#   3. Per-entity VAE embeddings merged to $EMBED (see train_*_vae.py + merge_embeddings.py)
#   4. UV-only and NURBS-only ablation embeddings ($EMBED_UV_ONLY, $EMBED_NURBS_ONLY)
#      produced by re-training the Face VAE with the corresponding branch disabled
#      and re-merging.
#
# Usage:
#   bash run_ablations.sh /path/to/breps_root /path/to/repo/fusion360
#
# Result models are written to $REPO/models_rhc_*; numbers should match
# fusion360/results/models_rhc_*/results.json.

set -e

BASE=${1:-/path/to/breps}
SCRIPT_DIR=${2:-$(cd "$(dirname "$0")/.." && pwd)}

JOINT="$BASE/nurbs_10x10"
HC="$BASE/handcrafted"
SEG="$BASE/seg"
EMBED="$BASE/embeddings_merged"
EMBED_UV_ONLY="$BASE/embeddings_merged_uv_only"
EMBED_NURBS_ONLY="$BASE/embeddings_merged_nurbs_only"
SPLIT="$SCRIPT_DIR/train_test.json"

# Reduced handcrafted feature subset reported in Table 3 of the paper.
FACE_KEEP="0,1,2,3,4,5,6,7,8,16"
EDGE_KEEP="0,1,2,3,4,5,7"
VERTEX_KEEP="0,1,2,3,4,6"
COMMON_HC="--keep_face_hc_indices $FACE_KEEP --keep_edge_hc_indices $EDGE_KEEP --keep_vertex_hc_indices $VERTEX_KEEP"

GNN_BASE="python $SCRIPT_DIR/train_gatv2.py --split_file $SPLIT \
  --hidden_dim 128 --num_layers 6 --num_heads 4 \
  --lr 0.001 --batch_size 128 --seed 42 --epochs 200 --device cuda"

build () {  # build $name $extra_flags  [$embed_dir]
    local NAME=$1; local FLAGS=$2; local E=${3:-$EMBED}
    python $SCRIPT_DIR/build_graphs.py \
        --joint_nurbs_dir "$JOINT" --embed_dir "$E" \
        --handcrafted_dir "$HC" --seg_dir "$SEG" \
        --output_dir "$BASE/graphs_rhc_$NAME" \
        $FLAGS
}
train () {  # train $name [$extra_flags]
    local NAME=$1; shift
    $GNN_BASE --graphs_dir "$BASE/graphs_rhc_$NAME" \
              --output_dir "$SCRIPT_DIR/models_rhc_$NAME" "$@"
}

# Table 8 (Handcrafted features) + Tables 9-11 share the same baseline.
build baseline    "--use_face --use_edge --use_vertex $COMMON_HC"
# The baseline row was trained with 4 layers, unlike the other ablations (6).
train baseline --num_layers 4

build no_hc       "--use_face --use_edge --use_vertex --no_face_hc --no_edge_hc --no_vertex_hc"
train no_hc

build hc_only     "--no_face --no_edge --no_vertex $COMMON_HC"
train hc_only

# Table 9 (Topology)
build face_only   "--use_face --use_edge --use_vertex $COMMON_HC --remove_edge_nodes --remove_vertex_nodes"
train face_only
build no_edges    "--use_face --use_edge --use_vertex $COMMON_HC --remove_edge_nodes"
train no_edges
build no_vertices "--use_face --use_edge --use_vertex $COMMON_HC --remove_vertex_nodes"
train no_vertices
build 4relations  "--use_face --use_edge --use_vertex $COMMON_HC --drop_relation_types next_in_loop,has,belongs_to"
train 4relations

# Table 10 (Geometric inputs)
build uv_only    "--use_face --use_edge --use_vertex $COMMON_HC" "$EMBED_UV_ONLY"
train uv_only
build nurbs_only "--use_face --use_edge --use_vertex $COMMON_HC" "$EMBED_NURBS_ONLY"
train nurbs_only

# Table 11 (Capacity & pretraining)
$GNN_BASE --graphs_dir "$BASE/graphs_rhc_baseline" \
          --output_dir "$SCRIPT_DIR/models_rhc_simplified" \
          --hidden_dim 64 --num_layers 2 --num_heads 2

build no_vae "--use_face --use_edge --use_vertex $COMMON_HC --random_embeddings"
train no_vae

echo "All ablations complete. Compare against fusion360/results/models_rhc_*/results.json"
