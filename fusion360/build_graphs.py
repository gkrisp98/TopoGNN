"""
Build Heterogeneous Graphs using Pre-computed Embeddings (Separate VAEs)
=========================================================================

Uses pre-computed face, edge, and vertex embeddings from the separate VAE
training pipeline (train_face_vae.py, train_edge_vae.py, train_vertex_vae.py
+ merge_embeddings.py).

Unlike build_fusion360_graphs.py which loads encoder checkpoints and encodes
edge/vertex on-the-fly, this script reads ALL embeddings from the merged
pickle files. No encoder code is needed.

Usage:
    python build_graphs_separate.py \\
        --joint_nurbs_dir /path/to/joint_nurbs \\
        --embed_dir /path/to/merged/embeddings \\
        --handcrafted_dir /path/to/handcrafted \\
        --seg_dir /path/to/seg \\
        --output_dir /path/to/graphs \\
        --device cuda
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
import pickle
from tqdm import tqdm
from typing import Dict, Optional, List


# =============================================================================
# FUSION 360 CLASSES
# =============================================================================

NUM_CLASSES = 8
CLASS_NAMES = [
    'ExtrudeSide',    # 0
    'ExtrudeEnd',     # 1
    'CutSide',        # 2
    'CutEnd',         # 3
    'Fillet',         # 4
    'Chamfer',        # 5
    'RevolveSide',    # 6
    'RevolveEnd',     # 7
]

MAX_CURVES = 8


# =============================================================================
# LABEL LOADING
# =============================================================================

def load_seg_file(seg_path: Path) -> Optional[List[int]]:
    if not seg_path.exists():
        return None
    labels = []
    with open(seg_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    labels.append(int(line))
                except ValueError:
                    continue
    return labels if labels else None


def load_labels_for_model(seg_dir: Optional[Path], model_name: str, n_faces: int) -> Optional[List[int]]:
    if seg_dir is None:
        return None
    labels = load_seg_file(seg_dir / f"{model_name}.seg")
    if labels is None:
        return None
    if len(labels) < n_faces:
        labels = labels + [-1] * (n_faces - len(labels))
    elif len(labels) > n_faces:
        labels = labels[:n_faces]
    return labels


# =============================================================================
# LOADING HELPERS
# =============================================================================

def load_handcrafted_features(handcrafted_dir: Path, model_name: str) -> Optional[Dict]:
    p = handcrafted_dir / f"{model_name}_handcrafted.pkl"
    if p.exists():
        with open(p, 'rb') as f:
            return pickle.load(f)
    return None


def load_embeddings(embed_file: Path):
    """
    Load merged embeddings from file.

    Returns dict with face_embeddings, edge_embeddings, vertex_embeddings,
    model_name, face_indices.
    """
    with open(embed_file, 'rb') as f:
        data = pickle.load(f)
    return data


# =============================================================================
# GRAPH BUILDING
# =============================================================================

def build_graph_for_model(
    model_name: str,
    joint_nurbs_path: Path,
    embeddings: Dict,
    handcrafted: Dict,
    embed_dim: int,
    labels: Optional[List[int]] = None,
    embed_scale: float = 0.25,
    use_face_embed: bool = True,
    use_edge_embed: bool = True,
    use_vertex_embed: bool = True,
    use_face_hc: bool = True,
    use_edge_hc: bool = True,
    use_vertex_hc: bool = True,
    drop_face_hc_indices: list = None,
    drop_edge_hc_indices: list = None,
    drop_vertex_hc_indices: list = None,
    keep_face_hc_indices: list = None,
    keep_edge_hc_indices: list = None,
    keep_vertex_hc_indices: list = None,
    remove_edge_nodes: bool = False,
    remove_vertex_nodes: bool = False,
    drop_relation_types: Optional[set] = None,
    random_embeddings: bool = False,
    random_seed: int = 42,
) -> Optional[HeteroData]:

    drop_rels = set(drop_relation_types) if drop_relation_types else set()

    with open(joint_nurbs_path, 'rb') as f:
        joint = pickle.load(f)

    face_data = joint['face_data']
    n_faces = int(joint.get('n_faces', handcrafted.get('n_faces', len(handcrafted.get('face_features', [])))))
    n_edges = int(joint.get('n_edges', handcrafted.get('n_edges', len(handcrafted.get('edge_features', [])))))
    n_vertices = int(joint.get('n_vertices', handcrafted.get('n_vertices', len(handcrafted.get('vertex_features', [])))))

    face_hc = handcrafted.get('face_features', np.zeros((n_faces, 17), dtype=np.float32))
    edge_hc = handcrafted.get('edge_features', np.zeros((n_edges, 8), dtype=np.float32))
    vertex_hc = handcrafted.get('vertex_features', np.zeros((n_vertices, 8), dtype=np.float32))

    # Extract boundary flag from edge_hc BEFORE any column selection/zeroing
    # (boundary is always at index 7 in the original 8D edge features)
    _orig_edge_boundary = edge_hc[:, 7:8] if edge_hc.ndim == 2 and edge_hc.shape[1] >= 8 else np.zeros((n_edges, 1), dtype=np.float32)

    # Select/zero handcrafted features for ablation
    if not use_face_hc:
        face_hc = np.zeros_like(face_hc)
    elif keep_face_hc_indices:
        valid = [i for i in keep_face_hc_indices if i < face_hc.shape[1]]
        face_hc = face_hc[:, valid].copy()
    elif drop_face_hc_indices:
        face_hc = face_hc.copy()
        for idx in drop_face_hc_indices:
            if idx < face_hc.shape[1]:
                face_hc[:, idx] = 0.0
    if not use_edge_hc:
        edge_hc = np.zeros_like(edge_hc)
    elif keep_edge_hc_indices:
        valid = [i for i in keep_edge_hc_indices if i < edge_hc.shape[1]]
        edge_hc = edge_hc[:, valid].copy()
    elif drop_edge_hc_indices:
        edge_hc = edge_hc.copy()
        for idx in drop_edge_hc_indices:
            if idx < edge_hc.shape[1]:
                edge_hc[:, idx] = 0.0
    if not use_vertex_hc:
        vertex_hc = np.zeros_like(vertex_hc)
    elif keep_vertex_hc_indices:
        valid = [i for i in keep_vertex_hc_indices if i < vertex_hc.shape[1]]
        vertex_hc = vertex_hc[:, valid].copy()
    elif drop_vertex_hc_indices:
        vertex_hc = vertex_hc.copy()
        for idx in drop_vertex_hc_indices:
            if idx < vertex_hc.shape[1]:
                vertex_hc[:, idx] = 0.0

    # ---- Face embeddings placement via face_indices ----
    face_embed_raw = embeddings.get('face_embeddings', None) if use_face_embed else None
    face_embed_indices = embeddings.get('face_indices', None)

    if face_embed_raw is not None and len(face_embed_raw) > 0:
        face_embed = np.zeros((n_faces, face_embed_raw.shape[1]), dtype=np.float32)
        if face_embed_indices is not None:
            for i, fi in enumerate(face_embed_indices):
                if 0 <= fi < n_faces and i < face_embed_raw.shape[0]:
                    face_embed[fi] = face_embed_raw[i]
        else:
            for i, fd in enumerate(face_data):
                fi = int(fd.get('face_idx', i))
                if 0 <= fi < n_faces and i < face_embed_raw.shape[0]:
                    face_embed[fi] = face_embed_raw[i]
    else:
        face_embed = None

    # ---- Edge embeddings (pre-computed) ----
    edge_embed_raw = embeddings.get('edge_embeddings', None) if use_edge_embed else None
    if edge_embed_raw is not None and len(edge_embed_raw) > 0:
        # edge_embeddings are aligned with edge_data (same order)
        if edge_embed_raw.shape[0] == n_edges:
            edge_embed = edge_embed_raw.astype(np.float32)
        else:
            # Pad/truncate to n_edges
            edge_embed = np.zeros((n_edges, embed_dim), dtype=np.float32)
            n_copy = min(edge_embed_raw.shape[0], n_edges)
            edge_embed[:n_copy] = edge_embed_raw[:n_copy]
    else:
        edge_embed = None

    # ---- Vertex embeddings (pre-computed) ----
    vertex_embed_raw = embeddings.get('vertex_embeddings', None) if use_vertex_embed else None
    if vertex_embed_raw is not None and len(vertex_embed_raw) > 0:
        if vertex_embed_raw.shape[0] == n_vertices:
            vertex_embed = vertex_embed_raw.astype(np.float32)
        else:
            vertex_embed = np.zeros((n_vertices, embed_dim), dtype=np.float32)
            n_copy = min(vertex_embed_raw.shape[0], n_vertices)
            vertex_embed[:n_copy] = vertex_embed_raw[:n_copy]
    else:
        vertex_embed = None

    # ---- Random embeddings override ----
    if random_embeddings:
        rng = np.random.RandomState(abs(hash(model_name)) % (2**31))
        if use_face_embed:
            face_embed = rng.randn(n_faces, embed_dim).astype(np.float32) * 0.1
        if use_edge_embed:
            edge_embed = rng.randn(n_edges, embed_dim).astype(np.float32) * 0.1
        if use_vertex_embed:
            vertex_embed = rng.randn(n_vertices, embed_dim).astype(np.float32) * 0.1

    # ---- Edge topology features ----
    edge_data = joint.get('edge_data', None)
    if edge_data is not None:
        edge_has = np.array([ed.get('has_curve', 1.0) for ed in edge_data], dtype=np.float32)
        v_start_idx = np.array([ed.get('v_start_idx', -1) for ed in edge_data], dtype=np.int32)
        v_end_idx = np.array([ed.get('v_end_idx', -1) for ed in edge_data], dtype=np.int32)
    else:
        edge_has = np.zeros((n_edges,), dtype=np.float32)
        v_start_idx = np.full((n_edges,), -1, dtype=np.int32)
        v_end_idx = np.full((n_edges,), -1, dtype=np.int32)

    edge_topo = np.concatenate([edge_has.reshape(-1, 1), _orig_edge_boundary], axis=1).astype(np.float32)

    # ---- Node feature assembly ----
    if face_embed is not None:
        face_x = np.concatenate([face_embed * embed_scale, face_hc.astype(np.float32)], axis=1)
    else:
        face_x = face_hc.astype(np.float32)

    if edge_embed is not None:
        edge_x = np.concatenate([edge_embed * embed_scale, edge_hc.astype(np.float32), edge_topo], axis=1)
    else:
        edge_x = np.concatenate([edge_hc.astype(np.float32), edge_topo], axis=1)

    if vertex_embed is not None:
        vertex_x = np.concatenate([vertex_embed * embed_scale, vertex_hc.astype(np.float32)], axis=1)
    else:
        vertex_x = vertex_hc.astype(np.float32)

    data = HeteroData()
    data['face'].x = torch.from_numpy(face_x)
    data['edge'].x = torch.from_numpy(edge_x)
    data['vertex'].x = torch.from_numpy(vertex_x)

    # Labels
    if labels is not None:
        if len(labels) < n_faces:
            labels = labels + [-1] * (n_faces - len(labels))
        elif len(labels) > n_faces:
            labels = labels[:n_faces]
        data['face'].y = torch.LongTensor(labels)

    # ---- Relations: face -> edge (directed by orientation) ----
    fwd_src, fwd_dst = [], []
    rev_src, rev_dst = [], []

    for fd in face_data:
        fi = int(fd.get('face_idx', -1))
        if fi < 0 or fi >= n_faces:
            continue

        ce = fd.get('curve_edge_idx', None)
        cm = fd.get('curves_mask', None)
        co = fd.get('curve_orientation', None)
        if ce is None or cm is None or co is None:
            continue

        for slot in range(min(len(ce), MAX_CURVES)):
            if float(cm[slot]) <= 0:
                continue
            e_idx = int(ce[slot])
            if e_idx < 0 or e_idx >= n_edges:
                continue
            is_fwd = float(co[slot]) > 0.5
            if is_fwd:
                fwd_src.append(fi); fwd_dst.append(e_idx)
            else:
                rev_src.append(fi); rev_dst.append(e_idx)

    if len(fwd_src) > 0:
        if 'uses_fwd' not in drop_rels:
            data['face', 'uses_fwd', 'edge'].edge_index = torch.tensor([fwd_src, fwd_dst], dtype=torch.long)
        if 'used_by_fwd' not in drop_rels:
            data['edge', 'used_by_fwd', 'face'].edge_index = torch.tensor([fwd_dst, fwd_src], dtype=torch.long)

    if len(rev_src) > 0:
        if 'uses_rev' not in drop_rels:
            data['face', 'uses_rev', 'edge'].edge_index = torch.tensor([rev_src, rev_dst], dtype=torch.long)
        if 'used_by_rev' not in drop_rels:
            data['edge', 'used_by_rev', 'face'].edge_index = torch.tensor([rev_dst, rev_src], dtype=torch.long)

    # ---- Loop relations ----
    loop_src, loop_dst = [], []
    for fd in face_data:
        ce = fd.get('curve_edge_idx', None)
        cm = fd.get('curves_mask', None)
        cc = fd.get('curve_connectivity', None)

        if ce is None or cm is None or cc is None:
            continue

        for slot in range(min(len(ce), MAX_CURVES)):
            if float(cm[slot]) <= 0:
                continue
            e_idx = int(ce[slot])
            if e_idx < 0 or e_idx >= n_edges:
                continue

            next_slot = int(cc[slot, 1]) if cc.ndim == 2 and slot < cc.shape[0] else -1
            if 0 <= next_slot < min(len(ce), MAX_CURVES) and float(cm[next_slot]) > 0:
                next_e_idx = int(ce[next_slot])
                if 0 <= next_e_idx < n_edges and next_e_idx != e_idx:
                    loop_src.append(e_idx)
                    loop_dst.append(next_e_idx)

    if len(loop_src) > 0 and 'next_in_loop' not in drop_rels:
        data['edge', 'next_in_loop', 'edge'].edge_index = torch.tensor([loop_src, loop_dst], dtype=torch.long)

    # ---- Relations: edge -> vertex ----
    e2v_src, e2v_dst, e2v_attr = [], [], []
    for e_idx in range(n_edges):
        vs = int(v_start_idx[e_idx]) if e_idx < len(v_start_idx) else -1
        ve = int(v_end_idx[e_idx]) if e_idx < len(v_end_idx) else -1

        if 0 <= vs < n_vertices:
            e2v_src.append(e_idx); e2v_dst.append(vs); e2v_attr.append([1.0])
        if 0 <= ve < n_vertices:
            e2v_src.append(e_idx); e2v_dst.append(ve); e2v_attr.append([0.0])

    if len(e2v_src) > 0:
        if 'has' not in drop_rels:
            data['edge', 'has', 'vertex'].edge_index = torch.tensor([e2v_src, e2v_dst], dtype=torch.long)
            data['edge', 'has', 'vertex'].edge_attr = torch.tensor(e2v_attr, dtype=torch.float32)
        if 'belongs_to' not in drop_rels:
            data['vertex', 'belongs_to', 'edge'].edge_index = torch.tensor([e2v_dst, e2v_src], dtype=torch.long)

    # ---- Face-face adjacency ----
    edge_to_faces = handcrafted.get('edge_to_faces', {})
    adjacency_set = set()

    if isinstance(edge_to_faces, dict):
        for _, faces_list in edge_to_faces.items():
            if not isinstance(faces_list, list):
                continue
            valid_faces = [int(f) for f in faces_list if 0 <= int(f) < n_faces]
            for i in range(len(valid_faces)):
                for j in range(i + 1, len(valid_faces)):
                    a, b = valid_faces[i], valid_faces[j]
                    adjacency_set.add((min(a, b), max(a, b)))

    ff_src, ff_dst = [], []
    for a, b in adjacency_set:
        ff_src.extend([a, b])
        ff_dst.extend([b, a])

    if len(ff_src) > 0 and 'adjacent_to' not in drop_rels:
        data['face', 'adjacent_to', 'face'].edge_index = torch.tensor([ff_src, ff_dst], dtype=torch.long)

    data.model_name = model_name
    data.n_faces = n_faces
    data.n_edges = n_edges
    data.n_vertices = n_vertices

    # ---- Topology ablation: remove node types entirely ----
    if remove_edge_nodes:
        edge_rel_types = [et for et in data.edge_types if 'edge' in et[0] or 'edge' in et[2]]
        for et in edge_rel_types:
            del data[et]
        if 'edge' in data.node_types:
            del data['edge']

    if remove_vertex_nodes:
        vertex_rel_types = [et for et in data.edge_types if 'vertex' in et[0] or 'vertex' in et[2]]
        for et in vertex_rel_types:
            del data[et]
        if 'vertex' in data.node_types:
            del data['vertex']

    return data


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build graphs using pre-computed embeddings (separate VAEs)")
    parser.add_argument("--joint_nurbs_dir", type=str, required=True,
                        help="Directory with *_joint_nurbs.pkl files")
    parser.add_argument("--embed_dir", type=str, required=True,
                        help="Directory with merged *_embeddings.pkl (from merge_embeddings.py)")
    parser.add_argument("--handcrafted_dir", type=str, required=True,
                        help="Directory with *_handcrafted.pkl files")
    parser.add_argument("--seg_dir", type=str, default=None,
                        help="Directory with .seg label files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for *_graph.pt files")
    parser.add_argument("--embed_scale", type=float, default=1,
                        help="Scale factor for embeddings")
    parser.add_argument("--use_face", action="store_true", default=True,
                        help="Include face VAE embeddings (default: True)")
    parser.add_argument("--no_face", action="store_true",
                        help="Exclude face VAE embeddings")
    parser.add_argument("--use_edge", action="store_true", default=True,
                        help="Include edge VAE embeddings (default: True)")
    parser.add_argument("--no_edge", action="store_true",
                        help="Exclude edge VAE embeddings")
    parser.add_argument("--use_vertex", action="store_true", default=True,
                        help="Include vertex VAE embeddings (default: True)")
    parser.add_argument("--no_vertex", action="store_true",
                        help="Exclude vertex VAE embeddings")
    parser.add_argument("--no_face_hc", action="store_true",
                        help="Zero out face handcrafted features")
    parser.add_argument("--no_edge_hc", action="store_true",
                        help="Zero out edge handcrafted features")
    parser.add_argument("--no_vertex_hc", action="store_true",
                        help="Zero out vertex handcrafted features")
    parser.add_argument("--drop_face_hc_indices", type=str, default=None,
                        help="Comma-separated indices of face HC features to zero out (e.g. '9,10,11,12,13,14,15')")
    parser.add_argument("--drop_edge_hc_indices", type=str, default=None,
                        help="Comma-separated indices of edge HC features to zero out (e.g. '6')")
    parser.add_argument("--drop_vertex_hc_indices", type=str, default=None,
                        help="Comma-separated indices of vertex HC features to zero out (e.g. '5,7')")
    parser.add_argument("--keep_face_hc_indices", type=str, default=None,
                        help="Comma-separated indices of face HC features to KEEP (removes others, e.g. '0,1,2,3,4,5,6,7,8,16')")
    parser.add_argument("--keep_edge_hc_indices", type=str, default=None,
                        help="Comma-separated indices of edge HC features to KEEP (e.g. '0,1,2,3,4,5,7')")
    parser.add_argument("--keep_vertex_hc_indices", type=str, default=None,
                        help="Comma-separated indices of vertex HC features to KEEP (e.g. '0,1,2,3,4,6')")
    parser.add_argument("--remove_edge_nodes", action="store_true",
                        help="Remove edge nodes entirely from graph (topology ablation)")
    parser.add_argument("--remove_vertex_nodes", action="store_true",
                        help="Remove vertex nodes entirely from graph (topology ablation)")
    parser.add_argument("--random_embeddings", action="store_true",
                        help="Replace VAE embeddings with random vectors (no-VAE ablation)")
    parser.add_argument("--drop_relation_types", type=str, default=None,
                        help="Comma-separated relation names to drop from the graph (e.g. 'next_in_loop,has,belongs_to'). Valid names: uses_fwd, uses_rev, used_by_fwd, used_by_rev, next_in_loop, has, belongs_to, adjacent_to")
    args = parser.parse_args()

    # Handle --no_X flags
    if args.no_face:
        args.use_face = False
    if args.no_edge:
        args.use_edge = False
    if args.no_vertex:
        args.use_vertex = False

    # Parse drop indices
    def _parse_indices(s):
        if s is None:
            return None
        return [int(x.strip()) for x in s.split(',') if x.strip()]

    drop_face_hc = _parse_indices(args.drop_face_hc_indices)
    drop_edge_hc = _parse_indices(args.drop_edge_hc_indices)
    drop_vertex_hc = _parse_indices(args.drop_vertex_hc_indices)
    keep_face_hc = _parse_indices(args.keep_face_hc_indices)
    keep_edge_hc = _parse_indices(args.keep_edge_hc_indices)
    keep_vertex_hc = _parse_indices(args.keep_vertex_hc_indices)

    drop_relation_types = None
    if args.drop_relation_types:
        drop_relation_types = {r.strip() for r in args.drop_relation_types.split(',') if r.strip()}

    joint_nurbs_dir = Path(args.joint_nurbs_dir)
    embed_dir = Path(args.embed_dir)
    handcrafted_dir = Path(args.handcrafted_dir)
    seg_dir = Path(args.seg_dir) if args.seg_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embed_files = sorted(embed_dir.glob("*_embeddings.pkl"))

    print(f"\n{'='*70}")
    print("BUILDING GRAPHS (Pre-computed Embeddings from Separate VAEs)")
    print(f"{'='*70}")
    print(f"Embedding files: {len(embed_files)}")
    print(f"Classes: {NUM_CLASSES} ({', '.join(CLASS_NAMES)})")

    if seg_dir:
        seg_files = list(seg_dir.glob("*.seg"))
        print(f"Seg files: {len(seg_files)}")

    # Detect embed_dim from first file
    embed_dim = 64
    if embed_files:
        sample = load_embeddings(embed_files[0])
        fe = sample.get('face_embeddings', None)
        if fe is not None and len(fe) > 0:
            embed_dim = int(fe.shape[1])
    print(f"Embed dim: {embed_dim}")
    print(f"Embed scale: {args.embed_scale}")
    print(f"Embeddings: Face={'ON' if args.use_face else 'OFF'}, Edge={'ON' if args.use_edge else 'OFF'}, Vertex={'ON' if args.use_vertex else 'OFF'}")
    print(f"Handcrafted: Face={'OFF' if args.no_face_hc else 'ON'}, Edge={'OFF' if args.no_edge_hc else 'ON'}, Vertex={'OFF' if args.no_vertex_hc else 'ON'}")
    if drop_face_hc:
        print(f"  Face HC dropped indices: {drop_face_hc}")
    if drop_edge_hc:
        print(f"  Edge HC dropped indices: {drop_edge_hc}")
    if drop_vertex_hc:
        print(f"  Vertex HC dropped indices: {drop_vertex_hc}")
    if keep_face_hc:
        print(f"  Face HC KEEP indices: {keep_face_hc}")
    if keep_edge_hc:
        print(f"  Edge HC KEEP indices: {keep_edge_hc}")
    if keep_vertex_hc:
        print(f"  Vertex HC KEEP indices: {keep_vertex_hc}")
    if args.remove_edge_nodes:
        print("  REMOVING edge nodes from graph (topology ablation)")
    if args.remove_vertex_nodes:
        print("  REMOVING vertex nodes from graph (topology ablation)")
    if args.random_embeddings:
        print("  USING RANDOM EMBEDDINGS (no-VAE ablation)")
    if drop_relation_types:
        print(f"  DROPPING relation types: {sorted(drop_relation_types)}")

    print(f"\n{'='*70}")
    print("BUILDING GRAPHS")
    print(f"{'='*70}")

    success = 0
    failed = 0
    with_labels = 0
    skipped_no_labels = 0

    for f in tqdm(embed_files, desc="Building graphs"):
        embeddings = load_embeddings(f)
        model_name = embeddings.get('model_name', f.stem.replace('_embeddings', ''))

        joint_path = joint_nurbs_dir / f"{model_name}_joint_nurbs.pkl"
        if not joint_path.exists():
            failed += 1
            continue

        handcrafted = load_handcrafted_features(handcrafted_dir, model_name)
        if handcrafted is None:
            failed += 1
            continue

        n_faces_estimate = len(handcrafted.get('face_features', []))
        labels = load_labels_for_model(seg_dir, model_name, n_faces_estimate)

        if labels is not None:
            with_labels += 1
        else:
            skipped_no_labels += 1

        try:
            g = build_graph_for_model(
                model_name=model_name,
                joint_nurbs_path=joint_path,
                embeddings=embeddings,
                handcrafted=handcrafted,
                embed_dim=embed_dim,
                labels=labels,
                embed_scale=args.embed_scale,
                use_face_embed=args.use_face,
                use_edge_embed=args.use_edge,
                use_vertex_embed=args.use_vertex,
                use_face_hc=not args.no_face_hc,
                use_edge_hc=not args.no_edge_hc,
                use_vertex_hc=not args.no_vertex_hc,
                drop_face_hc_indices=drop_face_hc,
                drop_edge_hc_indices=drop_edge_hc,
                drop_vertex_hc_indices=drop_vertex_hc,
                keep_face_hc_indices=keep_face_hc,
                keep_edge_hc_indices=keep_edge_hc,
                keep_vertex_hc_indices=keep_vertex_hc,
                remove_edge_nodes=args.remove_edge_nodes,
                remove_vertex_nodes=args.remove_vertex_nodes,
                drop_relation_types=drop_relation_types,
                random_embeddings=args.random_embeddings,
            )
            if g is None:
                failed += 1
                continue

            torch.save(g, output_dir / f"{model_name}_graph.pt")
            success += 1
        except Exception as e:
            print(f"\n  Error on {model_name}: {e}")
            failed += 1

    # Summary
    print(f"\n{'='*70}")
    print("GRAPH BUILDING COMPLETE")
    print(f"{'='*70}")
    print(f"Success: {success}")
    print(f"  With labels: {with_labels}")
    print(f"  Without labels: {skipped_no_labels}")
    print(f"Failed: {failed}")
    print(f"Output: {output_dir}")

    sample_graphs = list(output_dir.glob("*_graph.pt"))
    if sample_graphs:
        g = torch.load(sample_graphs[0], weights_only=False)
        print(f"\nNode feature dimensions (from {sample_graphs[0].stem}):")
        for nt in ['face', 'edge', 'vertex']:
            if nt in g.node_types and hasattr(g[nt], 'x') and g[nt].x is not None:
                print(f"  {nt.capitalize():8s}: {g[nt].x.shape[1]}")

        rel_info = []
        for edge_type in g.edge_types:
            ei = g[edge_type].edge_index
            rel_info.append(f"  {edge_type}: {ei.shape[1]} edges")
        print(f"\nRelation types ({len(g.edge_types)}):")
        for r in rel_info:
            print(r)

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

#python build_graphs_separate.py --joint_nurbs_dir /home/konstantinos/Downloads/s2.0.0/breps/nurbs_10x10 --embed_dir /home/konstantinos/Downloads/s2.0.0/breps/embeddings_merged --handcrafted_dir /home/konstantinos/Downloads/s2.0.0/breps/handcrafted --seg_dir /home/konstantinos/Downloads/s2.0.0/breps/seg --output_dir /home/konstantinos/Downloads/s2.0.0/breps/graphs_separate
