"""
Build Heterogeneous Graphs with EDGE Nodes for MechCAD Classification
======================================================================

Adapted from Fusion 360 edge-based graph builder (build_graphs.py) for
MechCAD classification task.

Key differences from coedge variant (build_mechcad_graphs_coedge.py):
- Node types: face, edge, vertex (NOT face, coedge, vertex)
- Edges are unique topological entities (not duplicated per face-use)
- Relations: uses_fwd, uses_rev, next_in_loop, has(edge→vertex),
  adjacent_to(face→face) + reverses
- Label is GRAPH-LEVEL (data.y = class_label) not per-face

Usage:
    # WITH VAE embeddings:
    python build_mechcad_graphs_edge.py \
        --joint_nurbs_dir ./mechcad/nurbs \
        --joint_embed_dir ./mechcad/encoders/embeddings \
        --handcrafted_dir ./mechcad/handcrafted \
        --encoder_ckpt ./mechcad/encoders/hetero_vae_best.pt \
        --output_dir ./mechcad/graphs_edge \
        --device cuda

    # HANDCRAFTED ONLY:
    python build_mechcad_graphs_edge.py \
        --joint_nurbs_dir ./mechcad/nurbs \
        --handcrafted_dir ./mechcad/handcrafted \
        --output_dir ./mechcad/graphs_edge_handcrafted \
        --no_embeddings
"""

import warnings
warnings.filterwarnings('ignore')

import math
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from pathlib import Path
import pickle
import json
from tqdm import tqdm
from typing import Dict, Optional, List
from collections import defaultdict


# =============================================================================
# CONFIG - Must match extraction scripts
# =============================================================================

MAX_CURVE_CTRL = 20
MAX_CURVE_KNOTS = 25
MAX_CURVES = 8


# =============================================================================
# ENCODERS (same architecture as train_encoder.py)
# =============================================================================

def normalize_edge_coords(curve_pw, curve_mask):
    xyz = curve_pw[..., :3]
    mask_exp = curve_mask.unsqueeze(-1)
    sum_coords = (xyz * mask_exp).sum(dim=1)
    count = mask_exp.sum(dim=(1, 2)).clamp(min=1.0)
    centroid = sum_coords / count.unsqueeze(-1)
    curve_pw_norm = curve_pw.clone()
    curve_pw_norm[..., :3] = curve_pw[..., :3] - centroid.unsqueeze(1)
    return curve_pw_norm


def normalize_vertex_positions(vertex_features: np.ndarray) -> np.ndarray:
    if vertex_features is None or len(vertex_features) == 0:
        return vertex_features

    vertex_features = np.asarray(vertex_features, dtype=np.float32).copy()
    xyz = vertex_features[:, :3]

    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    max_dist = np.linalg.norm(centered, axis=1).max()
    scale = max_dist if max_dist > 1e-6 else 1.0

    vertex_features[:, :3] = centered / scale
    return vertex_features


class EdgeEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64, normalize_coords: bool = True):
        super().__init__()
        self.normalize_coords = normalize_coords
        self.conv = nn.Sequential(
            nn.Conv1d(4, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.knot_embed = nn.Linear(MAX_CURVE_KNOTS, 32)
        self.mlp = nn.Sequential(
            nn.Linear(64 * 4 + 32, 128), nn.GELU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, curve_pw, curve_u, curve_mask=None):
        bs = curve_pw.shape[0]
        if self.normalize_coords and curve_mask is not None:
            curve_pw = normalize_edge_coords(curve_pw, curve_mask)
        x = curve_pw.permute(0, 2, 1)
        if curve_mask is not None:
            x = x * curve_mask.unsqueeze(1)
        conv_feat = self.conv(x).contiguous().view(bs, -1)
        knot_feat = self.knot_embed(curve_u)
        return self.mlp(torch.cat([conv_feat, knot_feat], dim=-1))


class VertexEncoder(nn.Module):
    def __init__(self, input_dim: int = 8, embed_dim: int = 64, num_frequencies: int = 10):
        super().__init__()
        self.num_frequencies = num_frequencies
        pos_feat_dim = 3 * 2 * num_frequencies

        self.mlp = nn.Sequential(
            nn.Linear(pos_feat_dim + input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        frequencies = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('frequencies', frequencies)

    def fourier_encode(self, positions):
        pos_expanded = positions.unsqueeze(-1) * self.frequencies
        feats = torch.cat([torch.sin(pos_expanded * math.pi), torch.cos(pos_expanded * math.pi)], dim=-1)
        return feats.view(positions.shape[0], -1)

    def forward(self, vertex_features):
        positions = vertex_features[:, :3]
        fourier_feats = self.fourier_encode(positions)
        return self.mlp(torch.cat([fourier_feats, vertex_features], dim=-1))


def load_encoders(ckpt_path: Path, device: torch.device, vertex_dim: int, embed_dim_guess: int):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    embed_dim = int(cfg.get('embed_dim', embed_dim_guess))

    edge_encoder = None
    vertex_encoder = None
    edge_mu = None
    vertex_mu = None

    if 'edge_encoder_state_dict' in ckpt:
        edge_encoder = EdgeEncoder(embed_dim=embed_dim).to(device)
        edge_encoder.load_state_dict(ckpt['edge_encoder_state_dict'])
        edge_encoder.eval()
        print(f"  Loaded EdgeEncoder (embed_dim={embed_dim})")
    else:
        print("  WARNING: No edge_encoder_state_dict; edge embeddings will be zeros.")

    if 'vertex_encoder_state_dict' in ckpt:
        vertex_encoder = VertexEncoder(input_dim=vertex_dim, embed_dim=embed_dim).to(device)
        vertex_encoder.load_state_dict(ckpt['vertex_encoder_state_dict'])
        vertex_encoder.eval()
        print(f"  Loaded VertexEncoder (input_dim={vertex_dim})")
    else:
        print("  WARNING: No vertex_encoder_state_dict; vertex embeddings will be omitted.")

    if 'model_state_dict' in ckpt:
        msd = ckpt['model_state_dict']

        if 'edge_mu.weight' in msd and 'edge_mu.bias' in msd:
            edge_mu = nn.Linear(embed_dim, embed_dim).to(device)
            edge_mu.load_state_dict({
                'weight': msd['edge_mu.weight'],
                'bias': msd['edge_mu.bias']
            })
            edge_mu.eval()
            print("  Loaded edge_mu layer")

        if 'vertex_mu.weight' in msd and 'vertex_mu.bias' in msd:
            vertex_mu = nn.Linear(embed_dim, embed_dim).to(device)
            vertex_mu.load_state_dict({
                'weight': msd['vertex_mu.weight'],
                'bias': msd['vertex_mu.bias']
            })
            vertex_mu.eval()
            print("  Loaded vertex_mu layer")

    return embed_dim, edge_encoder, vertex_encoder, edge_mu, vertex_mu


# =============================================================================
# LOADING HELPERS
# =============================================================================

def load_handcrafted_features(handcrafted_dir: Path, model_name: str) -> Optional[Dict]:
    p = handcrafted_dir / f"{model_name}_handcrafted.pkl"
    if p.exists():
        with open(p, 'rb') as f:
            return pickle.load(f)
    return None


def load_face_embeddings(embed_file: Path):
    with open(embed_file, 'rb') as f:
        data = pickle.load(f)
    embeddings = data['face_embeddings']
    model_name = data.get('model_name', embed_file.stem.replace('_embeddings', ''))
    face_indices = data.get('face_indices', None)
    return embeddings, model_name, face_indices


# =============================================================================
# GRAPH BUILDING (EDGE VARIANT - CLASSIFICATION)
# =============================================================================

def build_graph_for_model(
    model_name: str,
    joint_nurbs_path: Path,
    face_embed_raw: Optional[np.ndarray],
    face_embed_indices: Optional[List[int]],
    handcrafted: Dict,
    device: torch.device,
    embed_dim: int,
    edge_encoder: Optional[nn.Module],
    vertex_encoder: Optional[nn.Module],
    edge_mu: Optional[nn.Module] = None,
    vertex_mu: Optional[nn.Module] = None,
    class_label: Optional[int] = None,
    class_name: Optional[str] = None,
    use_embeddings: bool = True,
    embed_scale: float = 0.25,
) -> Optional[HeteroData]:

    with open(joint_nurbs_path, 'rb') as f:
        joint = pickle.load(f)

    face_data = joint['face_data']
    n_faces = int(joint.get('n_faces', handcrafted.get('n_faces', len(handcrafted.get('face_features', [])))))
    n_edges = int(joint.get('n_edges', handcrafted.get('n_edges', len(handcrafted.get('edge_features', [])))))
    n_vertices = int(joint.get('n_vertices', handcrafted.get('n_vertices', len(handcrafted.get('vertex_features', [])))))

    face_hc = handcrafted.get('face_features', np.zeros((n_faces, 17), dtype=np.float32))
    edge_hc = handcrafted.get('edge_features', np.zeros((n_edges, 8), dtype=np.float32))
    vertex_hc = handcrafted.get('vertex_features', np.zeros((n_vertices, 8), dtype=np.float32))

    # ---- Face embeddings placement via face_indices ----
    if use_embeddings and face_embed_raw is not None:
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

    # ---- Edge curve data (from joint['edge_data']) ----
    edge_data = joint.get('edge_data', None)

    if edge_data is None:
        edge_pw = np.zeros((n_edges, MAX_CURVE_CTRL, 4), dtype=np.float32)
        edge_u = np.zeros((n_edges, MAX_CURVE_KNOTS), dtype=np.float32)
        edge_mask = np.zeros((n_edges, MAX_CURVE_CTRL), dtype=np.float32)
        edge_has = np.zeros((n_edges,), dtype=np.float32)
        v_start_idx = np.full((n_edges,), -1, dtype=np.int32)
        v_end_idx = np.full((n_edges,), -1, dtype=np.int32)
    else:
        edge_pw = np.stack([ed['curve_pw'] for ed in edge_data], axis=0).astype(np.float32)
        edge_u = np.stack([ed['curve_u'] for ed in edge_data], axis=0).astype(np.float32)
        edge_mask = np.stack([ed['curve_mask'] for ed in edge_data], axis=0).astype(np.float32)
        edge_has = np.array([ed.get('has_curve', 1.0) for ed in edge_data], dtype=np.float32)
        v_start_idx = np.array([ed.get('v_start_idx', -1) for ed in edge_data], dtype=np.int32)
        v_end_idx = np.array([ed.get('v_end_idx', -1) for ed in edge_data], dtype=np.int32)

    # ---- Edge embeddings ----
    if use_embeddings and edge_encoder is not None:
        edge_embed = np.zeros((n_edges, embed_dim), dtype=np.float32)
        edge_encoder.eval()
        bs = 256
        with torch.no_grad():
            for s in range(0, n_edges, bs):
                pw = torch.from_numpy(edge_pw[s:s+bs]).to(device)
                uu = torch.from_numpy(edge_u[s:s+bs]).to(device)
                mm = torch.from_numpy(edge_mask[s:s+bs]).to(device)
                z = edge_encoder(pw, uu, mm)
                if edge_mu is not None:
                    z = edge_mu(z)
                edge_embed[s:s+bs] = z.detach().cpu().numpy()
    else:
        edge_embed = None

    # Edge topology features: [has_curve, is_boundary_edge]
    is_boundary_edge = edge_hc[:, 7:8] if edge_hc.ndim == 2 and edge_hc.shape[1] >= 8 else np.zeros((n_edges, 1), dtype=np.float32)
    edge_topo = np.concatenate([edge_has.reshape(-1, 1), is_boundary_edge], axis=1).astype(np.float32)

    # ---- Vertex embeddings ----
    if use_embeddings and vertex_encoder is not None and vertex_hc is not None and len(vertex_hc) == n_vertices:
        vertex_embed = np.zeros((n_vertices, embed_dim), dtype=np.float32)
        vertex_hc_norm = normalize_vertex_positions(vertex_hc)
        with torch.no_grad():
            vv = torch.from_numpy(vertex_hc_norm.astype(np.float32)).to(device)
            z = vertex_encoder(vv)
            if vertex_mu is not None:
                z = vertex_mu(z)
            vertex_embed[:] = z.detach().cpu().numpy()
    else:
        vertex_embed = None

    # ---- Node feature assembly ----
    if face_embed is not None:
        face_x = np.concatenate([face_embed.astype(np.float32) * embed_scale, face_hc.astype(np.float32)], axis=1)
    else:
        face_x = face_hc.astype(np.float32)

    if edge_embed is not None:
        edge_x = np.concatenate([edge_embed.astype(np.float32) * embed_scale, edge_hc.astype(np.float32), edge_topo], axis=1)
    else:
        edge_x = np.concatenate([edge_hc.astype(np.float32), edge_topo], axis=1)

    if vertex_embed is not None:
        vertex_x = np.concatenate([vertex_embed.astype(np.float32) * embed_scale, vertex_hc.astype(np.float32)], axis=1)
    else:
        vertex_x = vertex_hc.astype(np.float32)

    data = HeteroData()
    data['face'].x = torch.from_numpy(face_x)
    data['edge'].x = torch.from_numpy(edge_x)
    data['vertex'].x = torch.from_numpy(vertex_x)

    # *** CLASSIFICATION: graph-level label ***
    if class_label is not None:
        data.y = torch.LongTensor([class_label])
    if class_name is not None:
        data.class_name = class_name

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
        data['face', 'uses_fwd', 'edge'].edge_index = torch.tensor([fwd_src, fwd_dst], dtype=torch.long)
        data['edge', 'used_by_fwd', 'face'].edge_index = torch.tensor([fwd_dst, fwd_src], dtype=torch.long)

    if len(rev_src) > 0:
        data['face', 'uses_rev', 'edge'].edge_index = torch.tensor([rev_src, rev_dst], dtype=torch.long)
        data['edge', 'used_by_rev', 'face'].edge_index = torch.tensor([rev_dst, rev_src], dtype=torch.long)

    # ---- Loop relations: edge -> next_edge within boundary loop ----
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

    if len(loop_src) > 0:
        data['edge', 'next_in_loop', 'edge'].edge_index = torch.tensor([loop_src, loop_dst], dtype=torch.long)

    # ---- Relations: edge -> vertex (canonical endpoints) ----
    e2v_src, e2v_dst, e2v_attr = [], [], []
    for e_idx in range(n_edges):
        vs = int(v_start_idx[e_idx]) if e_idx < len(v_start_idx) else -1
        ve = int(v_end_idx[e_idx]) if e_idx < len(v_end_idx) else -1

        if 0 <= vs < n_vertices:
            e2v_src.append(e_idx); e2v_dst.append(vs); e2v_attr.append([1.0])
        if 0 <= ve < n_vertices:
            e2v_src.append(e_idx); e2v_dst.append(ve); e2v_attr.append([0.0])

    if len(e2v_src) > 0:
        data['edge', 'has', 'vertex'].edge_index = torch.tensor([e2v_src, e2v_dst], dtype=torch.long)
        data['edge', 'has', 'vertex'].edge_attr = torch.tensor(e2v_attr, dtype=torch.float32)
        data['vertex', 'belongs_to', 'edge'].edge_index = torch.tensor([e2v_dst, e2v_src], dtype=torch.long)

    # ---- Face-face adjacency from handcrafted edge_to_faces ----
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

    if len(ff_src) > 0:
        data['face', 'adjacent_to', 'face'].edge_index = torch.tensor([ff_src, ff_dst], dtype=torch.long)

    data.model_name = model_name
    data.n_faces = n_faces
    data.n_edges = n_edges
    data.n_vertices = n_vertices

    return data


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build heterogeneous graphs with EDGE nodes for MechCAD classification")
    parser.add_argument("--joint_nurbs_dir", type=str, required=True,
                        help="Directory with *_joint_nurbs.pkl files")
    parser.add_argument("--joint_embed_dir", type=str, default=None,
                        help="Directory with *_embeddings.pkl (from train_encoder.py)")
    parser.add_argument("--handcrafted_dir", type=str, required=True,
                        help="Directory with *_handcrafted.pkl files")
    parser.add_argument("--encoder_ckpt", type=str, default=None,
                        help="Path to hetero_vae_best.pt (from train_encoder.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for *_graph.pt files")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_embeddings", action="store_true",
                        help="Build graphs with only handcrafted features (no VAE embeddings)")
    parser.add_argument("--embed_scale", type=float, default=0.25,
                        help="Scale factor for embeddings")
    args = parser.parse_args()

    if not args.no_embeddings:
        if args.joint_embed_dir is None:
            parser.error("--joint_embed_dir is required unless --no_embeddings is set")
        if args.encoder_ckpt is None:
            parser.error("--encoder_ckpt is required unless --no_embeddings is set")

    joint_nurbs_dir = Path(args.joint_nurbs_dir)
    handcrafted_dir = Path(args.handcrafted_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_embeddings = not args.no_embeddings
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load class map
    class_map_path = joint_nurbs_dir / 'class_map.json'
    if class_map_path.exists():
        with open(class_map_path) as f:
            class_map = json.load(f)
    else:
        stats_path = joint_nurbs_dir / 'stats.json'
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            class_map = stats.get('class_map', {})
        else:
            class_map = {}

    num_classes = len(class_map)
    class_names = sorted(class_map.keys(), key=lambda x: class_map[x])

    print(f"\n{'='*70}")
    print("BUILDING HETEROGENEOUS GRAPHS WITH EDGE NODES (MechCAD Classification)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Task: Classification ({num_classes} classes)")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Node types: face, edge, vertex")

    if use_embeddings:
        print(f"Mode: WITH VAE embeddings (scale={args.embed_scale})")
        joint_embed_dir = Path(args.joint_embed_dir)
        embed_files = sorted(joint_embed_dir.glob("*_embeddings.pkl"))
        print(f"Embedding files: {len(embed_files)}")
    else:
        print("Mode: HANDCRAFTED ONLY (no VAE embeddings)")
        embed_files = None

    # Determine vertex_dim
    vertex_dim = 8
    hc_files = list(handcrafted_dir.glob("*_handcrafted.pkl"))
    if hc_files:
        with open(hc_files[0], 'rb') as f:
            tmp = pickle.load(f)
        vf = tmp.get('vertex_features', None)
        if vf is not None and len(vf) > 0:
            vertex_dim = int(np.asarray(vf).shape[1])
    print(f"Vertex feature dim: {vertex_dim}")

    # Load encoders
    embed_dim = 64
    edge_enc, vertex_enc = None, None
    edge_mu, vertex_mu = None, None
    if use_embeddings:
        embed_dim_guess = 64
        if embed_files:
            fe, _, _ = load_face_embeddings(embed_files[0])
            embed_dim_guess = int(fe.shape[1])

        print(f"\nLoading encoders from: {args.encoder_ckpt}")
        embed_dim, edge_enc, vertex_enc, edge_mu, vertex_mu = load_encoders(
            Path(args.encoder_ckpt), device, vertex_dim, embed_dim_guess
        )

    # Determine iteration source
    if use_embeddings:
        iter_files = embed_files
    else:
        iter_files = sorted(joint_nurbs_dir.glob("*_joint_nurbs.pkl"))
        print(f"Joint NURBS files: {len(iter_files)}")

    print(f"\n{'='*70}")
    print("BUILDING GRAPHS")
    print(f"{'='*70}")

    success = 0
    failed = 0
    class_dist = defaultdict(int)

    for f in tqdm(iter_files, desc="Building graphs"):
        if use_embeddings:
            face_embed_raw, model_name, face_embed_indices = load_face_embeddings(f)
        else:
            model_name = f.stem.replace('_joint_nurbs', '')
            face_embed_raw = None
            face_embed_indices = None

        joint_path = joint_nurbs_dir / f"{model_name}_joint_nurbs.pkl"
        if not joint_path.exists():
            failed += 1
            continue

        # Load class label from joint NURBS pkl
        with open(joint_path, 'rb') as fp:
            joint_data = pickle.load(fp)
        class_label = joint_data.get('class_label', None)
        class_name_model = joint_data.get('class_name', None)

        handcrafted = load_handcrafted_features(handcrafted_dir, model_name)
        if handcrafted is None:
            if failed == 0:
                print(f"\n  [WARN] Missing handcrafted file for '{model_name}' in {handcrafted_dir}")
            failed += 1
            continue

        try:
            g = build_graph_for_model(
                model_name=model_name,
                joint_nurbs_path=joint_path,
                face_embed_raw=face_embed_raw,
                face_embed_indices=face_embed_indices,
                handcrafted=handcrafted,
                device=device,
                embed_dim=embed_dim,
                edge_encoder=edge_enc,
                vertex_encoder=vertex_enc,
                edge_mu=edge_mu,
                vertex_mu=vertex_mu,
                class_label=class_label,
                class_name=class_name_model,
                use_embeddings=use_embeddings,
                embed_scale=args.embed_scale,
            )
            if g is None:
                failed += 1
                continue

            torch.save(g, output_dir / f"{model_name}_graph.pt")
            success += 1
            if class_name_model:
                class_dist[class_name_model] += 1
        except Exception as e:
            print(f"\n  Error on {model_name}: {e}")
            failed += 1

    # Summary
    print(f"\n{'='*70}")
    print("EDGE GRAPH BUILDING COMPLETE (MechCAD Classification)")
    print(f"{'='*70}")
    mode_str = "WITH embeddings" if use_embeddings else "HANDCRAFTED ONLY"
    print(f"Mode: {mode_str}")
    print(f"Success: {success}")
    print(f"Failed: {failed}")
    print(f"\nClass distribution in graphs:")
    for name in class_names:
        print(f"  {name}: {class_dist.get(name, 0)}")
    print(f"Output: {output_dir}")

    # Print feature dimensions from a sample graph
    sample_graphs = list(output_dir.glob("*_graph.pt"))
    if sample_graphs:
        g = torch.load(sample_graphs[0], weights_only=False)
        print(f"\nNode feature dimensions (from {sample_graphs[0].stem}):")
        print(f"  Face:   {g['face'].x.shape[1]}")
        print(f"  Edge:   {g['edge'].x.shape[1]}")
        print(f"  Vertex: {g['vertex'].x.shape[1]}")
        if hasattr(g, 'y'):
            print(f"  Graph label: {g.y.item()} ({g.class_name if hasattr(g, 'class_name') else '?'})")

        rel_info = []
        for edge_type in g.edge_types:
            ei = g[edge_type].edge_index
            rel_info.append(f"  {edge_type}: {ei.shape[1]} edges")
        print(f"\nRelation types ({len(g.edge_types)}):")
        for r in rel_info:
            print(r)

    # Save class map in output dir for training script
    with open(output_dir / 'class_map.json', 'w') as f:
        json.dump(class_map, f, indent=2)

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
