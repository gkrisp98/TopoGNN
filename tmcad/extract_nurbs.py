"""
MechCAD NURBS Extraction (adapted from Fusion 360 V2.1)
========================================================

Adapted for classification dataset: /mechcad/<class_name>/<file>.step
10 classes, label is per-MODEL (not per-face).

Changes from Fusion 360 version:
- Scans class subfolders to infer model-level labels
- No .seg files needed (label comes from folder name)
- NURBS surface grid: 10x10 (same as original)
- Stores 'class_label' and 'class_name' per model

Output per face (same as original):
- surf_pw, surf_u, surf_v, surf_mask, has_nurbs
- uv_samples, uv_mask
- curves_pw, curves_u, curves_mask, curves_ctrl_mask
- curve_edge_idx, curve_start_vidx, curve_end_vidx
- curve_connectivity, curve_orientation

Model-level:
- edge_data, n_faces, n_edges, n_vertices
- class_label (int), class_name (str)

Usage:
    python extract_mechcad_nurbs.py \
        --data_dir /path/to/mechcad \
        --output_dir /path/to/mechcad/nurbs
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# OCC imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopAbs import (
    TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE, TopAbs_VERTEX,
    TopAbs_REVERSED,
)
from OCC.Core.TopoDS import topods, TopoDS_Face, TopoDS_Edge, TopoDS_Wire
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface,
)
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCC.Core.GeomConvert import geomconvert
from OCC.Core.Geom import Geom_TrimmedCurve
from OCC.Core.gp import gp_Pnt
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepTools import BRepTools_WireExplorer

# OCCWL - Same library BRepNet uses (for UV sampling)
from occwl.uvgrid import uvgrid as occwl_uvgrid
from occwl.face import Face as OccwlFace
from occwl.solid import Solid as OccwlSolid

# BRepNet utilities
import utils.scale_utils as scale_utils


# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_SURF_CTRL_U = 10
MAX_SURF_CTRL_V = 10
MAX_SURF_KNOTS_U = 25
MAX_SURF_KNOTS_V = 25

UV_SAMPLES_U = 16
UV_SAMPLES_V = 16
UV_CHANNELS = 7  # x, y, z, nx, ny, nz, mask (OCCWL uvgrid)

MAX_CURVES = 8
MAX_CURVE_CTRL = 20
MAX_CURVE_KNOTS = 25

SURFACE_TYPE_MAP = {
    GeomAbs_Plane: 0,
    GeomAbs_Cylinder: 1,
    GeomAbs_Cone: 2,
    GeomAbs_Sphere: 3,
    GeomAbs_Torus: 4,
    GeomAbs_BSplineSurface: 5,
    GeomAbs_BezierSurface: 6,
}


# =============================================================================
# MECHCAD CLASS DEFINITIONS
# =============================================================================

# Will be auto-discovered from folder names; this is the expected set
MECHCAD_CLASSES = None  # Set dynamically


def discover_classes(data_dir: Path) -> Dict[str, int]:
    """
    Discover class names from subdirectory names.
    Returns sorted dict: class_name -> class_index.
    """
    class_dirs = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    return {name: idx for idx, name in enumerate(class_dirs)}


def make_unique_name(class_idx: int, stem: str) -> str:
    """Create a unique model name: classIdx__stem (class-blind)."""
    return f"{class_idx}__{stem}"


def discover_step_files(data_dir: Path, class_map: Dict[str, int]) -> List[Tuple[Path, int, str]]:
    """
    Find all STEP files with their class labels.
    Returns list of (step_path, class_label, class_name).
    """
    files = []
    for class_name, class_idx in class_map.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        for ext in ['*.stp', '*.step', '*.STEP', '*.STP']:
            for f in class_dir.glob(ext):
                files.append((f, class_idx, class_name))
    return files


# =============================================================================
# STEP + BBOX
# =============================================================================

def load_step_file(step_path: Path):
    """Load STEP file."""
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        return None
    reader.TransferRoots()
    return reader.OneShape()


def compute_bounding_box(shape) -> Tuple[np.ndarray, float]:
    """Compute bounding box center and diagonal size."""
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    min_pt = np.array([xmin, ymin, zmin])
    max_pt = np.array([xmax, ymax, zmax])
    size = np.linalg.norm(max_pt - min_pt)
    return min_pt, max(size, 1e-6)


# =============================================================================
# GLOBAL ENTITY LISTS (TopExp_Explorer ordering + deduplication)
# =============================================================================

def deduplicate_shapes(shapes, upper=1000003):
    unique = []
    seen_hashes = {}

    for s in shapes:
        try:
            h = s.HashCode(upper)
        except Exception:
            h = None

        is_dup = False
        for idx in seen_hashes.get(h, []):
            if s.IsSame(unique[idx]):
                is_dup = True
                break

        if not is_dup:
            seen_hashes.setdefault(h, []).append(len(unique))
            unique.append(s)

    return unique, len(shapes) - len(unique)


def list_entities(shape):
    faces_raw, edges_raw, vertices_raw = [], [], []

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces_raw.append(topods.Face(exp.Current()))
        exp.Next()

    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edges_raw.append(topods.Edge(exp.Current()))
        exp.Next()

    exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    while exp.More():
        vertices_raw.append(topods.Vertex(exp.Current()))
        exp.Next()

    faces, _ = deduplicate_shapes(faces_raw)
    edges, _ = deduplicate_shapes(edges_raw)
    vertices, _ = deduplicate_shapes(vertices_raw)

    return faces, edges, vertices


def build_hash_index(shapes, upper=1000003):
    hmap = {}
    for i, s in enumerate(shapes):
        try:
            h = s.HashCode(upper)
        except Exception:
            h = None
        hmap.setdefault(h, []).append(i)
    return hmap


def find_index_is_same(target, shapes, hash_index=None, upper=1000003) -> int:
    if target is None:
        return -1

    if hash_index is not None:
        try:
            h = target.HashCode(upper)
        except Exception:
            h = None
        for i in hash_index.get(h, []):
            if target.IsSame(shapes[i]):
                return i

    for i, s in enumerate(shapes):
        if target.IsSame(s):
            return i
    return -1


# =============================================================================
# BOUNDARY LOOP EXTRACTION
# =============================================================================

def get_boundary_loops(face: TopoDS_Face) -> List[List[Tuple[TopoDS_Edge, bool]]]:
    loops = []
    wire_exp = TopExp_Explorer(face, TopAbs_WIRE)

    while wire_exp.More():
        wire = topods.Wire(wire_exp.Current())
        loop_edges = []

        try:
            wire_explorer = BRepTools_WireExplorer(wire, face)
            while wire_explorer.More():
                e = topods.Edge(wire_explorer.Current())
                is_forward = e.Orientation() != TopAbs_REVERSED
                loop_edges.append((e, is_forward))
                wire_explorer.Next()
        except Exception:
            edge_exp = TopExp_Explorer(wire, TopAbs_EDGE)
            while edge_exp.More():
                e = topods.Edge(edge_exp.Current())
                is_forward = e.Orientation() != TopAbs_REVERSED
                loop_edges.append((e, is_forward))
                edge_exp.Next()

        if loop_edges:
            loops.append(loop_edges)

        wire_exp.Next()

    return loops


def flatten_boundary_loops(loops):
    edges = []
    orientations = []
    connectivity = np.full((MAX_CURVES, 2), -1, dtype=np.int32)

    global_idx = 0
    for loop in loops:
        loop_start = global_idx
        for i, (edge, is_forward) in enumerate(loop):
            if global_idx >= MAX_CURVES:
                break
            edges.append(edge)
            orientations.append(is_forward)

            if i == 0:
                prev_idx = loop_start + (len(loop) - 1)
                if prev_idx >= MAX_CURVES:
                    prev_idx = -1
            else:
                prev_idx = global_idx - 1

            if i == len(loop) - 1:
                next_idx = loop_start
            else:
                next_idx = global_idx + 1
                if next_idx >= MAX_CURVES:
                    next_idx = -1

            connectivity[global_idx] = [prev_idx, next_idx]
            global_idx += 1

        if global_idx >= MAX_CURVES:
            break

    return edges, orientations, connectivity


# =============================================================================
# UV SAMPLING (using OCCWL uvgrid)
# =============================================================================

def sample_uvgrid_occwl(face: OccwlFace, num_u: int = UV_SAMPLES_U, num_v: int = UV_SAMPLES_V) -> Dict:
    try:
        points = occwl_uvgrid(face, num_u, num_v, method="point")
        normals = occwl_uvgrid(face, num_u, num_v, method="normal")
        mask = occwl_uvgrid(face, num_u, num_v, method="inside")

        uv_samples = np.concatenate([points, normals, mask], axis=2)
        uv_samples = np.nan_to_num(uv_samples, nan=0.0, posinf=1.0, neginf=-1.0)
        uv_samples = uv_samples.astype(np.float32)
        uv_mask = mask.squeeze(-1).astype(np.float32)

        return {'uv_samples': uv_samples, 'uv_mask': uv_mask}
    except Exception:
        return {
            'uv_samples': np.zeros((num_u, num_v, UV_CHANNELS), dtype=np.float32),
            'uv_mask': np.zeros((num_u, num_v), dtype=np.float32),
        }


# =============================================================================
# NURBS SURFACE EXTRACTION
# =============================================================================

def extract_nurbs_control_points(face: TopoDS_Face, bbox_size: float) -> Optional[Dict]:
    try:
        nurbs_face = topods.Face(BRepBuilderAPI_NurbsConvert(face).Shape())
        surface = BRep_Tool.Surface(nurbs_face)
        bspline = geomconvert.SurfaceToBSplineSurface(surface)
        if bspline is None:
            return None

        n_u = bspline.NbUPoles()
        n_v = bspline.NbVPoles()
        if n_u > MAX_SURF_CTRL_U or n_v > MAX_SURF_CTRL_V:
            return None

        poles = np.zeros((n_u, n_v, 3), dtype=np.float32)
        weights = np.zeros((n_u, n_v, 1), dtype=np.float32)
        for i in range(n_u):
            for j in range(n_v):
                p = bspline.Pole(i + 1, j + 1)
                poles[i, j] = [p.X(), p.Y(), p.Z()]
                weights[i, j, 0] = bspline.Weight(i + 1, j + 1)

        poles_norm = np.clip(poles / bbox_size, -10, 10)
        ctrl_pts = np.concatenate([poles_norm, weights], axis=-1)

        surf_pw = np.zeros((MAX_SURF_CTRL_U, MAX_SURF_CTRL_V, 4), dtype=np.float32)
        surf_pw[:n_u, :n_v] = ctrl_pts

        surf_mask = np.zeros((MAX_SURF_CTRL_U, MAX_SURF_CTRL_V), dtype=np.float32)
        surf_mask[:n_u, :n_v] = 1.0

        # Knots
        u_knots = []
        for i in range(1, bspline.NbUKnots() + 1):
            u_knots.extend([bspline.UKnot(i)] * bspline.UMultiplicity(i))
        u_knots = np.array(u_knots)

        v_knots = []
        for i in range(1, bspline.NbVKnots() + 1):
            v_knots.extend([bspline.VKnot(i)] * bspline.VMultiplicity(i))
        v_knots = np.array(v_knots)

        if len(u_knots) > MAX_SURF_KNOTS_U or len(v_knots) > MAX_SURF_KNOTS_V:
            return None

        def normalize_knots(k):
            if len(k) == 0:
                return np.zeros(1)
            r = k.max() - k.min()
            if r > 1e-10:
                return (k - k.min()) / r
            return np.linspace(0, 1, len(k))

        surf_u = np.zeros(MAX_SURF_KNOTS_U, dtype=np.float32)
        surf_v = np.zeros(MAX_SURF_KNOTS_V, dtype=np.float32)
        surf_u[:len(u_knots)] = normalize_knots(u_knots)
        surf_v[:len(v_knots)] = normalize_knots(v_knots)

        return {'surf_pw': surf_pw, 'surf_u': surf_u, 'surf_v': surf_v, 'surf_mask': surf_mask}
    except Exception:
        return None


# =============================================================================
# CURVE NURBS EXTRACTION
# =============================================================================

def get_edge_vertices_points(edge: TopoDS_Edge) -> Tuple[Optional[gp_Pnt], Optional[gp_Pnt]]:
    try:
        v_start = topexp.FirstVertex(edge)
        v_end = topexp.LastVertex(edge)
        return BRep_Tool.Pnt(v_start), BRep_Tool.Pnt(v_end)
    except Exception:
        return None, None


_curve_reject_stats = {"no_curve": 0, "too_many_poles": 0, "too_many_knots": 0, "exception": 0, "ok": 0}

def get_curve_reject_stats():
    return dict(_curve_reject_stats)

def extract_curve_nurbs(edge: TopoDS_Edge, bbox_size: float) -> Optional[Dict]:
    try:
        curve, u1, u2 = BRep_Tool.Curve(edge)
        if curve is None:
            _curve_reject_stats["no_curve"] += 1
            return None

        params_reversed = (u2 < u1)
        if params_reversed:
            u1, u2 = u2, u1

        trimmed_curve = Geom_TrimmedCurve(curve, u1, u2)
        bspline = geomconvert.CurveToBSplineCurve(trimmed_curve)

        if params_reversed:
            bspline.Reverse()

        n_poles = bspline.NbPoles()
        if n_poles > MAX_CURVE_CTRL:
            _curve_reject_stats["too_many_poles"] += 1
            return None

        poles = np.zeros((n_poles, 3), dtype=np.float32)
        weights = np.zeros((n_poles, 1), dtype=np.float32)

        for i in range(n_poles):
            p = bspline.Pole(i + 1)
            poles[i] = [p.X(), p.Y(), p.Z()]
            weights[i, 0] = bspline.Weight(i + 1)

        poles_norm = np.clip(poles / bbox_size, -10, 10)
        ctrl_pts = np.concatenate([poles_norm, weights], axis=-1)

        curve_pw = np.zeros((MAX_CURVE_CTRL, 4), dtype=np.float32)
        curve_pw[:n_poles] = ctrl_pts

        curve_mask = np.zeros(MAX_CURVE_CTRL, dtype=np.float32)
        curve_mask[:n_poles] = 1.0

        knots = []
        for i in range(1, bspline.NbKnots() + 1):
            knots.extend([bspline.Knot(i)] * bspline.Multiplicity(i))
        knots = np.array(knots)

        if len(knots) > MAX_CURVE_KNOTS:
            _curve_reject_stats["too_many_knots"] += 1
            return None

        r = knots.max() - knots.min()
        if r > 1e-10:
            knots = (knots - knots.min()) / r
        else:
            knots = np.linspace(0, 1, len(knots))

        curve_u = np.zeros(MAX_CURVE_KNOTS, dtype=np.float32)
        curve_u[:len(knots)] = knots

        p_start, p_end = get_edge_vertices_points(edge)
        start_pt = np.array([p_start.X(), p_start.Y(), p_start.Z()], dtype=np.float32) / bbox_size if p_start else np.zeros(3, dtype=np.float32)
        end_pt = np.array([p_end.X(), p_end.Y(), p_end.Z()], dtype=np.float32) / bbox_size if p_end else np.zeros(3, dtype=np.float32)

        _curve_reject_stats["ok"] += 1
        return {
            'curve_pw': curve_pw,
            'curve_u': curve_u,
            'curve_mask': curve_mask,
            'start_pt': start_pt,
            'end_pt': end_pt,
        }
    except Exception:
        _curve_reject_stats["exception"] += 1
        return None


# =============================================================================
# FACE EXTRACTION
# =============================================================================

def extract_face_data(
    face: TopoDS_Face,
    occwl_face: OccwlFace,
    bbox_size: float,
    face_idx: int,
    global_edges: List[TopoDS_Edge],
    global_vertices: List,
    edge_hash_index: Dict,
    vertex_hash_index: Dict,
    upper_hash: int = 1000003,
) -> Dict:

    uv_data = sample_uvgrid_occwl(occwl_face)

    nurbs_data = extract_nurbs_control_points(face, bbox_size)
    if nurbs_data is not None:
        has_nurbs = True
        surf_pw = nurbs_data['surf_pw']
        surf_u = nurbs_data['surf_u']
        surf_v = nurbs_data['surf_v']
        surf_mask = nurbs_data['surf_mask']
    else:
        has_nurbs = False
        surf_pw = np.zeros((MAX_SURF_CTRL_U, MAX_SURF_CTRL_V, 4), dtype=np.float32)
        surf_u = np.zeros(MAX_SURF_KNOTS_U, dtype=np.float32)
        surf_v = np.zeros(MAX_SURF_KNOTS_V, dtype=np.float32)
        surf_mask = np.zeros((MAX_SURF_CTRL_U, MAX_SURF_CTRL_V), dtype=np.float32)

    boundary_loops = get_boundary_loops(face)
    edges_in_loops, orientations, connectivity = flatten_boundary_loops(boundary_loops)

    curves_pw = np.zeros((MAX_CURVES, MAX_CURVE_CTRL, 4), dtype=np.float32)
    curves_u = np.zeros((MAX_CURVES, MAX_CURVE_KNOTS), dtype=np.float32)
    curves_mask = np.zeros(MAX_CURVES, dtype=np.float32)
    curves_ctrl_mask = np.zeros((MAX_CURVES, MAX_CURVE_CTRL), dtype=np.float32)
    curve_orientation = np.zeros(MAX_CURVES, dtype=np.float32)
    curve_start_pts = np.zeros((MAX_CURVES, 3), dtype=np.float32)
    curve_end_pts = np.zeros((MAX_CURVES, 3), dtype=np.float32)

    curve_edge_idx = np.full((MAX_CURVES,), -1, dtype=np.int32)
    curve_start_vidx = np.full((MAX_CURVES,), -1, dtype=np.int32)
    curve_end_vidx = np.full((MAX_CURVES,), -1, dtype=np.int32)

    num_curves = 0
    for i, (edge, is_forward) in enumerate(zip(edges_in_loops, orientations)):
        if i >= MAX_CURVES:
            break

        e_idx = find_index_is_same(edge, global_edges, edge_hash_index, upper_hash)
        curve_edge_idx[i] = e_idx

        try:
            v1 = topexp.FirstVertex(edge)
            v2 = topexp.LastVertex(edge)
            if not is_forward:
                v1, v2 = v2, v1
            v_start, v_end = v1, v2
        except Exception:
            v_start, v_end = None, None

        curve_start_vidx[i] = find_index_is_same(v_start, global_vertices, vertex_hash_index, upper_hash)
        curve_end_vidx[i] = find_index_is_same(v_end, global_vertices, vertex_hash_index, upper_hash)

        curve_data = extract_curve_nurbs(edge, bbox_size)
        if curve_data is not None:
            curves_pw[i] = curve_data['curve_pw']
            curves_u[i] = curve_data['curve_u']
            curves_mask[i] = 1.0
            curves_ctrl_mask[i] = curve_data['curve_mask']
            curve_orientation[i] = 1.0 if is_forward else 0.0
            curve_start_pts[i] = curve_data['start_pt']
            curve_end_pts[i] = curve_data['end_pt']
            num_curves += 1
        else:
            curves_mask[i] = 0.0

    adaptor = BRepAdaptor_Surface(face)
    surface_type = SURFACE_TYPE_MAP.get(adaptor.GetType(), 5)

    return {
        'face_idx': face_idx,
        'surf_pw': surf_pw,
        'surf_u': surf_u,
        'surf_v': surf_v,
        'surf_mask': surf_mask,
        'has_nurbs': has_nurbs,
        'uv_samples': uv_data['uv_samples'],
        'uv_mask': uv_data['uv_mask'],
        'curves_pw': curves_pw,
        'curves_u': curves_u,
        'curves_mask': curves_mask,
        'curves_ctrl_mask': curves_ctrl_mask,
        'num_curves': num_curves,
        'curve_connectivity': connectivity,
        'curve_orientation': curve_orientation,
        'curve_start_pts': curve_start_pts,
        'curve_end_pts': curve_end_pts,
        'num_boundary_loops': len(boundary_loops),
        'curve_edge_idx': curve_edge_idx,
        'curve_start_vidx': curve_start_vidx,
        'curve_end_vidx': curve_end_vidx,
        'surface_type': surface_type,
    }


# =============================================================================
# MODEL PROCESSING
# =============================================================================

def process_model(step_path: Path, class_label: int, class_name: str) -> Optional[Dict]:
    """Process a single model. Label is per-model (classification)."""
    shape = load_step_file(step_path)
    if shape is None:
        return None

    # Unique name: class__stem
    unique_name = make_unique_name(class_label, step_path.stem)

    # Scale to unit box
    shape = scale_utils.scale_solid_to_unit_box(shape)

    _, bbox_size = compute_bounding_box(shape)

    faces, edges, vertices = list_entities(shape)
    edge_hash = build_hash_index(edges)
    vertex_hash = build_hash_index(vertices)

    n_faces = len(faces)
    if n_faces == 0:
        return None

    # Create OCCWL faces for UV sampling
    try:
        solid = OccwlSolid(shape)
        occwl_faces = list(solid.faces())
        if len(occwl_faces) != n_faces:
            occwl_faces = [OccwlFace(f) for f in faces]
    except:
        occwl_faces = [OccwlFace(f) for f in faces]

    # Extract face data (no per-face labels for classification)
    face_data_list = []
    n_with_nurbs = 0

    for face_idx, face in enumerate(faces):
        fd = extract_face_data(
            face, occwl_faces[face_idx], bbox_size, face_idx,
            global_edges=edges,
            global_vertices=vertices,
            edge_hash_index=edge_hash,
            vertex_hash_index=vertex_hash,
        )
        # No per-face label — classification is per-model
        fd['label'] = -1
        if fd['has_nurbs']:
            n_with_nurbs += 1
        face_data_list.append(fd)

    # Model-level edge data
    edge_data_list = []
    for e_idx, edge in enumerate(edges):
        try:
            v_start = topexp.FirstVertex(edge)
            v_end = topexp.LastVertex(edge)
        except Exception:
            v_start, v_end = None, None

        v_start_idx = find_index_is_same(v_start, vertices, vertex_hash)
        v_end_idx = find_index_is_same(v_end, vertices, vertex_hash)

        cdata = extract_curve_nurbs(edge, bbox_size)
        if cdata is None:
            cdata = {
                'curve_pw': np.zeros((MAX_CURVE_CTRL, 4), dtype=np.float32),
                'curve_u': np.zeros((MAX_CURVE_KNOTS,), dtype=np.float32),
                'curve_mask': np.zeros((MAX_CURVE_CTRL,), dtype=np.float32),
                'start_pt': np.zeros((3,), dtype=np.float32),
                'end_pt': np.zeros((3,), dtype=np.float32),
            }
            has_curve = 0.0
        else:
            has_curve = 1.0

        edge_data_list.append({
            'edge_idx': int(e_idx),
            'curve_pw': cdata['curve_pw'],
            'curve_u': cdata['curve_u'],
            'curve_mask': cdata['curve_mask'],
            'v_start_idx': int(v_start_idx),
            'v_end_idx': int(v_end_idx),
            'has_curve': float(has_curve),
        })

    total_loops = sum(fd['num_boundary_loops'] for fd in face_data_list)

    return {
        'model_name': unique_name,
        'face_data': face_data_list,
        'edge_data': edge_data_list,
        'n_faces': n_faces,
        'n_edges': len(edges),
        'n_vertices': len(vertices),
        'n_with_nurbs': n_with_nurbs,
        'total_boundary_loops': total_loops,
        'bbox_size': bbox_size,
        # Classification-specific fields
        'class_label': class_label,
        'class_name': class_name,
        'has_labels': True,  # Always true for classification
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory with class subfolders (e.g. /mechcad)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for pkl files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover classes from folder structure
    class_map = discover_classes(data_dir)
    num_classes = len(class_map)

    # Find all STEP files
    step_files = discover_step_files(data_dir, class_map)

    print(f"\n{'='*70}")
    print("MECHCAD NURBS EXTRACTION (Classification Dataset)")
    print(f"{'='*70}")
    print(f"Data dir: {data_dir}")
    print(f"Classes ({num_classes}):")
    for name, idx in class_map.items():
        count = sum(1 for f, _, n in step_files if n == name)
        print(f"  [{idx}] {name}: {count} files")
    print(f"Total STEP files: {len(step_files)}")
    print(f"UV samples: {UV_SAMPLES_U}x{UV_SAMPLES_V}")
    print(f"Surface NURBS: {MAX_SURF_CTRL_U}x{MAX_SURF_CTRL_V}")
    print(f"Curve NURBS: {MAX_CURVE_CTRL} ctrl pts, {MAX_CURVES} curves/face")
    print(f"{'='*70}\n")

    stats = {
        'success': 0,
        'failed': 0,
        'total_faces': 0,
        'faces_with_nurbs': 0,
        'total_edges': 0,
        'total_vertices': 0,
        'total_curves': 0,
        'class_dist': defaultdict(int),
    }

    import sys, atexit

    def _print_curve_stats():
        """Print curve extraction stats - called on exit or crash."""
        cstats = get_curve_reject_stats()
        total_attempts = sum(cstats.values())
        if total_attempts > 0:
            print(f"\nCurve extraction breakdown ({total_attempts} total):")
            print(f"  OK:              {cstats['ok']:>8} ({100*cstats['ok']/total_attempts:.1f}%)")
            print(f"  No curve geom:   {cstats['no_curve']:>8} ({100*cstats['no_curve']/total_attempts:.1f}%)")
            print(f"  Too many poles:  {cstats['too_many_poles']:>8} ({100*cstats['too_many_poles']/total_attempts:.1f}%)  (limit={MAX_CURVE_CTRL})")
            print(f"  Too many knots:  {cstats['too_many_knots']:>8} ({100*cstats['too_many_knots']/total_attempts:.1f}%)  (limit={MAX_CURVE_KNOTS})")
            print(f"  Exception:       {cstats['exception']:>8} ({100*cstats['exception']/total_attempts:.1f}%)")
            sys.stdout.flush()

    atexit.register(_print_curve_stats)

    for step_file, class_label, class_name in tqdm(step_files, desc="Extracting"):
        try:
            result = process_model(step_file, class_label, class_name)

            if result is not None:
                out_file = output_dir / f"{result['model_name']}_joint_nurbs.pkl"
                with open(out_file, 'wb') as f:
                    pickle.dump(result, f)

                stats['success'] += 1
                stats['total_faces'] += result['n_faces']
                stats['faces_with_nurbs'] += result['n_with_nurbs']
                stats['total_edges'] += result['n_edges']
                stats['total_vertices'] += result['n_vertices']
                stats['class_dist'][class_name] += 1

                for fd in result['face_data']:
                    stats['total_curves'] += fd['num_curves']
            else:
                stats['failed'] += 1

        except Exception as e:
            stats['failed'] += 1
            print(f"\nError {step_file.name}: {e}")

    # Save stats + class mapping
    stats_out = dict(stats)
    stats_out['class_dist'] = dict(stats['class_dist'])
    stats_out['class_map'] = class_map
    stats_out['num_classes'] = num_classes
    stats_out['config'] = {
        'UV_SAMPLES': (UV_SAMPLES_U, UV_SAMPLES_V),
        'UV_CHANNELS': UV_CHANNELS,
        'SURF_NURBS_CTRL': (MAX_SURF_CTRL_U, MAX_SURF_CTRL_V),
        'CURVE_NURBS_CTRL': MAX_CURVE_CTRL,
        'MAX_CURVES_PER_FACE': MAX_CURVES,
        'task': 'classification',
        'method': 'unified V2.1 adapted for MechCAD',
    }
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats_out, f, indent=2)

    # Also save class_map separately for downstream scripts
    with open(output_dir / 'class_map.json', 'w') as f:
        json.dump(class_map, f, indent=2)

    pct_nurbs = 100 * stats['faces_with_nurbs'] / max(stats['total_faces'], 1)

    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Models:")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    print(f"\nClass distribution:")
    for name, count in sorted(stats['class_dist'].items()):
        print(f"  {name}: {count}")
    print(f"\nTopology:")
    print(f"  Total faces: {stats['total_faces']}")
    print(f"  With NURBS: {stats['faces_with_nurbs']} ({pct_nurbs:.1f}%)")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Total vertices: {stats['total_vertices']}")
    print(f"  Total curves: {stats['total_curves']}")

    cstats = get_curve_reject_stats()
    total_attempts = sum(cstats.values())
    if total_attempts > 0:
        print(f"\nCurve extraction breakdown ({total_attempts} total):")
        print(f"  OK:              {cstats['ok']:>8} ({100*cstats['ok']/total_attempts:.1f}%)")
        print(f"  No curve geom:   {cstats['no_curve']:>8} ({100*cstats['no_curve']/total_attempts:.1f}%)")
        print(f"  Too many poles:  {cstats['too_many_poles']:>8} ({100*cstats['too_many_poles']/total_attempts:.1f}%)  (limit={MAX_CURVE_CTRL})")
        print(f"  Too many knots:  {cstats['too_many_knots']:>8} ({100*cstats['too_many_knots']/total_attempts:.1f}%)  (limit={MAX_CURVE_KNOTS})")
        print(f"  Exception:       {cstats['exception']:>8} ({100*cstats['exception']/total_attempts:.1f}%)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

# Usage:
#python extract_mechcad_nurbs.py --data_dir /home/konstantinos/Downloads/mechcad/data --output_dir /home/konstantinos/Downloads/mechcad/output/nurbs