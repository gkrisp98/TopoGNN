"""
Fusion 360 NURBS Extraction (V2.1 - unified architecture)
=============================================================

Ported from the unified extraction pipeline for consistency across datasets.

Key improvements over V1:
- Global topology: curve_edge_idx, curve_start_vidx, curve_end_vidx per curve slot
- Model-level edge_data with canonical vertex indices
- BRepTools_WireExplorer for ordered boundary loops with correct orientation
- Proper curve trimming (Geom_TrimmedCurve) with reversed-parameter handling
- Entity deduplication (IsSame-based)
- Hash-based lookups (O(N) average)
- UV sampling with OCCWL uvgrid point/normal/inside queries

Kept from original Fusion pipeline:
- scale_utils.scale_solid_to_unit_box (BRepNet compatibility)
- .seg file label loading

Output per face:
- surf_pw: (10, 10, 4)        - NURBS surface control points + weights
- surf_u: (25,)               - U knot vector
- surf_v: (25,)               - V knot vector
- surf_mask: (10, 10)         - valid control points
- has_nurbs: bool
- uv_samples: (16, 16, 7)    - x,y,z, nx,ny,nz, mask (OCCWL uvgrid)
- uv_mask: (16, 16)           - OCCWL inside mask
- curves_pw: (8, 20, 4)       - boundary curve NURBS
- curves_u: (8, 25)           - curve knot vectors
- curves_mask: (8,)           - which curves exist
- curves_ctrl_mask: (8, 20)   - which control points per curve
- curve_edge_idx: (8,)        - global edge index
- curve_start_vidx: (8,)      - global start vertex index
- curve_end_vidx: (8,)        - global end vertex index
- curve_connectivity: (8, 2)  - prev/next within loop
- curve_orientation: (8,)     - forward/reversed in wire

Model-level:
- edge_data: list[dict]       - one per global edge
- n_faces, n_edges, n_vertices
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
# STEP + BBOX
# =============================================================================

def load_step_file(step_path: Path):
    """Load STEP file."""
    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        return None
    reader.TransferRoots()
    return reader.OneShape()


def load_seg_file(seg_path: Path) -> Optional[List[int]]:
    """Load segmentation labels."""
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
    """
    Remove duplicate shapes (by IsSame) preserving first-encounter order.
    TopExp_Explorer can return duplicates in some topologies.
    """
    unique = []
    seen_hashes = {}  # hash -> list of indices in unique

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
    """
    List all faces, edges, vertices using TopExp_Explorer.
    Deduplicates to ensure each entity appears exactly once.
    """
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
    """Build hash-based index for O(1) average lookup."""
    hmap = {}
    for i, s in enumerate(shapes):
        try:
            h = s.HashCode(upper)
        except Exception:
            h = None
        hmap.setdefault(h, []).append(i)
    return hmap


def find_index_is_same(target, shapes, hash_index=None, upper=1000003) -> int:
    """Find the index of target in shapes using IsSame, with hash acceleration."""
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

    # Fallback full scan
    for i, s in enumerate(shapes):
        if target.IsSame(s):
            return i
    return -1


# =============================================================================
# BOUNDARY LOOP EXTRACTION (ordered, using BRepTools_WireExplorer)
# =============================================================================

def get_boundary_loops(face: TopoDS_Face) -> List[List[Tuple[TopoDS_Edge, bool]]]:
    """
    Returns list of loops; each loop is list of (edge, is_forward_in_wire).
    Uses BRepTools_WireExplorer for proper ordering.
    """
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
            # Fallback to unordered exploration
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


def flatten_boundary_loops(
    loops: List[List[Tuple[TopoDS_Edge, bool]]]
) -> Tuple[List[TopoDS_Edge], List[bool], np.ndarray]:
    """Flatten loops into edge list with connectivity."""
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

            # prev/next within loop
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
# UV SAMPLING (using OCCWL uvgrid - same as BRepNet)
# =============================================================================

def sample_uvgrid_occwl(face: OccwlFace, num_u: int = UV_SAMPLES_U, num_v: int = UV_SAMPLES_V) -> Dict:
    """
    Sample UV grid using occwl - the same method BRepNet uses.
    Battle-tested on Fusion 360 and handles all edge cases properly.
    
    Returns:
        dict with 'uv_samples': (num_u, num_v, 7) - x,y,z, nx,ny,nz, mask
                    'uv_mask': (num_u, num_v)
    """
    try:
        # These are the exact same calls BRepNet makes!
        points = occwl_uvgrid(face, num_u, num_v, method="point")    # (num_u, num_v, 3)
        normals = occwl_uvgrid(face, num_u, num_v, method="normal")  # (num_u, num_v, 3)
        mask = occwl_uvgrid(face, num_u, num_v, method="inside")     # (num_u, num_v, 1)

        # Combine into single array
        uv_samples = np.concatenate([points, normals, mask], axis=2)  # (num_u, num_v, 7)

        # Clean any NaN/Inf
        uv_samples = np.nan_to_num(uv_samples, nan=0.0, posinf=1.0, neginf=-1.0)
        uv_samples = uv_samples.astype(np.float32)

        # Extract mask as separate 2D array
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
    """Extract NURBS control points from a face."""
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
# CURVE NURBS EXTRACTION (with proper trimming)
# =============================================================================

def get_edge_vertices_points(edge: TopoDS_Edge) -> Tuple[Optional[gp_Pnt], Optional[gp_Pnt]]:
    """Get start/end vertex points of an edge."""
    try:
        v_start = topexp.FirstVertex(edge)
        v_end = topexp.LastVertex(edge)
        return BRep_Tool.Pnt(v_start), BRep_Tool.Pnt(v_end)
    except Exception:
        return None, None


def extract_curve_nurbs(edge: TopoDS_Edge, bbox_size: float) -> Optional[Dict]:
    """
    Extract NURBS control points from an edge.
    Uses Geom_TrimmedCurve for proper trimming and handles reversed parameters.
    """
    try:
        curve, u1, u2 = BRep_Tool.Curve(edge)
        if curve is None:
            return None

        # Guard against reversed parameter order
        params_reversed = (u2 < u1)
        if params_reversed:
            u1, u2 = u2, u1

        # Trim curve to edge's parameter range before conversion
        trimmed_curve = Geom_TrimmedCurve(curve, u1, u2)
        bspline = geomconvert.CurveToBSplineCurve(trimmed_curve)

        # If we swapped params, reverse BSpline to restore edge direction
        if params_reversed:
            bspline.Reverse()

        n_poles = bspline.NbPoles()
        if n_poles > MAX_CURVE_CTRL:
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

        # Knots
        knots = []
        for i in range(1, bspline.NbKnots() + 1):
            knots.extend([bspline.Knot(i)] * bspline.Multiplicity(i))
        knots = np.array(knots)

        if len(knots) > MAX_CURVE_KNOTS:
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

        return {
            'curve_pw': curve_pw,
            'curve_u': curve_u,
            'curve_mask': curve_mask,
            'start_pt': start_pt,
            'end_pt': end_pt,
        }
    except Exception:
        return None


# =============================================================================
# FACE EXTRACTION (with global topology indices)
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
    """Extract all data for one face, including global topology indices."""

    # UV sampling using OCCWL (same as BRepNet)
    uv_data = sample_uvgrid_occwl(occwl_face)

    # Normalize UV sample positions by bbox diagonal (matching NURBS ctrl pts scale)
    uv_samples = uv_data['uv_samples']
    if bbox_size > 1e-6:
        uv_samples[..., :3] = np.clip(uv_samples[..., :3] / bbox_size, -10, 10)
    uv_data['uv_samples'] = uv_samples

    # NURBS surface
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

    # Boundary loops (ordered)
    boundary_loops = get_boundary_loops(face)
    edges_in_loops, orientations, connectivity = flatten_boundary_loops(boundary_loops)

    # Curve arrays
    curves_pw = np.zeros((MAX_CURVES, MAX_CURVE_CTRL, 4), dtype=np.float32)
    curves_u = np.zeros((MAX_CURVES, MAX_CURVE_KNOTS), dtype=np.float32)
    curves_mask = np.zeros(MAX_CURVES, dtype=np.float32)
    curves_ctrl_mask = np.zeros((MAX_CURVES, MAX_CURVE_CTRL), dtype=np.float32)
    curve_orientation = np.zeros(MAX_CURVES, dtype=np.float32)
    curve_start_pts = np.zeros((MAX_CURVES, 3), dtype=np.float32)
    curve_end_pts = np.zeros((MAX_CURVES, 3), dtype=np.float32)

    # Global topology indices per curve slot
    curve_edge_idx = np.full((MAX_CURVES,), -1, dtype=np.int32)
    curve_start_vidx = np.full((MAX_CURVES,), -1, dtype=np.int32)
    curve_end_vidx = np.full((MAX_CURVES,), -1, dtype=np.int32)

    num_curves = 0
    for i, (edge, is_forward) in enumerate(zip(edges_in_loops, orientations)):
        if i >= MAX_CURVES:
            break

        # Global edge index
        e_idx = find_index_is_same(edge, global_edges, edge_hash_index, upper_hash)
        curve_edge_idx[i] = e_idx

        # Oriented start/end vertex index
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

        # Extract curve NURBS
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

    # Surface type
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

        # Global topology indices
        'curve_edge_idx': curve_edge_idx,
        'curve_start_vidx': curve_start_vidx,
        'curve_end_vidx': curve_end_vidx,

        'surface_type': surface_type,
    }


# =============================================================================
# MODEL PROCESSING
# =============================================================================

def process_model(step_path: Path, seg_path: Optional[Path] = None) -> Optional[Dict]:
    """Process a single model."""
    shape = load_step_file(step_path)
    if shape is None:
        return None

    # Scale to unit box - SAME AS BREPNET
    shape = scale_utils.scale_solid_to_unit_box(shape)

    _, bbox_size = compute_bounding_box(shape)

    # Global entity lists (TopExp_Explorer ordering, deduplicated)
    faces, edges, vertices = list_entities(shape)
    edge_hash = build_hash_index(edges)
    vertex_hash = build_hash_index(vertices)

    n_faces = len(faces)
    if n_faces == 0:
        return None

    # Create OCCWL faces for UV sampling (from original scaled shape)
    try:
        solid = OccwlSolid(shape)
        occwl_faces = list(solid.faces())
        if len(occwl_faces) != n_faces:
            occwl_faces = [OccwlFace(f) for f in faces]
    except:
        occwl_faces = [OccwlFace(f) for f in faces]

    # Load labels
    labels = None
    if seg_path is not None:
        labels = load_seg_file(seg_path)
        if labels is not None and len(labels) != n_faces:
            print(f"Warning: {step_path.stem} has {n_faces} faces but {len(labels)} labels")
            labels = None

    # Extract face data
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
        fd['label'] = labels[face_idx] if (labels is not None and face_idx < len(labels)) else -1
        if fd['has_nurbs']:
            n_with_nurbs += 1
        face_data_list.append(fd)

    # Model-level edge data (canonical per global edge)
    edge_data_list = []
    for e_idx, edge in enumerate(edges):
        # Canonical endpoints
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
        'model_name': step_path.stem,
        'face_data': face_data_list,
        'edge_data': edge_data_list,
        'n_faces': n_faces,
        'n_edges': len(edges),
        'n_vertices': len(vertices),
        'n_with_nurbs': n_with_nurbs,
        'total_boundary_loops': total_loops,
        'bbox_size': bbox_size,
        'has_labels': labels is not None,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with STEP files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seg_dir", type=str, default=None, help="Directory with .seg files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = Path(args.seg_dir) if args.seg_dir else None

    # Find STEP files
    step_files = list(data_dir.rglob("*.stp")) + list(data_dir.rglob("*.step")) + \
                 list(data_dir.rglob("*.STEP")) + list(data_dir.rglob("*.STP"))

    print(f"\n{'='*70}")
    print("FUSION 360 NURBS EXTRACTION (V2.1 - unified architecture)")
    print(f"{'='*70}")
    print(f"Found {len(step_files)} STEP files")
    print(f"UV samples: {UV_SAMPLES_U}x{UV_SAMPLES_V} (OCCWL uvgrid - same as BRepNet)")
    print(f"Surface NURBS: {MAX_SURF_CTRL_U}x{MAX_SURF_CTRL_V}")
    print(f"Curve NURBS: {MAX_CURVE_CTRL} ctrl pts, {MAX_CURVES} curves/face")
    print(f"Topology: global edge/vertex indices, ordered loops, deduplication")
    print(f"Curve trimming: Geom_TrimmedCurve with reversed-param handling")
    print(f"{'='*70}\n")

    stats = {
        'success': 0,
        'failed': 0,
        'with_labels': 0,
        'total_faces': 0,
        'faces_with_nurbs': 0,
        'total_edges': 0,
        'total_vertices': 0,
        'total_curves': 0,
        'label_dist': defaultdict(int),
    }

    for step_file in tqdm(step_files, desc="Extracting"):
        # Find seg file
        seg_path = None
        if seg_dir:
            seg_path = seg_dir / f"{step_file.stem}.seg"
            if not seg_path.exists():
                seg_path = None

        try:
            result = process_model(step_file, seg_path)

            if result is not None:
                out_file = output_dir / f"{result['model_name']}_joint_nurbs.pkl"
                with open(out_file, 'wb') as f:
                    pickle.dump(result, f)

                stats['success'] += 1
                stats['total_faces'] += result['n_faces']
                stats['faces_with_nurbs'] += result['n_with_nurbs']
                stats['total_edges'] += result['n_edges']
                stats['total_vertices'] += result['n_vertices']

                for fd in result['face_data']:
                    stats['total_curves'] += fd['num_curves']

                if result['has_labels']:
                    stats['with_labels'] += 1
                    for fd in result['face_data']:
                        if fd['label'] >= 0:
                            stats['label_dist'][fd['label']] += 1
            else:
                stats['failed'] += 1

        except Exception as e:
            stats['failed'] += 1
            print(f"\nError {step_file.name}: {e}")

    # Save stats
    stats_out = dict(stats)
    stats_out['label_dist'] = dict(stats['label_dist'])
    stats_out['config'] = {
        'UV_SAMPLES': (UV_SAMPLES_U, UV_SAMPLES_V),
        'UV_CHANNELS': UV_CHANNELS,
        'SURF_NURBS_CTRL': (MAX_SURF_CTRL_U, MAX_SURF_CTRL_V),
        'CURVE_NURBS_CTRL': MAX_CURVE_CTRL,
        'MAX_CURVES_PER_FACE': MAX_CURVES,
        'method': 'unified V2.1 (global topology, ordered loops, trimmed curves)',
    }
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats_out, f, indent=2)

    pct_nurbs = 100 * stats['faces_with_nurbs'] / max(stats['total_faces'], 1)

    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Models:")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  With labels: {stats['with_labels']}")
    print(f"\nTopology:")
    print(f"  Total faces: {stats['total_faces']}")
    print(f"  With NURBS: {stats['faces_with_nurbs']} ({pct_nurbs:.1f}%)")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Total vertices: {stats['total_vertices']}")
    print(f"  Total curves: {stats['total_curves']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

# Usage:
#python extract_fusion360_nurbs_occwl.py --data_dir /home/konstantinos/Downloads/s2.0.0/breps/step --seg_dir /home/konstantinos/Downloads/s2.0.0/breps/seg --output_dir /home/konstantinos/Downloads/s2.0.0/nurbs
