"""
Extract Handcrafted Features for MechCAD (Classification Dataset)
==================================================================

Adapted from Fusion 360 handcrafted extraction.
Scans class subfolders: /mechcad/<class_name>/<file>.step

Changes from Fusion 360 version:
- Discovers classes from folder structure
- Stores class_label and class_name per model
- No .seg files needed

Face features (17 dims), Edge features (8 dims), Vertex features (8 dims)
— Same dimensions as Fusion 360 version.

Usage:
    python extract_handcrafted_mechcad.py \
        --data_dir /path/to/mechcad \
        --output_dir /path/to/mechcad/handcrafted
"""

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pickle
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# OCC imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE
from OCC.Core.TopoDS import topods_Face, topods_Edge, topods_Vertex, topods
from OCC.Core.TopTools import TopTools_ShapeMapHasher
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BSplineSurface,
    GeomAbs_BezierSurface, GeomAbs_Circle,
)
from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf

from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepBndLib import brepbndlib

try:
    from OCC.Core.GeomLProp import GeomLProp_SLProps
    HAS_GEOM_LPROP = True
except ImportError:
    HAS_GEOM_LPROP = False
    print("Warning: GeomLProp not available, curvature features will be estimated")

# BRepNet utilities
import utils.scale_utils as scale_utils


# =============================================================================
# CONFIGURATION
# =============================================================================

FACE_DIM = 17
EDGE_DIM = 8
VERTEX_DIM = 8

SURFACE_TYPE_MAP = {
    GeomAbs_Plane: 0,
    GeomAbs_Cylinder: 1,
    GeomAbs_Cone: 2,
    GeomAbs_Sphere: 3,
    GeomAbs_Torus: 4,
    GeomAbs_BSplineSurface: 5,
    GeomAbs_BezierSurface: 5,
}

HASH_BUCKETS = 65536


# =============================================================================
# CLASS DISCOVERY
# =============================================================================

def discover_classes(data_dir: Path) -> Dict[str, int]:
    class_dirs = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    return {name: idx for idx, name in enumerate(class_dirs)}


def make_unique_name(class_idx: int, stem: str) -> str:
    """Create a unique model name: classIdx__stem (class-blind)."""
    return f"{class_idx}__{stem}"


def discover_step_files(data_dir: Path, class_map: Dict[str, int]) -> List[Tuple[Path, int, str]]:
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
# DEDUPLICATION
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

    return unique


# =============================================================================
# GLOBAL GEOMETRY HELPERS
# =============================================================================

def compute_global_bbox(shape) -> Tuple[np.ndarray, float]:
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    arr = np.array([xmin, ymin, zmin, xmax, ymax, zmax], dtype=np.float32)
    diag = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)
    return arr, max(diag, 1e-6)


def compute_center_of_gravity(shape) -> np.ndarray:
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    mass = props.Mass()

    if abs(mass) < 1e-10:
        props = GProp_GProps()
        brepgprop.SurfaceProperties(shape, props)
        mass = props.Mass()

    if abs(mass) < 1e-10:
        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        return np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2], dtype=np.float32)

    cog = props.CentreOfMass()
    return np.array([cog.X(), cog.Y(), cog.Z()], dtype=np.float32)


# =============================================================================
# CURVATURE HELPERS
# =============================================================================

def compute_shape_index(k1: float, k2: float) -> float:
    if abs(k1 - k2) < 1e-10:
        return 0.0
    return (2.0 / np.pi) * np.arctan((k1 + k2) / (k1 - k2))


def compute_shape_index_distribution(adaptor: BRepAdaptor_Surface, n_samples: int = 16) -> np.ndarray:
    default = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

    if not HAS_GEOM_LPROP:
        return default

    try:
        u_min, u_max = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v_min, v_max = adaptor.FirstVParameter(), adaptor.LastVParameter()

        n = int(np.sqrt(n_samples))
        count = 0
        bins = np.zeros(4, dtype=np.float32)

        for i in range(n):
            for j in range(n):
                u = u_min + (u_max - u_min) * (i + 0.5) / n
                v = v_min + (v_max - v_min) * (j + 0.5) / n

                try:
                    props = GeomLProp_SLProps(adaptor.Surface().Surface(), u, v, 2, 1e-6)
                    if props.IsCurvatureDefined():
                        k1, k2 = props.MaxCurvature(), props.MinCurvature()
                        si = compute_shape_index(k1, k2)

                        if si < -0.5:
                            bins[0] += 1
                        elif si < 0:
                            bins[1] += 1
                        elif si < 0.5:
                            bins[2] += 1
                        else:
                            bins[3] += 1
                        count += 1
                except:
                    pass

        if count > 0:
            bins /= count
        else:
            bins = default
    except:
        bins = default

    return bins


def compute_curvature_at_center(adaptor: BRepAdaptor_Surface) -> Tuple[float, float, float, float]:
    if not HAS_GEOM_LPROP:
        return 0.0, 0.0, 0.0, 0.0

    try:
        u = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
        v = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2

        props = GeomLProp_SLProps(adaptor.Surface().Surface(), u, v, 2, 1e-6)
        if props.IsCurvatureDefined():
            k1, k2 = props.MaxCurvature(), props.MinCurvature()
            return (k1 + k2) / 2, k1 * k2, k1, k2
    except:
        pass
    return 0.0, 0.0, 0.0, 0.0


# =============================================================================
# DIHEDRAL ANGLE
# =============================================================================

def compute_dihedral_angle(face1, face2, shared_edge=None) -> Tuple[float, float]:
    try:
        adaptor1 = BRepAdaptor_Surface(face1)
        adaptor2 = BRepAdaptor_Surface(face2)

        edge_point = None
        edge_tangent = None
        if shared_edge is not None:
            try:
                edge_adaptor = BRepAdaptor_Curve(shared_edge)
                t_mid = (edge_adaptor.FirstParameter() + edge_adaptor.LastParameter()) / 2
                edge_point = edge_adaptor.Value(t_mid)
                edge_tangent = gp_Vec()
                edge_adaptor.D1(t_mid, gp_Pnt(), edge_tangent)
                if edge_tangent.Magnitude() > 1e-10:
                    edge_tangent.Normalize()
                else:
                    edge_tangent = None
            except:
                pass

        def get_normal_at_point(adaptor, face, point):
            surf = adaptor.Surface().Surface()

            if point is not None:
                try:
                    proj = GeomAPI_ProjectPointOnSurf(point, surf)
                    if proj.NbPoints() > 0:
                        u, v = proj.LowerDistanceParameters()
                    else:
                        u = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
                        v = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2
                except:
                    u = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
                    v = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2
            else:
                u = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
                v = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2

            if HAS_GEOM_LPROP:
                try:
                    props = GeomLProp_SLProps(surf, u, v, 1, 1e-6)
                    if props.IsNormalDefined():
                        n = props.Normal()
                        if face.Orientation() == 1:
                            return np.array([-n.X(), -n.Y(), -n.Z()])
                        return np.array([n.X(), n.Y(), n.Z()])
                except:
                    pass
            return None

        n1 = get_normal_at_point(adaptor1, face1, edge_point)
        n2 = get_normal_at_point(adaptor2, face2, edge_point)

        if n1 is not None and n2 is not None:
            dot = np.clip(np.dot(n1, n2), -1, 1)
            angle = np.arccos(dot) / np.pi

            if edge_tangent is not None:
                try:
                    cross = np.cross(n1, n2)
                    edge_t = np.array([edge_tangent.X(), edge_tangent.Y(), edge_tangent.Z()])
                    sign = np.dot(cross, edge_t)
                    if abs(sign) > 1e-6:
                        convexity = 1.0 if sign > 0 else -1.0
                    else:
                        convexity = 0.0
                except:
                    convexity = 1.0 if angle < 0.4 else (-1.0 if angle > 0.6 else 0.0)
            else:
                convexity = 1.0 if angle < 0.4 else (-1.0 if angle > 0.6 else 0.0)

            return angle, convexity
    except:
        pass
    return 0.5, 0.0


# =============================================================================
# SIMPLE HELPERS
# =============================================================================

def get_surface_type_onehot(face) -> np.ndarray:
    onehot = np.zeros(7, dtype=np.float32)
    try:
        adaptor = BRepAdaptor_Surface(face)
        surf_type = adaptor.GetType()
        idx = SURFACE_TYPE_MAP.get(surf_type, 6)
        onehot[idx] = 1.0
    except:
        onehot[6] = 1.0
    return onehot


def get_face_area(face) -> float:
    try:
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        area = props.Mass()
        return area if np.isfinite(area) else 0.0
    except:
        return 0.0


def get_edge_length(edge) -> float:
    try:
        props = GProp_GProps()
        brepgprop.LinearProperties(edge, props)
        length = props.Mass()
        return length if np.isfinite(length) else 0.0
    except:
        return 0.0


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def process_step_file(step_path: Path) -> Dict:
    """Extract handcrafted features from STEP file."""

    reader = STEPControl_Reader()
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise ValueError(f"Failed to read: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    shape = scale_utils.scale_solid_to_unit_box(shape)

    bbox, bbox_diag = compute_global_bbox(shape)
    cog = compute_center_of_gravity(shape)
    bbox_min, bbox_max = bbox[:3], bbox[3:]
    bbox_size = bbox_max - bbox_min

    # Collect entities
    faces, edges, vertices = [], [], []

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        faces.append(topods_Face(explorer.Current()))
        explorer.Next()

    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edges.append(topods_Edge(explorer.Current()))
        explorer.Next()
    edges = deduplicate_shapes(edges)

    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while explorer.More():
        vertices.append(topods_Vertex(explorer.Current()))
        explorer.Next()
    vertices = deduplicate_shapes(vertices)

    n_faces = len(faces)
    n_edges = len(edges)
    n_vertices = len(vertices)

    # Hash-based adjacency
    edge_hash_buckets = defaultdict(list)
    for ei, edge in enumerate(edges):
        h = TopTools_ShapeMapHasher.HashCode(edge, HASH_BUCKETS)
        edge_hash_buckets[h].append((ei, edge))

    vertex_hash_buckets = defaultdict(list)
    for vi, vert in enumerate(vertices):
        h = TopTools_ShapeMapHasher.HashCode(vert, HASH_BUCKETS)
        vertex_hash_buckets[h].append((vi, vert))

    edge_to_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        exp = TopExp_Explorer(face, TopAbs_EDGE)
        while exp.More():
            e = exp.Current()
            h = TopTools_ShapeMapHasher.HashCode(e, HASH_BUCKETS)
            for ei, edge in edge_hash_buckets[h]:
                if TopTools_ShapeMapHasher.IsEqual(e, edge):
                    if fi not in edge_to_faces[ei]:
                        edge_to_faces[ei].append(fi)
                    break
            exp.Next()

    vertex_to_edges = defaultdict(list)
    for ei, edge in enumerate(edges):
        exp = TopExp_Explorer(edge, TopAbs_VERTEX)
        while exp.More():
            v = exp.Current()
            h = TopTools_ShapeMapHasher.HashCode(v, HASH_BUCKETS)
            for vi, vert in vertex_hash_buckets[h]:
                if TopTools_ShapeMapHasher.IsEqual(v, vert):
                    if ei not in vertex_to_edges[vi]:
                        vertex_to_edges[vi].append(ei)
                    break
            exp.Next()

    edge_lengths = [get_edge_length(edge) for edge in edges]
    avg_edge_len = np.mean(edge_lengths) if edge_lengths else 1.0
    avg_edge_len = max(avg_edge_len, 1e-6)

    # FACE FEATURES (17 dims)
    face_features = []

    for fi, face in enumerate(faces):
        feat = np.zeros(FACE_DIM, dtype=np.float32)
        adaptor = None

        try:
            adaptor = BRepAdaptor_Surface(face)
        except:
            pass

        feat[0:7] = get_surface_type_onehot(face)

        try:
            area = get_face_area(face)
            feat[7] = np.log1p(area) / np.log1p(bbox_diag**2 + 1e-10)
        except:
            pass

        try:
            fb = Bnd_Box()
            brepbndlib.Add(face, fb)
            fx1, fy1, fz1, fx2, fy2, fz2 = fb.Get()
            extents = sorted([fx2 - fx1, fy2 - fy1, fz2 - fz1], reverse=True)
            ar = extents[0] / (extents[1] + 1e-10)
            feat[8] = min(ar, 10.0) / 10.0
        except:
            pass

        try:
            if adaptor is not None:
                feat[9:13] = compute_shape_index_distribution(adaptor)
            else:
                feat[9:13] = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        except:
            feat[9:13] = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

        try:
            if adaptor is not None:
                mean_c, gauss_c, k1, k2 = compute_curvature_at_center(adaptor)
                feat[13] = np.tanh(mean_c * bbox_diag)
                feat[14] = np.tanh(gauss_c * bbox_diag**2)
                feat[15] = np.tanh(k1 / (abs(k2) + 1e-10)) if abs(k2) > 1e-10 else 0.0
        except:
            pass

        try:
            n_loops = 0
            wexp = TopExp_Explorer(face, TopAbs_WIRE)
            while wexp.More():
                n_loops += 1
                wexp.Next()
            feat[16] = min(n_loops, 5) / 5.0
        except:
            pass

        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
        face_features.append(feat)

    # EDGE FEATURES (8 dims)
    edge_features = []
    edge_dihedrals = []

    for ei, edge in enumerate(edges):
        feat = np.zeros(EDGE_DIM, dtype=np.float32)

        try:
            length = edge_lengths[ei]
            feat[0] = np.log1p(length) / np.log1p(bbox_diag + 1e-10)
            feat[1] = min(length / (avg_edge_len + 1e-10), 5.0) / 5.0

            adj_faces = edge_to_faces[ei]
            if len(adj_faces) >= 2:
                dihedral, convexity = compute_dihedral_angle(
                    faces[adj_faces[0]], faces[adj_faces[1]], shared_edge=edge
                )
                is_boundary = 0.0
            else:
                dihedral, convexity = 0.5, 0.0
                is_boundary = 1.0

            feat[2] = dihedral
            feat[3] = np.sin(dihedral * np.pi)
            feat[4] = np.cos(dihedral * np.pi)
            feat[5] = convexity

            try:
                adaptor = BRepAdaptor_Curve(edge)
                if adaptor.GetType() == GeomAbs_Circle:
                    radius = adaptor.Circle().Radius()
                    feat[6] = np.tanh(bbox_diag / (radius + 1e-10))
            except:
                pass

            feat[7] = is_boundary

            edge_dihedrals.append(dihedral)

        except:
            edge_dihedrals.append(0.5)

        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
        edge_features.append(feat)

    # VERTEX FEATURES (8 dims)
    vertex_features = []

    for vi, vertex in enumerate(vertices):
        feat = np.zeros(VERTEX_DIM, dtype=np.float32)

        try:
            pnt = BRep_Tool.Pnt(vertex)
            pos = np.array([pnt.X(), pnt.Y(), pnt.Z()])
            feat[0:3] = (pos - bbox_min) / (bbox_size + 1e-10)

            dist = np.linalg.norm(pos - cog)
            feat[3] = dist / (bbox_diag + 1e-10)

            incident = vertex_to_edges[vi]
            feat[4] = min(len(incident), 10) / 10.0

            inc_lens = [edge_lengths[e] for e in incident if e < len(edge_lengths)]
            if inc_lens:
                avg_inc = np.mean(inc_lens)
                feat[5] = np.log1p(avg_inc) / np.log1p(bbox_diag + 1e-10)

            is_boundary_vertex = any(
                len(edge_to_faces[e]) == 1 for e in incident if e in edge_to_faces
            )
            feat[6] = 1.0 if is_boundary_vertex else 0.0

            inc_dihedrals = [edge_dihedrals[e] for e in incident if e < len(edge_dihedrals)]
            feat[7] = np.mean(inc_dihedrals) if inc_dihedrals else 0.5

        except:
            pass

        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
        vertex_features.append(feat)

    return {
        'face_features': np.array(face_features, dtype=np.float32),
        'edge_features': np.array(edge_features, dtype=np.float32),
        'vertex_features': np.array(vertex_features, dtype=np.float32),
        'edge_to_faces': dict(edge_to_faces),
        'vertex_to_edges': dict(vertex_to_edges),
        'n_faces': n_faces,
        'n_edges': n_edges,
        'n_vertices': n_vertices,
        'bbox': bbox,
        'cog': cog,
    }


def process_dataset(data_dir: Path, output_dir: Path):
    """Process entire MechCAD dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    class_map = discover_classes(data_dir)
    step_files = discover_step_files(data_dir, class_map)

    print(f"\n{'='*70}")
    print("HANDCRAFTED FEATURE EXTRACTION (MechCAD Classification)")
    print(f"{'='*70}")
    print(f"Files: {len(step_files)}")
    print(f"Classes ({len(class_map)}):")
    for name, idx in class_map.items():
        count = sum(1 for f, _, n in step_files if n == name)
        print(f"  [{idx}] {name}: {count} files")
    print(f"Face: {FACE_DIM} dims, Edge: {EDGE_DIM} dims, Vertex: {VERTEX_DIM} dims")
    print(f"GeomLProp available: {HAS_GEOM_LPROP}")
    print(f"{'='*70}\n")

    stats = {'files': 0, 'faces': 0, 'edges': 0, 'vertices': 0, 'failed': 0,
             'class_dist': defaultdict(int)}

    for step_file, class_label, class_name in tqdm(step_files, desc="Extracting"):
        try:
            result = process_step_file(step_file)

            # Add classification metadata
            unique_name = make_unique_name(class_label, step_file.stem)
            result['model_name'] = unique_name
            result['class_label'] = class_label
            result['class_name'] = class_name

            with open(output_dir / f"{unique_name}_handcrafted.pkl", 'wb') as f:
                pickle.dump(result, f)

            stats['files'] += 1
            stats['faces'] += result['n_faces']
            stats['edges'] += result['n_edges']
            stats['vertices'] += result['n_vertices']
            stats['class_dist'][class_name] += 1

        except Exception as e:
            stats['failed'] += 1
            # print(f"\nError {step_file.name}: {e}")

    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Success: {stats['files']}")
    print(f"Failed: {stats['failed']}")
    print(f"\nClass distribution:")
    for name, count in sorted(stats['class_dist'].items()):
        print(f"  {name}: {count}")
    print(f"Total faces: {stats['faces']}")
    print(f"Total edges: {stats['edges']}")
    print(f"Total vertices: {stats['vertices']}")
    print(f"{'='*70}\n")

    stats_out = dict(stats)
    stats_out['class_dist'] = dict(stats['class_dist'])
    stats_out['class_map'] = class_map
    stats_out['config'] = {
        'FACE_DIM': FACE_DIM,
        'EDGE_DIM': EDGE_DIM,
        'VERTEX_DIM': VERTEX_DIM,
        'task': 'classification',
        'method': 'unified adapted for MechCAD',
        'geom_lprop': HAS_GEOM_LPROP,
    }
    with open(output_dir / "stats.json", 'w') as f:
        json.dump(stats_out, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory with class subfolders (e.g. /mechcad)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for handcrafted pkl files")
    args = parser.parse_args()

    process_dataset(Path(args.data_dir), Path(args.output_dir))

# Usage:
# python extract_handcrafted_mechcad.py --data_dir /home/konstantinos/Downloads/mechcad/data --output_dir /home/konstantinos/Downloads/mechcad/output/handcrafted
