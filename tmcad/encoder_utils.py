"""
Shared utilities for separate VAE training scripts.

Contains constants, normalization functions, data loading, and helper functions
used by train_face_vae.py, train_edge_vae.py, and train_vertex_vae.py.
"""

import os
import random
import json
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn


# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_SURF_CTRL_U = 10
MAX_SURF_CTRL_V = 10
MAX_SURF_KNOTS = 25

UV_SAMPLES_U = 16
UV_SAMPLES_V = 16
UV_CHANNELS = 7

MAX_CURVES = 8
MAX_CURVE_CTRL = 20
MAX_CURVE_KNOTS = 25

EMBED_DIM = 64


# =============================================================================
# HELPERS
# =============================================================================

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split_file(split_path: Path) -> dict:
    """Load split file - supports multiple key conventions."""
    with open(split_path, "r") as f:
        split = json.load(f)

    def _get_set(keys):
        for k in keys:
            if k in split and split[k] is not None:
                return set(split[k])
        return set()

    train_set = _get_set(["train", "training_set", "train_set", "training"])
    val_set = _get_set(["val", "valid", "validation", "validation_set", "val_set"])
    test_set = _get_set(["test", "test_set", "testing_set", "testing"])

    all_set = train_set | val_set | test_set

    return {
        "train": train_set,
        "val": val_set,
        "test": test_set,
        "all": all_set,
    }


def masked_mse(pred, target, mask, n_channels: int = 1, eps: float = 1e-8):
    """
    pred, target: same shape
    mask: broadcastable to pred/target (typically ... x 1)
    n_channels: multiply denom by channels if mask doesn't include channel dim
    """
    diff = (pred - target) ** 2
    diff = diff * mask
    denom = mask.sum().clamp(min=1.0) * n_channels
    return diff.sum() / (denom + eps)


# =============================================================================
# COORDINATE NORMALIZATION
# =============================================================================

def normalize_coordinates_face(surf_pw, surf_mask, uv_samples, uv_mask):
    """
    Normalize coordinates by subtracting centroid to prevent position shortcuts.

    Computes centroid from surface control points and UV samples ONLY (not curves).
    """
    bs = surf_pw.shape[0]

    surf_xyz = surf_pw[..., :3]
    surf_mask_exp = surf_mask.unsqueeze(-1)

    uv_xyz = uv_samples[..., :3]
    uv_mask_exp = uv_mask.unsqueeze(-1)

    sum_coords = (
        (surf_xyz * surf_mask_exp).sum(dim=(1, 2)) +
        (uv_xyz * uv_mask_exp).sum(dim=(1, 2))
    )
    count = (
        surf_mask_exp.sum(dim=(1, 2, 3)) +
        uv_mask_exp.sum(dim=(1, 2, 3))
    ).clamp(min=1.0)

    centroid = sum_coords / count.unsqueeze(-1)

    centroid_surf = centroid.view(bs, 1, 1, 3)
    centroid_uv = centroid.view(bs, 1, 1, 3)

    surf_pw_norm = surf_pw.clone()
    surf_pw_norm[..., :3] = surf_pw[..., :3] - centroid_surf

    uv_samples_norm = uv_samples.clone()
    uv_samples_norm[..., :3] = uv_samples[..., :3] - centroid_uv

    return surf_pw_norm, uv_samples_norm, centroid


def normalize_coordinates_edge(curve_pw, curve_mask):
    """Normalize edge coordinates by subtracting centroid."""
    xyz = curve_pw[..., :3]
    mask_exp = curve_mask.unsqueeze(-1)

    sum_coords = (xyz * mask_exp).sum(dim=1)
    count = mask_exp.sum(dim=(1, 2)).clamp(min=1.0)

    centroid = sum_coords / count.unsqueeze(-1)

    curve_pw_norm = curve_pw.clone()
    curve_pw_norm[..., :3] = curve_pw[..., :3] - centroid.unsqueeze(1)

    return curve_pw_norm, centroid


def normalize_coordinates_curves(curves_pw, curves_mask, curves_ctrl_mask, face_centroid=None):
    """
    Normalize boundary curve coordinates using FACE centroid (not per-curve).

    Using the face centroid preserves relative arrangement while removing absolute position.
    """
    bs = curves_pw.shape[0]
    xyz = curves_pw[..., :3]
    point_mask = (curves_mask.unsqueeze(-1) * curves_ctrl_mask).unsqueeze(-1)

    if face_centroid is None:
        sum_coords = (xyz * point_mask).sum(dim=(1, 2))
        count = point_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
        face_centroid = sum_coords / count.unsqueeze(-1)

    curves_pw_norm = curves_pw.clone()
    curves_pw_norm[..., :3] = curves_pw[..., :3] - face_centroid.view(bs, 1, 1, 3)

    return curves_pw_norm


def normalize_vertex_positions(vertex_features: np.ndarray) -> np.ndarray:
    """
    Normalize vertex xyz positions per-model to prevent Fourier aliasing.
    """
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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_face_data(joint_nurbs_dir: Path, allowed_models: set = None,
                    normalize_uv_bbox: bool = False) -> list:
    all_faces = []
    files = sorted(joint_nurbs_dir.glob("*_joint_nurbs.pkl"))

    for f in tqdm(files, desc="Loading faces"):
        model_name = f.stem.replace("_joint_nurbs", "")
        if allowed_models is not None and model_name not in allowed_models:
            continue

        with open(f, "rb") as fp:
            data = pickle.load(fp)

        bbox_size = data.get("bbox_size", 1.0) if normalize_uv_bbox else None

        for face_data in data["face_data"]:
            uv = face_data.get("uv_samples", np.zeros((UV_SAMPLES_U, UV_SAMPLES_V, UV_CHANNELS)))
            if normalize_uv_bbox and bbox_size > 1e-6:
                uv = uv.copy()
                uv[..., :3] = np.clip(uv[..., :3] / bbox_size, -10, 10)

            face = {
                "surf_pw": face_data.get("surf_pw", np.zeros((MAX_SURF_CTRL_U, MAX_SURF_CTRL_V, 4))),
                "surf_u": face_data.get("surf_u", np.zeros(MAX_SURF_KNOTS)),
                "surf_v": face_data.get("surf_v", np.zeros(MAX_SURF_KNOTS)),
                "surf_mask": face_data.get("surf_mask", np.zeros((MAX_SURF_CTRL_U, MAX_SURF_CTRL_V))),
                "uv_samples": uv,
                "uv_mask": face_data.get("uv_mask", np.zeros((UV_SAMPLES_U, UV_SAMPLES_V))),
                "curves_pw": face_data.get("curves_pw", np.zeros((MAX_CURVES, MAX_CURVE_CTRL, 4))),
                "curves_u": face_data.get("curves_u", np.zeros((MAX_CURVES, MAX_CURVE_KNOTS))),
                "curves_mask": face_data.get("curves_mask", np.zeros(MAX_CURVES)),
                "curves_ctrl_mask": face_data.get("curves_ctrl_mask", np.zeros((MAX_CURVES, MAX_CURVE_CTRL))),
                "curve_orientation": face_data.get("curve_orientation", np.zeros(MAX_CURVES)),
            }
            all_faces.append(face)

    return all_faces


def load_edge_data(joint_nurbs_dir: Path, allowed_models: set = None) -> list:
    """
    Load UNIQUE edges from edge_data (not per-face boundary curves).
    """
    all_edges = []
    skipped_empty = 0
    skipped_nan = 0
    files = sorted(joint_nurbs_dir.glob("*_joint_nurbs.pkl"))

    def _sanitize(arr):
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    for f in tqdm(files, desc="Loading edges"):
        model_name = f.stem.replace("_joint_nurbs", "")
        if allowed_models is not None and model_name not in allowed_models:
            continue

        with open(f, "rb") as fp:
            data = pickle.load(fp)

        edge_data = data.get("edge_data", None)

        if edge_data is not None:
            for ed in edge_data:
                raw_mask = ed.get("curve_mask", np.zeros(MAX_CURVE_CTRL))
                raw_pw = ed.get("curve_pw", np.zeros((MAX_CURVE_CTRL, 4)))
                raw_u = ed.get("curve_u", np.zeros(MAX_CURVE_KNOTS))

                has_nan = (np.any(~np.isfinite(raw_mask)) or
                          np.any(~np.isfinite(raw_pw)) or
                          np.any(~np.isfinite(raw_u)))
                if has_nan:
                    skipped_nan += 1

                curve_mask = _sanitize(raw_mask)
                curve_pw = _sanitize(raw_pw)
                curve_u = _sanitize(raw_u)

                if np.sum(curve_mask) < 1e-6:
                    skipped_empty += 1
                    continue
                all_edges.append({
                    "curve_pw": curve_pw,
                    "curve_u": curve_u,
                    "curve_mask": curve_mask,
                })
        else:
            seen_edge_idx = set()
            for face_data_item in data["face_data"]:
                curve_edge_idx = face_data_item.get("curve_edge_idx", None)
                raw_curves_pw = face_data_item.get("curves_pw", np.zeros((MAX_CURVES, MAX_CURVE_CTRL, 4)))
                raw_curves_u = face_data_item.get("curves_u", np.zeros((MAX_CURVES, MAX_CURVE_KNOTS)))
                curves_mask = face_data_item.get("curves_mask", np.zeros(MAX_CURVES))
                raw_curves_ctrl_mask = face_data_item.get("curves_ctrl_mask", np.zeros((MAX_CURVES, MAX_CURVE_CTRL)))

                curves_pw = _sanitize(raw_curves_pw)
                curves_u = _sanitize(raw_curves_u)
                curves_ctrl_mask = _sanitize(raw_curves_ctrl_mask)

                for i in range(MAX_CURVES):
                    if curves_mask[i] > 0:
                        ctrl_mask = curves_ctrl_mask[i]
                        if np.sum(ctrl_mask) < 1e-6:
                            skipped_empty += 1
                            continue

                        if curve_edge_idx is not None and i < len(curve_edge_idx):
                            e_idx = int(curve_edge_idx[i])
                            if e_idx < 0:
                                continue
                            if e_idx in seen_edge_idx:
                                continue
                            seen_edge_idx.add(e_idx)

                        all_edges.append({
                            "curve_pw": curves_pw[i],
                            "curve_u": curves_u[i],
                            "curve_mask": ctrl_mask,
                        })

    if skipped_empty > 0:
        print(f"  [Note] Skipped {skipped_empty} edges with empty curve data")
    if skipped_nan > 0:
        print(f"  [Note] Sanitized {skipped_nan} edges with NaN/Inf values")

    return all_edges


def load_vertex_data(handcrafted_dir: Path, allowed_models: set = None) -> list:
    """Load vertex features with per-model xyz normalization."""
    all_vertices = []
    files = sorted(handcrafted_dir.glob("*_handcrafted.pkl"))

    for f in tqdm(files, desc="Loading vertices"):
        model_name = f.stem.replace("_handcrafted", "")
        if allowed_models is not None and model_name not in allowed_models:
            continue

        with open(f, "rb") as fp:
            data = pickle.load(fp)

        vertex_features = data.get("vertex_features", None)
        if vertex_features is not None and len(vertex_features) > 0:
            vertex_features = normalize_vertex_positions(np.array(vertex_features))
            for v_feat in vertex_features:
                all_vertices.append(v_feat)

    return all_vertices


# =============================================================================
# DATASET CLASSES
# =============================================================================

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
        self._clean_data()

    def _clean_data(self):
        for item in self.data:
            for k, v in item.items():
                if isinstance(v, np.ndarray):
                    item[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        out = {}
        for k, v in item.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.FloatTensor(v)
            else:
                out[k] = v
        return out


class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list
        self._clean_data()

    def _clean_data(self):
        for item in self.data:
            for k, v in item.items():
                if isinstance(v, np.ndarray):
                    item[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "curve_pw": torch.FloatTensor(item["curve_pw"]),
            "curve_u": torch.FloatTensor(item["curve_u"]),
            "curve_mask": torch.FloatTensor(item["curve_mask"]),
        }


class VertexDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx])


def face_collate_fn(batch):
    result = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        else:
            result[key] = [b[key] for b in batch]
    return result
