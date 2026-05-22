"""
Face VAE Training (Separate)
=============================

Trains ONLY the face encoder/decoder VAE:
  - NURBSEncoder (transformer-based, NeuroNURBS-style)
  - UVEncoder (Conv2D)
  - BoundaryCurveEncoder (Conv1D + transformer aggregation)
  - FaceEncoder (fusion of above three)
  - NURBSDecoder (joint ctrl pts + knots), UVDecoder

After training, extracts per-model face embeddings.

Usage:
    python train_face_vae.py \\
        --joint_nurbs_dir /path/to/joint_nurbs \\
        --output_dir ./output_face_vae \\
        --split_file ./train_test.json \\
        --epochs 100 --batch_size 64 --device cuda
"""

import os
import math
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from encoder_utils import (
    MAX_SURF_CTRL_U, MAX_SURF_CTRL_V, MAX_SURF_KNOTS,
    UV_SAMPLES_U, UV_SAMPLES_V, UV_CHANNELS,
    MAX_CURVES, MAX_CURVE_CTRL, MAX_CURVE_KNOTS,
    EMBED_DIM,
    set_seed, load_split_file, masked_mse,
    normalize_coordinates_face, normalize_coordinates_curves,
    load_face_data, FaceDataset, face_collate_fn,
)


# =============================================================================
# MODELS
# =============================================================================

class NURBSEncoder(nn.Module):
    """
    NeuroNURBS-style transformer encoder for NURBS surface control points.

    Handles dynamic control point grid sizes: positional embeddings are created
    for a max size and sliced at runtime based on actual input shape.
    """
    def __init__(self, embed_dim: int = 64, max_tokens: int = 900):
        super().__init__()
        self.max_tokens = max_tokens

        self.ctrl_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.knot_embed = nn.Sequential(
            nn.Linear(MAX_SURF_KNOTS, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Positional embeddings for up to max_tokens (handles varying grid sizes)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_tokens, embed_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=8, norm=nn.LayerNorm(embed_dim)
        )

    def forward(self, surf_pw, surf_u, surf_v, surf_mask=None):
        bs = surf_pw.shape[0]

        # Flatten control point grid: (bs, U, V, 4) → (bs, U*V, 4)
        n_tokens = surf_pw.shape[1] * surf_pw.shape[2]
        x = surf_pw.reshape(bs, n_tokens, 4)

        # Embed control points
        ctrl_feat = self.ctrl_embed(x)

        # Embed knot vectors and broadcast
        u_feat = self.knot_embed(surf_u).unsqueeze(1).expand_as(ctrl_feat)
        v_feat = self.knot_embed(surf_v).unsqueeze(1).expand_as(ctrl_feat)

        # Slice positional embeddings to match actual token count
        pos_emb = self.pos_embedding[:, :n_tokens, :]

        tokens = ctrl_feat + pos_emb + u_feat + v_feat

        # Build padding mask from surf_mask: True = ignore
        padding_mask = None
        if surf_mask is not None:
            flat_mask = surf_mask.reshape(bs, -1)
            padding_mask = (flat_mask == 0)
            all_masked = padding_mask.all(dim=1)
            padding_mask[all_masked, 0] = False

        out = self.transformer(tokens, src_key_padding_mask=padding_mask)

        # Masked mean pooling over valid tokens
        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            out = (out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        else:
            out = out.mean(dim=1)

        return out


class UVEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(UV_CHANNELS, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.GELU(),
            nn.Linear(256, embed_dim), nn.LayerNorm(embed_dim),
        )

    def forward(self, uv_samples, uv_mask=None):
        x = uv_samples.permute(0, 3, 1, 2)
        if uv_mask is not None:
            x = x * uv_mask.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x.contiguous().view(x.size(0), -1))


class BoundaryCurveEncoder(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.curve_conv = nn.Sequential(
            nn.Conv1d(4, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.knot_embed = nn.Linear(MAX_CURVE_KNOTS, 32)
        self.curve_mlp = nn.Sequential(
            nn.Linear(64 * 4 + 32 + 1, 128), nn.GELU(),
            nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1, activation="gelu",
            norm_first=True, batch_first=True,
        )
        self.curve_transformer = nn.TransformerEncoder(
            enc_layer, num_layers=2, norm=nn.LayerNorm(embed_dim)
        )
        self.curve_pos_enc = nn.Parameter(torch.randn(1, MAX_CURVES, embed_dim) * 0.02)

    def forward(self, curves_pw, curves_u, curves_mask, curves_ctrl_mask, curve_orientation):
        bs = curves_pw.shape[0]
        n_curves = curves_pw.shape[1]
        ctrl_pts = curves_pw.shape[2]

        flat_pw = curves_pw.reshape(bs * n_curves, ctrl_pts, 4).permute(0, 2, 1)
        flat_ctrl_mask = curves_ctrl_mask.reshape(bs * n_curves, ctrl_pts)
        flat_pw = flat_pw * flat_ctrl_mask.unsqueeze(1)

        conv_out = self.curve_conv(flat_pw)
        conv_feat = conv_out.contiguous().view(bs * n_curves, -1)

        flat_u = curves_u.reshape(bs * n_curves, -1)
        knot_feat = self.knot_embed(flat_u)

        flat_orient = curve_orientation.reshape(bs * n_curves, 1)

        combined = torch.cat([conv_feat, knot_feat, flat_orient], dim=-1)
        curve_embeds = self.curve_mlp(combined)
        curve_embeds = curve_embeds.view(bs, n_curves, -1)

        curve_embeds = curve_embeds + self.curve_pos_enc[:, :n_curves]

        padding_mask = (curves_mask == 0)
        all_masked = padding_mask.all(dim=1)
        padding_mask[all_masked, 0] = False

        out = self.curve_transformer(curve_embeds, src_key_padding_mask=padding_mask)

        valid_mask = (~padding_mask).unsqueeze(-1).float()
        out = (out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)

        return out


class FaceEncoder(nn.Module):
    """Complete face encoder combining NURBS, UV, and boundary curves."""
    def __init__(self, embed_dim: int = 64,
                 disable_nurbs: bool = False, disable_uv: bool = False, disable_curves: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.disable_nurbs = disable_nurbs
        self.disable_uv = disable_uv
        self.disable_curves = disable_curves

        self.nurbs_encoder = NURBSEncoder(embed_dim)
        self.uv_encoder = UVEncoder(embed_dim)
        self.curve_encoder = BoundaryCurveEncoder(embed_dim)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, batch):
        surf_pw = batch["surf_pw"]
        surf_u = batch["surf_u"]
        surf_v = batch["surf_v"]
        surf_mask = batch["surf_mask"]
        uv_samples = batch["uv_samples"]
        uv_mask = batch["uv_mask"]
        curves_pw = batch["curves_pw"]
        curves_u = batch["curves_u"]
        curves_mask = batch["curves_mask"]
        curves_ctrl_mask = batch["curves_ctrl_mask"]
        curve_orientation = batch["curve_orientation"]

        # Normalize coordinates
        surf_pw_norm, uv_samples_norm, face_centroid = normalize_coordinates_face(
            surf_pw, surf_mask, uv_samples, uv_mask
        )
        curves_pw_norm = normalize_coordinates_curves(
            curves_pw, curves_mask, curves_ctrl_mask, face_centroid=face_centroid
        )

        bs = surf_pw.shape[0]
        device = surf_pw.device
        zeros = torch.zeros(bs, self.embed_dim, device=device)

        z_nurbs = self.nurbs_encoder(surf_pw_norm, surf_u, surf_v, surf_mask) if not self.disable_nurbs else zeros
        z_uv = self.uv_encoder(uv_samples_norm, uv_mask) if not self.disable_uv else zeros
        z_curve = self.curve_encoder(curves_pw_norm, curves_u, curves_mask, curves_ctrl_mask, curve_orientation) if not self.disable_curves else zeros

        combined = torch.cat([z_nurbs, z_uv, z_curve], dim=-1)
        return self.fusion(combined)


class NURBSDecoder(nn.Module):
    """
    MLP-based decoder that jointly reconstructs control points and knot vectors.
    Handles dynamic grid sizes by storing the grid shape at init.
    """
    def __init__(self, embed_dim: int = 64, n_ctrl_u: int = None, n_ctrl_v: int = None):
        super().__init__()
        # These will be set dynamically before first forward if not provided
        self.n_ctrl_u = n_ctrl_u
        self.n_ctrl_v = n_ctrl_v
        self.n_tokens = None  # Set dynamically
        self.hidden_dim = embed_dim
        self.embed_dim = embed_dim

        # Will be initialized in set_grid_size or lazily
        self.expand = None
        self.head_ctrl = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 4),
        )
        self.head_knot_u = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, MAX_SURF_KNOTS),
            nn.Sigmoid(),
        )
        self.head_knot_v = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, MAX_SURF_KNOTS),
            nn.Sigmoid(),
        )

        if n_ctrl_u is not None and n_ctrl_v is not None:
            self._build_expand(n_ctrl_u * n_ctrl_v)

    def _build_expand(self, n_tokens):
        self.n_tokens = n_tokens
        self.expand = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, n_tokens * self.hidden_dim),
        )

    def set_grid_size(self, n_ctrl_u, n_ctrl_v):
        """Call this after loading data to set the actual grid dimensions."""
        self.n_ctrl_u = n_ctrl_u
        self.n_ctrl_v = n_ctrl_v
        n_tokens = n_ctrl_u * n_ctrl_v
        if self.expand is None or self.n_tokens != n_tokens:
            self._build_expand(n_tokens)

    def forward(self, z):
        bs = z.shape[0]
        tokens = self.expand(z).view(bs, self.n_tokens, self.hidden_dim)
        ctrl_pts = self.head_ctrl(tokens).view(bs, self.n_ctrl_u, self.n_ctrl_v, 4)
        surf_u = self.head_knot_u(tokens).mean(dim=1)
        surf_v = self.head_knot_v(tokens).mean(dim=1)
        return {"surf_pw": ctrl_pts, "surf_u": surf_u, "surf_v": surf_v}


class UVDecoder(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 128 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, UV_CHANNELS, 3, padding=1),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)
        return self.deconv(x).permute(0, 2, 3, 1)


# =============================================================================
# FACE VAE
# =============================================================================

class FaceVAE(nn.Module):
    def __init__(self, embed_dim: int = 64,
                 disable_nurbs: bool = False, disable_uv: bool = False, disable_curves: bool = False):
        super().__init__()
        self.face_encoder = FaceEncoder(embed_dim, disable_nurbs=disable_nurbs,
                                        disable_uv=disable_uv, disable_curves=disable_curves)

        self.mu = nn.Linear(embed_dim, embed_dim)
        self.logvar = nn.Linear(embed_dim, embed_dim)

        self.nurbs_decoder = NURBSDecoder(embed_dim)
        self.uv_decoder = UVDecoder(embed_dim)

    def set_grid_size(self, n_ctrl_u, n_ctrl_v):
        """Set the control point grid size for the decoder."""
        self.nurbs_decoder.set_grid_size(n_ctrl_u, n_ctrl_v)

    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, batch):
        h = self.face_encoder(batch)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        nurbs_out = self.nurbs_decoder(z)
        nurbs_out["uv_samples"] = self.uv_decoder(z)
        return nurbs_out

    def get_embedding(self, batch):
        h = self.face_encoder(batch)
        return self.mu(h)


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device, beta=0.001):
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for face_batch in pbar:
        face_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in face_batch.items()}

        optimizer.zero_grad()

        z, mu, logvar = model.encode(face_batch)
        recon = model.decode(z)

        surf_mask = face_batch["surf_mask"].unsqueeze(-1)
        uv_mask = face_batch["uv_mask"].unsqueeze(-1)

        surf_pw_tgt, uv_samples_tgt, _ = normalize_coordinates_face(
            face_batch["surf_pw"], face_batch["surf_mask"],
            face_batch["uv_samples"], face_batch["uv_mask"]
        )

        surf_xyz_loss = masked_mse(recon["surf_pw"][..., :3], surf_pw_tgt[..., :3], surf_mask, n_channels=3)
        surf_w_loss = masked_mse(recon["surf_pw"][..., 3:], surf_pw_tgt[..., 3:], surf_mask, n_channels=1)

        has_nurbs = (face_batch["surf_mask"].sum(dim=(1, 2)) > 0).float()
        n_with_nurbs = has_nurbs.sum().clamp(min=1.0)

        surf_u_diff = ((recon["surf_u"] - face_batch["surf_u"]) ** 2).mean(dim=1)
        surf_v_diff = ((recon["surf_v"] - face_batch["surf_v"]) ** 2).mean(dim=1)
        surf_u_loss = (surf_u_diff * has_nurbs).sum() / n_with_nurbs
        surf_v_loss = (surf_v_diff * has_nurbs).sum() / n_with_nurbs

        recon_loss = (
            surf_xyz_loss + 0.1 * surf_w_loss +
            masked_mse(recon["uv_samples"], uv_samples_tgt, uv_mask, n_channels=UV_CHANNELS) +
            0.1 * (surf_u_loss + surf_v_loss)
        )
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + beta * kl_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch(model, loader, device, beta=0.001):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for face_batch in loader:
        face_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in face_batch.items()}

        z, mu, logvar = model.encode(face_batch)
        recon = model.decode(z)

        surf_mask = face_batch["surf_mask"].unsqueeze(-1)
        uv_mask = face_batch["uv_mask"].unsqueeze(-1)

        surf_pw_tgt, uv_samples_tgt, _ = normalize_coordinates_face(
            face_batch["surf_pw"], face_batch["surf_mask"],
            face_batch["uv_samples"], face_batch["uv_mask"]
        )

        surf_xyz_loss = masked_mse(recon["surf_pw"][..., :3], surf_pw_tgt[..., :3], surf_mask, n_channels=3)
        surf_w_loss = masked_mse(recon["surf_pw"][..., 3:], surf_pw_tgt[..., 3:], surf_mask, n_channels=1)

        has_nurbs = (face_batch["surf_mask"].sum(dim=(1, 2)) > 0).float()
        n_with_nurbs = has_nurbs.sum().clamp(min=1.0)

        surf_u_diff = ((recon["surf_u"] - face_batch["surf_u"]) ** 2).mean(dim=1)
        surf_v_diff = ((recon["surf_v"] - face_batch["surf_v"]) ** 2).mean(dim=1)
        surf_u_loss = (surf_u_diff * has_nurbs).sum() / n_with_nurbs
        surf_v_loss = (surf_v_diff * has_nurbs).sum() / n_with_nurbs

        recon_loss = (
            surf_xyz_loss + 0.1 * surf_w_loss +
            masked_mse(recon["uv_samples"], uv_samples_tgt, uv_mask, n_channels=UV_CHANNELS) +
            0.1 * (surf_u_loss + surf_v_loss)
        )
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + beta * kl_loss
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train Face VAE (separate)")
    parser.add_argument("--joint_nurbs_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize_uv_bbox", action="store_true",
                        help="Normalize UV sample xyz by bbox diagonal (matching NURBS ctrl pts)")
    parser.add_argument("--disable_nurbs", action="store_true",
                        help="Zero out NURBS encoder branch (geometry ablation)")
    parser.add_argument("--disable_uv", action="store_true",
                        help="Zero out UV encoder branch (geometry ablation)")
    parser.add_argument("--disable_curves", action="store_true",
                        help="Zero out boundary curve encoder branch (geometry ablation)")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joint_nurbs_dir = Path(args.joint_nurbs_dir)

    print(f"\n{'='*60}")
    print("FACE VAE TRAINING (Separate)")
    print(f"{'='*60}")
    print(f"Device: {device} | Embed dim: {args.embed_dim} | Beta: {args.beta}")

    # ---- splits ----
    splits = None
    train_models = None
    val_models = None
    all_models = None

    if args.split_file is not None:
        splits = load_split_file(Path(args.split_file))
        train_models = splits["train"]
        val_models = splits["val"]
        all_models = splits["all"]
        print(f"Split: {len(train_models)} train, {len(val_models)} val, {len(splits['test'])} test")
    else:
        print("WARNING: No split file; training on ALL models.")

    # ---- load data ----
    print(f"UV bbox normalization: {'ON' if args.normalize_uv_bbox else 'OFF'}")
    disabled = []
    if args.disable_nurbs: disabled.append("NURBS")
    if args.disable_uv: disabled.append("UV")
    if args.disable_curves: disabled.append("Curves")
    if disabled:
        print(f"DISABLED branches: {', '.join(disabled)}")
    else:
        print("All encoder branches ENABLED")
    train_data = load_face_data(joint_nurbs_dir, allowed_models=train_models,
                                normalize_uv_bbox=args.normalize_uv_bbox)
    if len(train_data) == 0:
        print("ERROR: No training face data found.")
        return

    val_data = []
    have_val = (val_models is not None and len(val_models) > 0)
    if have_val:
        val_data = load_face_data(joint_nurbs_dir, allowed_models=val_models,
                                  normalize_uv_bbox=args.normalize_uv_bbox)
        if len(val_data) == 0:
            print("WARNING: Val data empty; disabling val eval.")
            have_val = False

    print(f"Data: TRAIN {len(train_data)} faces" +
          (f" | VAL {len(val_data)} faces" if have_val else ""))

    # Detect actual grid size from data
    sample_surf = train_data[0]["surf_pw"]
    n_ctrl_u, n_ctrl_v = sample_surf.shape[0], sample_surf.shape[1]
    print(f"Detected control point grid: {n_ctrl_u}x{n_ctrl_v} ({n_ctrl_u * n_ctrl_v} tokens)\n")

    # ---- loaders ----
    train_loader = DataLoader(
        FaceDataset(train_data), batch_size=args.batch_size,
        shuffle=True, collate_fn=face_collate_fn, drop_last=True,
    )
    val_loader = None
    if have_val:
        val_loader = DataLoader(
            FaceDataset(val_data), batch_size=args.batch_size,
            shuffle=False, collate_fn=face_collate_fn, drop_last=False,
        )

    # ---- model ----
    model = FaceVAE(embed_dim=args.embed_dim,
                    disable_nurbs=args.disable_nurbs,
                    disable_uv=args.disable_uv,
                    disable_curves=args.disable_curves).to(device)
    model.set_grid_size(n_ctrl_u, n_ctrl_v)
    # Move newly created expand layer to device
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-6)

    history = {"train_loss": [], "val_loss": []}
    best_metric = float("inf")

    print(f"{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, beta=args.beta)
        scheduler.step()

        metric = train_loss
        val_loss = None

        if have_val and val_loader is not None:
            val_loss = eval_epoch(model, val_loader, device, beta=args.beta)
            metric = val_loss
            history["val_loss"].append(val_loss)

        history["train_loss"].append(train_loss)

        if metric < best_metric:
            best_metric = metric
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_metric": best_metric,
                "grid_size": (n_ctrl_u, n_ctrl_v),
                "config": vars(args),
            }, output_dir / "face_vae_best.pt")

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            if have_val:
                print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_metric:.4f}")
            else:
                print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Best: {best_metric:.4f}")

    # ---- plot ----
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="train")
    if have_val and len(history["val_loss"]) == len(history["train_loss"]):
        plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Face VAE Training")
    plt.legend()
    plt.savefig(output_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- extract per-model face embeddings ----
    print(f"\n{'='*60}")
    print("EXTRACTING PER-MODEL FACE EMBEDDINGS")
    print(f"{'='*60}")

    ckpt = torch.load(output_dir / "face_vae_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    embed_dir = output_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    if all_models is not None and len(all_models) > 0:
        model_list = list(all_models)
    else:
        model_list = [f.stem.replace("_joint_nurbs", "")
                     for f in joint_nurbs_dir.glob("*_joint_nurbs.pkl")]

    embed_dim = model.mu.out_features

    print(f"Extracting face embeddings for {len(model_list)} models...")

    for model_name in tqdm(model_list, desc="Extracting"):
        nurbs_file = joint_nurbs_dir / f"{model_name}_joint_nurbs.pkl"
        if not nurbs_file.exists():
            continue

        with open(nurbs_file, "rb") as f:
            nurbs_data = pickle.load(f)

        face_data_list = []
        face_indices = []
        for i, face_data in enumerate(nurbs_data["face_data"]):
            face = {
                "surf_pw": face_data.get("surf_pw", np.zeros((MAX_SURF_CTRL_U, MAX_SURF_CTRL_V, 4))),
                "surf_u": face_data.get("surf_u", np.zeros(MAX_SURF_KNOTS)),
                "surf_v": face_data.get("surf_v", np.zeros(MAX_SURF_KNOTS)),
                "surf_mask": face_data.get("surf_mask", np.zeros((MAX_SURF_CTRL_U, MAX_SURF_CTRL_V))),
                "uv_samples": face_data.get("uv_samples", np.zeros((UV_SAMPLES_U, UV_SAMPLES_V, UV_CHANNELS))),
                "uv_mask": face_data.get("uv_mask", np.zeros((UV_SAMPLES_U, UV_SAMPLES_V))),
                "curves_pw": face_data.get("curves_pw", np.zeros((MAX_CURVES, MAX_CURVE_CTRL, 4))),
                "curves_u": face_data.get("curves_u", np.zeros((MAX_CURVES, MAX_CURVE_KNOTS))),
                "curves_mask": face_data.get("curves_mask", np.zeros(MAX_CURVES)),
                "curves_ctrl_mask": face_data.get("curves_ctrl_mask", np.zeros((MAX_CURVES, MAX_CURVE_CTRL))),
                "curve_orientation": face_data.get("curve_orientation", np.zeros(MAX_CURVES)),
            }
            for k, v in face.items():
                if isinstance(v, np.ndarray):
                    face[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            face_data_list.append(face)
            face_indices.append(int(face_data.get("face_idx", i)))

        if len(face_data_list) == 0:
            continue

        loader = DataLoader(
            FaceDataset(face_data_list), batch_size=args.batch_size,
            shuffle=False, collate_fn=face_collate_fn, drop_last=False,
        )

        embeds = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                z = model.get_embedding(batch)
                embeds.append(z.cpu().numpy())

        face_embeddings = np.concatenate(embeds, axis=0)

        out_data = {
            "face_embeddings": face_embeddings,
            "model_name": model_name,
            "n_faces": len(face_embeddings),
            "face_indices": face_indices,
        }

        with open(embed_dir / f"{model_name}_face_embeddings.pkl", "wb") as f:
            pickle.dump(out_data, f)

    # ---- save config ----
    cfg = vars(args).copy()
    cfg["grid_size"] = [n_ctrl_u, n_ctrl_v]
    if splits is not None:
        cfg["split_counts"] = {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        }
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nDone! Embeddings: {embed_dir}")


if __name__ == "__main__":
    main()

# Step 1: Train each VAE independently
#python train_face_vae.py --joint_nurbs_dir /home/konstantinos/Downloads/s2.0.0/breps/nurbs_10x10 --output_dir /home/konstantinos/Downloads/s2.0.0/breps --split_file ./train_test.json --epochs 100 --device cuda