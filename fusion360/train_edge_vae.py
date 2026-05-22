"""
Edge VAE Training (Separate)
=============================

Trains ONLY the edge encoder/decoder VAE:
  - EdgeEncoder (Conv1D + knot embedding)
  - CurveDecoder (ConvTranspose1d)
  - KnotDecoder (MLP)

After training, extracts per-model edge embeddings.

Usage:
    python train_edge_vae.py \\
        --joint_nurbs_dir /path/to/joint_nurbs \\
        --output_dir ./output_edge_vae \\
        --split_file ./train_test.json \\
        --epochs 100 --batch_size 64 --device cuda
"""

import os
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
    MAX_CURVE_CTRL, MAX_CURVE_KNOTS, EMBED_DIM,
    set_seed, load_split_file, masked_mse,
    normalize_coordinates_edge,
    load_edge_data, EdgeDataset,
)


# =============================================================================
# MODELS
# =============================================================================

class EdgeEncoder(nn.Module):
    """Encodes a single edge (curve)."""
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
            curve_pw, _ = normalize_coordinates_edge(curve_pw, curve_mask)

        x = curve_pw.permute(0, 2, 1)
        if curve_mask is not None:
            x = x * curve_mask.unsqueeze(1)
        conv_feat = self.conv(x).contiguous().view(bs, -1)
        knot_feat = self.knot_embed(curve_u)
        return self.mlp(torch.cat([conv_feat, knot_feat], dim=-1))


class CurveDecoder(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 64 * 5)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv1d(32, 4, 3, padding=1),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 64, 5)
        return self.deconv(x).permute(0, 2, 1)  # (bs, 20, 4)


class KnotDecoder(nn.Module):
    def __init__(self, embed_dim: int = 64, output_dim: int = 25):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.mlp(z)


# =============================================================================
# EDGE VAE
# =============================================================================

class EdgeVAE(nn.Module):
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.edge_encoder = EdgeEncoder(embed_dim)

        self.mu = nn.Linear(embed_dim, embed_dim)
        self.logvar = nn.Linear(embed_dim, embed_dim)

        self.curve_decoder = CurveDecoder(embed_dim)
        self.knot_decoder = KnotDecoder(embed_dim, MAX_CURVE_KNOTS)

    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, curve_pw, curve_u, curve_mask=None):
        h = self.edge_encoder(curve_pw, curve_u, curve_mask)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return {
            "curve_pw": self.curve_decoder(z),
            "curve_u": self.knot_decoder(z),
        }

    def get_embedding(self, curve_pw, curve_u, curve_mask=None):
        h = self.edge_encoder(curve_pw, curve_u, curve_mask)
        return self.mu(h)


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device, beta=0.001, noise_std=0.1):
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for edge_batch in pbar:
        edge_batch = {k: v.to(device) for k, v in edge_batch.items()}

        optimizer.zero_grad()

        # Denoising: add noise to prevent trivial solution
        noisy_curve_pw = edge_batch["curve_pw"].clone()
        noisy_curve_pw[..., :3] = noisy_curve_pw[..., :3] + noise_std * torch.randn_like(noisy_curve_pw[..., :3])

        z, mu, logvar = model.encode(noisy_curve_pw, edge_batch["curve_u"], edge_batch["curve_mask"])
        recon = model.decode(z)

        edge_mask = edge_batch["curve_mask"].unsqueeze(-1)

        # Normalize target
        curve_pw_tgt, _ = normalize_coordinates_edge(edge_batch["curve_pw"], edge_batch["curve_mask"])

        edge_xyz_loss = masked_mse(recon["curve_pw"][..., :3], curve_pw_tgt[..., :3], edge_mask, n_channels=3)
        edge_w_loss = masked_mse(recon["curve_pw"][..., 3:], curve_pw_tgt[..., 3:], edge_mask, n_channels=1)
        edge_u_loss = F.mse_loss(recon["curve_u"], edge_batch["curve_u"])

        recon_loss = edge_xyz_loss + 0.1 * edge_w_loss + 0.1 * edge_u_loss
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

    for edge_batch in loader:
        edge_batch = {k: v.to(device) for k, v in edge_batch.items()}

        z, mu, logvar = model.encode(edge_batch["curve_pw"], edge_batch["curve_u"], edge_batch["curve_mask"])
        recon = model.decode(z)

        edge_mask = edge_batch["curve_mask"].unsqueeze(-1)
        curve_pw_tgt, _ = normalize_coordinates_edge(edge_batch["curve_pw"], edge_batch["curve_mask"])

        edge_xyz_loss = masked_mse(recon["curve_pw"][..., :3], curve_pw_tgt[..., :3], edge_mask, n_channels=3)
        edge_w_loss = masked_mse(recon["curve_pw"][..., 3:], curve_pw_tgt[..., 3:], edge_mask, n_channels=1)
        edge_u_loss = F.mse_loss(recon["curve_u"], edge_batch["curve_u"])

        recon_loss = edge_xyz_loss + 0.1 * edge_w_loss + 0.1 * edge_u_loss
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

    parser = argparse.ArgumentParser(description="Train Edge VAE (separate)")
    parser.add_argument("--joint_nurbs_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="Denoising noise std for edge training")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joint_nurbs_dir = Path(args.joint_nurbs_dir)

    print(f"\n{'='*60}")
    print("EDGE VAE TRAINING (Separate)")
    print(f"{'='*60}")
    print(f"Device: {device} | Embed dim: {args.embed_dim} | Beta: {args.beta} | Noise: {args.noise_std}")

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
    train_data = load_edge_data(joint_nurbs_dir, allowed_models=train_models)
    if len(train_data) == 0:
        print("ERROR: No training edge data found.")
        return

    val_data = []
    have_val = (val_models is not None and len(val_models) > 0)
    if have_val:
        val_data = load_edge_data(joint_nurbs_dir, allowed_models=val_models)
        if len(val_data) == 0:
            print("WARNING: Val data empty; disabling val eval.")
            have_val = False

    print(f"Data: TRAIN {len(train_data)} edges" +
          (f" | VAL {len(val_data)} edges" if have_val else ""))

    # Edge stats
    if len(train_data) > 0:
        sample_size = min(100, len(train_data))
        avg_ctrl = np.mean([np.sum(e["curve_mask"]) for e in train_data[:sample_size]])
        avg_xyz_std = np.mean([np.std(e["curve_pw"][:, :3]) for e in train_data[:sample_size]])
        print(f"Edge stats (sample {sample_size}): avg ctrl pts={avg_ctrl:.1f}, avg xyz std={avg_xyz_std:.6f}\n")

    # ---- loaders ----
    train_loader = DataLoader(
        EdgeDataset(train_data), batch_size=args.batch_size,
        shuffle=True, drop_last=True,
    )
    val_loader = None
    if have_val:
        val_loader = DataLoader(
            EdgeDataset(val_data), batch_size=args.batch_size,
            shuffle=False, drop_last=False,
        )

    # ---- model ----
    model = EdgeVAE(embed_dim=args.embed_dim).to(device)
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
        train_loss = train_epoch(model, train_loader, optimizer, device,
                                 beta=args.beta, noise_std=args.noise_std)
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
                "config": vars(args),
            }, output_dir / "edge_vae_best.pt")

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
    plt.title("Edge VAE Training")
    plt.legend()
    plt.savefig(output_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- extract per-model edge embeddings ----
    print(f"\n{'='*60}")
    print("EXTRACTING PER-MODEL EDGE EMBEDDINGS")
    print(f"{'='*60}")

    ckpt = torch.load(output_dir / "edge_vae_best.pt", map_location=device)
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

    print(f"Extracting edge embeddings for {len(model_list)} models...")

    for model_name in tqdm(model_list, desc="Extracting"):
        nurbs_file = joint_nurbs_dir / f"{model_name}_joint_nurbs.pkl"
        if not nurbs_file.exists():
            continue

        with open(nurbs_file, "rb") as f:
            nurbs_data = pickle.load(f)

        edge_data_list = nurbs_data.get("edge_data", [])
        edge_embeddings = np.zeros((len(edge_data_list), embed_dim), dtype=np.float32)

        if len(edge_data_list) > 0:
            edge_batch_data = []
            for ed in edge_data_list:
                edge_batch_data.append({
                    "curve_pw": np.nan_to_num(ed.get("curve_pw", np.zeros((MAX_CURVE_CTRL, 4))), nan=0.0),
                    "curve_u": np.nan_to_num(ed.get("curve_u", np.zeros(MAX_CURVE_KNOTS)), nan=0.0),
                    "curve_mask": np.nan_to_num(ed.get("curve_mask", np.zeros(MAX_CURVE_CTRL)), nan=0.0),
                })

            loader = DataLoader(
                EdgeDataset(edge_batch_data), batch_size=args.batch_size,
                shuffle=False, drop_last=False,
            )

            embeds = []
            with torch.no_grad():
                for batch in loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    z = model.get_embedding(batch["curve_pw"], batch["curve_u"], batch["curve_mask"])
                    embeds.append(z.cpu().numpy())

            edge_embeddings = np.concatenate(embeds, axis=0)

        out_data = {
            "edge_embeddings": edge_embeddings,
            "model_name": model_name,
            "n_edges": len(edge_embeddings),
        }

        with open(embed_dir / f"{model_name}_edge_embeddings.pkl", "wb") as f:
            pickle.dump(out_data, f)

    # ---- save config ----
    cfg = vars(args).copy()
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

#python train_edge_vae.py --joint_nurbs_dir /home/konstantinos/Downloads/s2.0.0/breps/nurbs_10x10 --output_dir /home/konstantinos/Downloads/s2.0.0/breps --split_file ./train_test.json --epochs 100 --device cuda