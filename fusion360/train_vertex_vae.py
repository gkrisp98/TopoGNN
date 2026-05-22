"""
Vertex VAE Training (Separate)
================================

Trains ONLY the vertex encoder/decoder VAE:
  - VertexEncoder (Fourier positional encoding + MLP)
  - VertexDecoder (MLP)

After training, extracts per-model vertex embeddings.

Usage:
    python train_vertex_vae.py \\
        --handcrafted_dir /path/to/handcrafted \\
        --output_dir ./output_vertex_vae \\
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
    EMBED_DIM,
    set_seed, load_split_file,
    normalize_vertex_positions,
    load_vertex_data, VertexDataset,
)


# =============================================================================
# MODELS
# =============================================================================

class VertexEncoder(nn.Module):
    """Encodes vertex from position + local geometric features."""
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
        self.register_buffer("frequencies", frequencies)

    def fourier_encode(self, positions):
        pos_expanded = positions.unsqueeze(-1) * self.frequencies
        feats = torch.cat([torch.sin(pos_expanded * math.pi), torch.cos(pos_expanded * math.pi)], dim=-1)
        return feats.view(positions.shape[0], -1)

    def forward(self, vertex_features):
        positions = vertex_features[:, :3]
        fourier_feats = self.fourier_encode(positions)
        combined = torch.cat([fourier_feats, vertex_features], dim=-1)
        return self.mlp(combined)


class VertexDecoder(nn.Module):
    """Decodes vertex embedding back to ALL vertex features."""
    def __init__(self, embed_dim: int = 64, output_dim: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, z):
        return self.mlp(z)


# =============================================================================
# VERTEX VAE
# =============================================================================

class VertexVAE(nn.Module):
    def __init__(self, embed_dim: int = 64, vertex_input_dim: int = 8):
        super().__init__()
        self.vertex_encoder = VertexEncoder(vertex_input_dim, embed_dim)

        self.mu = nn.Linear(embed_dim, embed_dim)
        self.logvar = nn.Linear(embed_dim, embed_dim)

        self.vertex_decoder = VertexDecoder(embed_dim, output_dim=vertex_input_dim)

    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, vertex_features):
        h = self.vertex_encoder(vertex_features)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.vertex_decoder(z)

    def get_embedding(self, vertex_features):
        h = self.vertex_encoder(vertex_features)
        return self.mu(h)


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device, beta=0.001):
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for vertex_batch in pbar:
        vertex_batch = vertex_batch.to(device)

        optimizer.zero_grad()

        z, mu, logvar = model.encode(vertex_batch)
        recon = model.decode(z)

        # Dimension-weighted loss: skip dist_to_cog (index 3)
        # Dims: [xyz(3), dist_to_cog(1), valence(1), avg_edge_len(1), is_boundary(1), avg_dihedral(1)]
        vertex_xyz_loss = F.mse_loss(recon[:, :3], vertex_batch[:, :3])
        vertex_other_loss = F.mse_loss(recon[:, 4:], vertex_batch[:, 4:])
        recon_loss = vertex_xyz_loss + 0.5 * vertex_other_loss

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

    for vertex_batch in loader:
        vertex_batch = vertex_batch.to(device)

        z, mu, logvar = model.encode(vertex_batch)
        recon = model.decode(z)

        vertex_xyz_loss = F.mse_loss(recon[:, :3], vertex_batch[:, :3])
        vertex_other_loss = F.mse_loss(recon[:, 4:], vertex_batch[:, 4:])
        recon_loss = vertex_xyz_loss + 0.5 * vertex_other_loss

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

    parser = argparse.ArgumentParser(description="Train Vertex VAE (separate)")
    parser.add_argument("--handcrafted_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handcrafted_dir = Path(args.handcrafted_dir)

    print(f"\n{'='*60}")
    print("VERTEX VAE TRAINING (Separate)")
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
    train_data = load_vertex_data(handcrafted_dir, allowed_models=train_models)
    if len(train_data) == 0:
        print("ERROR: No training vertex data found.")
        return

    val_data = []
    have_val = (val_models is not None and len(val_models) > 0)
    if have_val:
        val_data = load_vertex_data(handcrafted_dir, allowed_models=val_models)
        if len(val_data) == 0:
            print("WARNING: Val data empty; disabling val eval.")
            have_val = False

    print(f"Data: TRAIN {len(train_data)} vertices" +
          (f" | VAL {len(val_data)} vertices" if have_val else ""))

    vertex_dim = len(train_data[0]) if train_data else 8
    print(f"Vertex feature dim: {vertex_dim}\n")

    # ---- loaders ----
    train_loader = DataLoader(
        VertexDataset(train_data), batch_size=args.batch_size,
        shuffle=True, drop_last=True,
    )
    val_loader = None
    if have_val:
        val_loader = DataLoader(
            VertexDataset(val_data), batch_size=args.batch_size,
            shuffle=False, drop_last=False,
        )

    # ---- model ----
    model = VertexVAE(embed_dim=args.embed_dim, vertex_input_dim=vertex_dim).to(device)
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
                "config": vars(args),
            }, output_dir / "vertex_vae_best.pt")

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
    plt.title("Vertex VAE Training")
    plt.legend()
    plt.savefig(output_dir / "training_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- extract per-model vertex embeddings ----
    print(f"\n{'='*60}")
    print("EXTRACTING PER-MODEL VERTEX EMBEDDINGS")
    print(f"{'='*60}")

    ckpt = torch.load(output_dir / "vertex_vae_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    embed_dir = output_dir / "embeddings"
    embed_dir.mkdir(parents=True, exist_ok=True)

    if all_models is not None and len(all_models) > 0:
        model_list = list(all_models)
    else:
        model_list = [f.stem.replace("_handcrafted", "")
                     for f in handcrafted_dir.glob("*_handcrafted.pkl")]

    embed_dim = model.mu.out_features

    print(f"Extracting vertex embeddings for {len(model_list)} models...")

    for model_name in tqdm(model_list, desc="Extracting"):
        handcrafted_file = handcrafted_dir / f"{model_name}_handcrafted.pkl"
        if not handcrafted_file.exists():
            continue

        with open(handcrafted_file, "rb") as f:
            handcrafted_data = pickle.load(f)

        vertex_features = handcrafted_data.get("vertex_features", None)
        vertex_embeddings = np.zeros((0, embed_dim), dtype=np.float32)

        if vertex_features is not None and len(vertex_features) > 0:
            vertex_features = normalize_vertex_positions(np.array(vertex_features))

            loader = DataLoader(
                VertexDataset(list(vertex_features)), batch_size=args.batch_size,
                shuffle=False, drop_last=False,
            )

            embeds = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    z = model.get_embedding(batch)
                    embeds.append(z.cpu().numpy())

            vertex_embeddings = np.concatenate(embeds, axis=0)

        out_data = {
            "vertex_embeddings": vertex_embeddings,
            "model_name": model_name,
            "n_vertices": len(vertex_embeddings),
        }

        with open(embed_dir / f"{model_name}_vertex_embeddings.pkl", "wb") as f:
            pickle.dump(out_data, f)

    # ---- save config ----
    cfg = vars(args).copy()
    cfg["vertex_dim"] = vertex_dim
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

#python train_vertex_vae.py --handcrafted_dir /home/konstantinos/Downloads/s2.0.0/breps/handcrafted --output_dir /home/konstantinos/Downloads/s2.0.0/breps --split_file ./train_test.json --epochs 100 --device cuda