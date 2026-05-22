"""
Visualize VAE Embeddings - t-SNE / UMAP scatter plots colored by class.
========================================================================

Pools per-graph node features (face, coedge, vertex) and projects to 2D
to see if the VAE embeddings already separate classes before GNN training.

Usage:
  python visualize_embeddings.py --graphs_dir /path/to/graphs_coedge --output_dir ./plots
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pool_graph_features(g):
    """Pool node features into a single graph-level vector (mean + std per node type)."""
    parts = []
    for ntype in ['face', 'coedge', 'vertex']:
        if ntype in g.node_types and hasattr(g[ntype], 'x') and g[ntype].x is not None and g[ntype].x.numel() > 0:
            x = g[ntype].x.float()
            parts.append(x.mean(dim=0))
            parts.append(x.std(dim=0).clamp(min=1e-8))
    if parts:
        return torch.cat(parts).numpy()
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize VAE embeddings with t-SNE/UMAP")
    parser.add_argument("--graphs_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./embedding_plots")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Subsample for faster plotting (default: use all)")
    parser.add_argument("--method", type=str, default="both", choices=["tsne", "umap", "both"])
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--paper", action="store_true",
                        help="Generate clean plots for paper (no titles, high DPI)")
    args = parser.parse_args()

    graphs_dir = Path(args.graphs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load class map
    class_map_path = graphs_dir / 'class_map.json'
    if class_map_path.exists():
        with open(class_map_path) as f:
            class_map = json.load(f)
        class_names = sorted(class_map.keys(), key=lambda x: class_map[x])
    else:
        class_names = None

    # Load graphs
    print("Loading graphs...")
    graph_feats = []
    graph_labels = []
    graph_names = []

    pt_files = sorted(graphs_dir.glob("*.pt"))
    pt_files = [f for f in pt_files if f.name not in ['all_graphs.pt', 'graph_stats.json']]

    for pt_file in pt_files:
        try:
            g = torch.load(pt_file, weights_only=False)
            if not hasattr(g, 'y') or g.y is None:
                continue
            feat = pool_graph_features(g)
            if feat is not None:
                graph_feats.append(feat)
                graph_labels.append(g.y.item())
                graph_names.append(pt_file.stem.replace('_graph', ''))
        except:
            continue

    X = np.array(graph_feats)
    y = np.array(graph_labels)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Loaded {len(X)} graphs, feature dim={X.shape[1]}")

    # Auto-detect classes
    num_classes = y.max() + 1
    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]

    # Print class distribution
    label_dist = Counter(y)
    print(f"Classes ({num_classes}):")
    for i, name in enumerate(class_names):
        print(f"  [{i}] {name}: {label_dist.get(i, 0)}")

    # Subsample if requested
    if args.max_samples and len(X) > args.max_samples:
        idx = np.random.RandomState(42).choice(len(X), args.max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        print(f"Subsampled to {args.max_samples}")

    # Standardize
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)

    # Quick linear probe accuracy
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X_scaled, y))
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled[train_idx], y[train_idx])
    lr_acc = lr.score(X_scaled[test_idx], y[test_idx])
    print(f"\nLinear probe accuracy: {100*lr_acc:.1f}% (random={100/num_classes:.1f}%)")

    # Colors
    cmap = matplotlib.colormaps.get_cmap('tab10')
    colors = [cmap(i) for i in range(num_classes)]

    def make_scatter(X_2d, y, title, save_path, no_title=False):
        if no_title:
            # IEEE column width ~3.5in, full width ~7.16in
            fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.2))
            dot_size = 6
            dot_alpha = 0.75
            legend_size = 6
            marker_scale = 2
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 7))
            dot_size = 18
            dot_alpha = 0.7
            legend_size = 9
            marker_scale = 2.5

        for i in range(num_classes):
            mask = y == i
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[colors[i]], label=class_names[i],
                       s=dot_size, alpha=dot_alpha, edgecolors='none')
        ax.legend(fontsize=legend_size, markerscale=marker_scale,
                  loc='upper right', framealpha=0.9, edgecolor='none',
                  handletextpad=0.3, borderpad=0.3, labelspacing=0.25)
        if not no_title:
            ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout(pad=0.3)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")

    # t-SNE
    if args.method in ('tsne', 'both'):
        print(f"\nRunning t-SNE (perplexity={args.perplexity})...")
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42,
                     max_iter=1000, learning_rate='auto', init='pca')
        X_tsne = tsne.fit_transform(X_scaled)
        make_scatter(X_tsne, y,
                     f"t-SNE of VAE Embeddings (linear probe={100*lr_acc:.1f}%)",
                     output_dir / "embeddings_tsne.png", no_title=args.paper)

    # UMAP
    if args.method in ('umap', 'both'):
        try:
            print("\nRunning UMAP...")
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
            X_umap = reducer.fit_transform(X_scaled)
            make_scatter(X_umap, y,
                         f"UMAP of VAE Embeddings (linear probe={100*lr_acc:.1f}%)",
                         output_dir / "embeddings_umap.png", no_title=args.paper)
        except ImportError:
            print("  umap-learn not installed, skipping UMAP (pip install umap-learn)")

    # Per-node-type plots (separate t-SNE for face, coedge, vertex features)
    if args.method in ('tsne', 'both'):
        print("\nPer-node-type t-SNE:")
        from sklearn.manifold import TSNE

        for ntype in ['face', 'coedge', 'vertex']:
            type_feats = []
            type_labels = []
            for pt_file in pt_files:
                try:
                    g = torch.load(pt_file, weights_only=False)
                    if not hasattr(g, 'y') or g.y is None:
                        continue
                    if ntype in g.node_types and hasattr(g[ntype], 'x') and g[ntype].x is not None and g[ntype].x.numel() > 0:
                        x = g[ntype].x.float()
                        feat = torch.cat([x.mean(dim=0), x.std(dim=0).clamp(min=1e-8)]).numpy()
                        type_feats.append(feat)
                        type_labels.append(g.y.item())
                except:
                    continue

            if len(type_feats) < 50:
                continue

            X_t = np.nan_to_num(np.array(type_feats), nan=0.0)
            y_t = np.array(type_labels)
            X_t_scaled = StandardScaler().fit_transform(X_t)

            # Linear probe for this node type
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            ti, vi = next(sss.split(X_t_scaled, y_t))
            lr_t = LogisticRegression(max_iter=1000, random_state=42)
            lr_t.fit(X_t_scaled[ti], y_t[ti])
            acc_t = lr_t.score(X_t_scaled[vi], y_t[vi])

            # Subsample if needed
            if args.max_samples and len(X_t) > args.max_samples:
                idx = np.random.RandomState(42).choice(len(X_t), args.max_samples, replace=False)
                X_t_scaled = X_t_scaled[idx]
                y_t = y_t[idx]

            tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42,
                         max_iter=1000, learning_rate='auto', init='pca')
            X_2d = tsne.fit_transform(X_t_scaled)
            make_scatter(X_2d, y_t,
                         f"{ntype.capitalize()} embeddings t-SNE (linear probe={100*acc_t:.1f}%)",
                         output_dir / f"embeddings_tsne_{ntype}.png", no_title=args.paper)

    print("\nDone!")


if __name__ == "__main__":
    main()

# Usage:
# python visualize_embeddings.py --graphs_dir /home/konstantinos/Downloads/mechcad/output/graphs_coedge --output_dir ./embedding_plots
