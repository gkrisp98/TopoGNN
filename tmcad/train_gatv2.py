"""
Train Heterogeneous GNN (GATv2) for MechCAD Graph Classification
=================================================================

GRAPH CLASSIFICATION (not face segmentation!):
- Each graph = one CAD model
- Label = model class (bearing, bolt, bracket, etc.)
- Uses global pooling to aggregate node features -> graph embedding

Architecture:
1. Input projection per node type (face, coedge, vertex)
2. GATv2 message passing on heterogeneous coedge graph
3. Multi-node global pooling (mean + max on face, coedge, vertex)
4. MLP classifier on graph embedding

Relations (from build_mechcad_graphs_coedge.py):
- face <-> coedge:  has / of
- coedge <-> coedge: partner, next, prev
- coedge <-> vertex: starts_at / start_of, ends_at / end_of

Usage:
    python train_gatv2_mechcad.py \\
        --graphs_dir ./mechcad/graphs_coedge \\
        --output_dir ./mechcad/models_gatv2 \\
        --hidden_dim 256 --num_heads 8 --num_layers 4 \\
        --batch_size 32 --epochs 200 --device cuda

    # With external split file:
    python train_gatv2_mechcad.py \\
        --graphs_dir ./mechcad/graphs_coedge \\
        --output_dir ./mechcad/models_gatv2 \\
        --split_file ./mechcad_split.json \\
        --device cuda
"""

import numpy as np
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear, LayerNorm
from torch_geometric.nn import global_mean_pool, global_max_pool
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# FEATURE NORMALIZATION
# =============================================================================

def detect_node_types(graphs: List[HeteroData]) -> List[str]:
    """Detect node types from graphs. Returns ordered list: [face, middle, vertex]."""
    for g in graphs:
        ntypes = set(g.node_types)
        if 'coedge' in ntypes:
            return ['face', 'coedge', 'vertex']
        elif 'edge' in ntypes:
            return ['face', 'edge', 'vertex']
    return ['face', 'coedge', 'vertex']


def compute_feature_statistics(graphs: List[HeteroData]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute mean and std for each node type's features."""
    node_types = detect_node_types(graphs)
    features = {nt: [] for nt in node_types}

    for g in graphs:
        for node_type in node_types:
            if node_type in g.node_types and g[node_type].x.shape[0] > 0:
                features[node_type].append(g[node_type].x.numpy())

    stats = {}
    for node_type, feat_list in features.items():
        if feat_list:
            all_feats = np.vstack(feat_list)
            mean = all_feats.mean(axis=0)
            std = all_feats.std(axis=0)
            std = np.where(std < 1e-6, 1.0, std)
            stats[node_type] = (mean.astype(np.float32), std.astype(np.float32))
        else:
            stats[node_type] = (np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32))

    return stats


def normalize_graphs(graphs: List[HeteroData], stats: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[HeteroData]:
    """Normalize features in all graphs using precomputed statistics."""
    for g in graphs:
        for node_type in stats:
            if node_type in g.node_types and g[node_type].x.shape[0] > 0:
                mean, std = stats[node_type]
                mean_t = torch.from_numpy(mean).to(g[node_type].x.device)
                std_t = torch.from_numpy(std).to(g[node_type].x.device)
                g[node_type].x = (g[node_type].x - mean_t) / std_t
    return graphs


def fix_nan_inf_in_graphs(graphs: List[HeteroData]) -> int:
    """Fix NaN/Inf values in graph features."""
    fixed_count = 0
    for g in graphs:
        has_issue = False
        for node_type in g.node_types:
            if g[node_type].x.shape[0] > 0:
                x = g[node_type].x
                if torch.isnan(x).any() or torch.isinf(x).any():
                    has_issue = True
                    g[node_type].x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        if has_issue:
            fixed_count += 1
    return fixed_count


# =============================================================================
# FOCAL LOSS WITH LABEL SMOOTHING
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 label_smoothing: float = 0.0, ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = targets != self.ignore_index
        inputs = inputs[mask]
        targets = targets[mask]

        if len(targets) == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        n_classes = inputs.shape[1]
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(inputs)
                smooth_targets.fill_(self.label_smoothing / (n_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)

            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


# =============================================================================
# GATv2 MODEL FOR GRAPH CLASSIFICATION (COEDGE VARIANT)
# =============================================================================

class HeteroGATv2GraphClassifier(nn.Module):
    """
    Heterogeneous GATv2 for GRAPH-LEVEL classification.

    Architecture:
    1. Input projections per node type
    2. GATv2 message passing with heterogeneous relations
    3. Global pooling (mean + max) on all node types
    4. MLP classifier

    Supports both coedge-based (face/coedge/vertex) and
    edge-based (face/edge/vertex) graph topologies.
    The conv layers are built dynamically from edge_types.
    """

    def __init__(
        self,
        face_dim: int,
        middle_dim: int,
        vertex_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_classes: int = 10,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        residual: bool = True,
        pool_all_nodes: bool = True,
        edge_types: List[tuple] = None,
        node_types: List[str] = None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.residual = residual
        self.pool_all_nodes = pool_all_nodes
        self.node_types = node_types or ['face', 'coedge', 'vertex']
        self.middle_type = self.node_types[1]  # 'coedge' or 'edge'

        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        # Input projections
        self.input_projs = nn.ModuleDict()
        dims = {'face': face_dim, self.middle_type: middle_dim, 'vertex': vertex_dim}
        for nt in self.node_types:
            self.input_projs[nt] = nn.Sequential(
                Linear(dims[nt], hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # GATv2 layers - build from actual edge_types
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer_idx in range(num_layers):
            conv_dict = {}
            for et in edge_types:
                src_type, rel, dst_type = et
                if src_type == dst_type:
                    conv_dict[et] = GATv2Conv(
                        hidden_dim, self.head_dim, heads=num_heads,
                        dropout=attention_dropout, add_self_loops=False, concat=True,
                        share_weights=False,
                    )
                else:
                    conv_dict[et] = GATv2Conv(
                        (hidden_dim, hidden_dim), self.head_dim, heads=num_heads,
                        dropout=attention_dropout, add_self_loops=False, concat=True,
                    )

            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

            self.norms.append(nn.ModuleDict({
                nt: nn.LayerNorm(hidden_dim) for nt in self.node_types
            }))

        self.dropout = nn.Dropout(dropout)

        # Global pooling: mean + max per node type
        if pool_all_nodes:
            pool_dim = hidden_dim * 6  # mean+max for 3 node types
        else:
            pool_dim = hidden_dim * 2  # mean+max for face only

        # Graph classifier head
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        device = data['face'].x.device

        # Input projections
        x_dict = {}
        for nt in self.node_types:
            if nt in data.node_types and data[nt].x.shape[0] > 0:
                x_dict[nt] = self.input_projs[nt](data[nt].x)
            else:
                x_dict[nt] = torch.zeros((0, self.hidden_dim), device=device)

        # GATv2 message passing
        for conv, norm in zip(self.convs, self.norms):
            x_dict_residual = {k: v.clone() for k, v in x_dict.items() if v.shape[0] > 0}

            x_dict_normed = {}
            for key in x_dict:
                if x_dict[key].shape[0] > 0:
                    x_dict_normed[key] = norm[key](x_dict[key])
                else:
                    x_dict_normed[key] = x_dict[key]

            edge_index_dict = {et: data.edge_index_dict[et] for et in data.edge_types
                              if et in data.edge_index_dict}

            x_dict_new = conv(x_dict_normed, edge_index_dict)

            for key in x_dict:
                if key in x_dict_new and x_dict_new[key].shape[0] > 0:
                    if self.residual and key in x_dict_residual:
                        x_dict[key] = x_dict_residual[key] + self.dropout(x_dict_new[key])
                    else:
                        x_dict[key] = self.dropout(F.gelu(x_dict_new[key]))

        # ===================================================================
        # GLOBAL POOLING: aggregate node features to graph-level
        # ===================================================================

        # Face pooling
        face_batch = data['face'].batch if hasattr(data['face'], 'batch') else torch.zeros(x_dict['face'].shape[0], dtype=torch.long, device=device)
        face_mean = global_mean_pool(x_dict['face'], face_batch)
        face_max = global_max_pool(x_dict['face'], face_batch)

        pool_parts = [face_mean, face_max]

        if self.pool_all_nodes:
            # Middle node (coedge or edge) pooling
            mid = self.middle_type
            if x_dict[mid].shape[0] > 0:
                mid_batch = data[mid].batch if hasattr(data[mid], 'batch') else torch.zeros(x_dict[mid].shape[0], dtype=torch.long, device=device)
                mid_mean = global_mean_pool(x_dict[mid], mid_batch)
                mid_max = global_max_pool(x_dict[mid], mid_batch)
            else:
                n_graphs = face_mean.shape[0]
                mid_mean = torch.zeros(n_graphs, self.hidden_dim, device=device)
                mid_max = torch.zeros(n_graphs, self.hidden_dim, device=device)
            pool_parts.extend([mid_mean, mid_max])

            # Vertex pooling
            if x_dict['vertex'].shape[0] > 0:
                vertex_batch = data['vertex'].batch if hasattr(data['vertex'], 'batch') else torch.zeros(x_dict['vertex'].shape[0], dtype=torch.long, device=device)
                vertex_mean = global_mean_pool(x_dict['vertex'], vertex_batch)
                vertex_max = global_max_pool(x_dict['vertex'], vertex_batch)
            else:
                n_graphs = face_mean.shape[0]
                vertex_mean = torch.zeros(n_graphs, self.hidden_dim, device=device)
                vertex_max = torch.zeros(n_graphs, self.hidden_dim, device=device)
            pool_parts.extend([vertex_mean, vertex_max])

        graph_embed = torch.cat(pool_parts, dim=-1)
        self._last_graph_embed = graph_embed  # store for visualization
        return self.classifier(graph_embed)

    @torch.no_grad()
    def get_graph_embeddings(self, loader, device):
        """Extract graph-level embeddings (before classifier) for all graphs in loader."""
        self.eval()
        all_embeds = []
        all_labels = []
        for data in loader:
            data = data.to(device)
            _ = self(data)  # forward pass populates _last_graph_embed
            all_embeds.append(self._last_graph_embed.cpu())
            if hasattr(data, 'y'):
                all_labels.append(data.y.cpu())
        embeds = torch.cat(all_embeds, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy() if all_labels else None
        return embeds, labels


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def has_labels(graph: HeteroData) -> bool:
    """Check if graph has a valid graph-level label."""
    if not hasattr(graph, 'y') or graph.y is None:
        return False
    return (graph.y >= 0).all().item()


def get_feature_dims(graphs: List[HeteroData]) -> Tuple[int, int, int]:
    """Get feature dimensions from graphs. Returns (face_dim, middle_dim, vertex_dim)."""
    node_types = detect_node_types(graphs)
    middle_type = node_types[1]  # 'coedge' or 'edge'
    for g in graphs:
        face_dim = g['face'].x.shape[1]
        middle_dim = g[middle_type].x.shape[1] if middle_type in g.node_types and g[middle_type].x.shape[0] > 0 else 64
        vertex_dim = g['vertex'].x.shape[1] if 'vertex' in g.node_types and g['vertex'].x.shape[0] > 0 else 8
        return face_dim, middle_dim, vertex_dim
    return 64, 64, 8


def compute_class_weights(graphs: List[HeteroData], num_classes: int) -> torch.Tensor:
    """Compute inverse frequency class weights for graph labels."""
    all_labels = []
    for g in graphs:
        if has_labels(g):
            all_labels.append(g.y.item())

    if not all_labels:
        return torch.ones(num_classes)

    counts = Counter(all_labels)
    total = sum(counts.values())

    weights = torch.ones(num_classes)
    for c, count in counts.items():
        if c < num_classes:
            weights[c] = total / (count * num_classes)

    weights = torch.clamp(weights, min=0.1, max=50.0)
    return weights


def train_epoch(model, loader, optimizer, criterion, device, num_classes,
                grad_accum_steps: int = 1, max_grad_norm: float = 1.0):
    """Train for one epoch — graph classification."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    optimizer.zero_grad()

    for batch_idx, data in enumerate(loader):
        data = data.to(device)

        out = model(data)  # (batch_size, num_classes)
        labels = data.y.view(-1)  # (batch_size,)

        mask = (labels >= 0) & (labels < num_classes)
        if mask.sum() == 0:
            continue

        loss = criterion(out[mask], labels[mask])

        if torch.isnan(loss):
            continue

        loss = loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps * mask.sum().item()
        total_correct += (out[mask].argmax(dim=1) == labels[mask]).sum().item()
        total_samples += mask.sum().item()

    # Handle remaining gradients
    if (batch_idx + 1) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def evaluate(model, loader, criterion, device, num_classes):
    """Evaluate model — graph classification."""
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            labels = data.y.view(-1)

            if torch.isnan(out).any():
                continue

            mask = (labels >= 0) & (labels < num_classes)
            if mask.sum() == 0:
                continue

            loss = criterion(out[mask], labels[mask])
            if not torch.isnan(loss):
                total_loss += loss.item() * mask.sum().item()
            total_samples += mask.sum().item()

            all_preds.extend(out[mask].argmax(dim=1).cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())

    if not all_preds:
        return 0.0, 0.0, 0.0, [], []

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return total_loss / max(total_samples, 1), accuracy, f1_macro, all_preds, all_labels


# =============================================================================
# SPLIT GENERATION / LOADING
# =============================================================================

def generate_stratified_split(graphs: List[HeteroData], train_ratio=0.7, val_ratio=0.15,
                               test_ratio=0.15, seed=42) -> Tuple[List, List, List]:
    """Generate a stratified train/val/test split based on graph labels."""
    labels = [g.y.item() for g in graphs]
    names = [g.model_name for g in graphs]

    # First split: train+val vs test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(names, labels))

    # Second split: train vs val (from trainval)
    trainval_labels = [labels[i] for i in trainval_idx]
    relative_val = val_ratio / (train_ratio + val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    train_sub_idx, val_sub_idx = next(sss2.split(trainval_idx, trainval_labels))

    train_names = [names[trainval_idx[i]] for i in train_sub_idx]
    val_names = [names[trainval_idx[i]] for i in val_sub_idx]
    test_names = [names[test_idx[i]] for i in range(len(test_idx))]

    return train_names, val_names, test_names


def normalize_model_name(name: str) -> str:
    if name.endswith('_graph'):
        name = name[:-6]
    return name.lower()


def load_split(split_file: Path, graph_names: Dict[str, HeteroData]) -> Tuple[List, List, List]:
    """Load train/val/test split from JSON file."""
    with open(split_file) as f:
        split = json.load(f)

    train_key = 'train' if 'train' in split else 'training_set'
    val_key = 'val' if 'val' in split else 'validation_set'
    test_key = 'test' if 'test' in split else 'test_set'

    normalized_to_original = {normalize_model_name(n): n for n in graph_names.keys()}

    def match_names(split_names):
        matched = []
        for name in split_names:
            norm = normalize_model_name(name)
            if norm in normalized_to_original:
                matched.append(normalized_to_original[norm])
        return matched

    train_matched = match_names(split.get(train_key, []))
    val_matched = match_names(split.get(val_key, []))
    test_matched = match_names(split.get(test_key, []))

    return train_matched, val_matched, test_matched


# =============================================================================
# PLOTTING
# =============================================================================

def plot_training_curves(history: Dict, output_dir: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r--', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r--', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Graph Classification Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['val_f1'], 'g-', linewidth=2)
    best_f1 = max(history['val_f1'])
    best_epoch = history['val_f1'].index(best_f1) + 1
    axes[1, 0].axhline(y=best_f1, color='r', linestyle='--', alpha=0.7,
                       label=f'Best: {best_f1:.3f} @ epoch {best_epoch}')
    axes[1, 0].scatter([best_epoch], [best_f1], color='r', s=100, zorder=5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1')
    axes[1, 0].set_title('Validation Macro F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['lr'], 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()


def plot_embedding_tsne(model, loader, device, class_names, num_classes, epoch,
                        output_dir: Path, acc=None, f1=None, paper_mode=False):
    """Plot t-SNE of GNN graph embeddings at a given epoch."""
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('tab10')
    colors = [cmap(i) for i in range(num_classes)]

    embeds, labels = model.get_graph_embeddings(loader, device)
    embeds = np.nan_to_num(embeds, nan=0.0, posinf=0.0, neginf=0.0)

    X_scaled = StandardScaler().fit_transform(embeds)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, learning_rate='auto', init='pca')
    X_2d = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for i in range(num_classes):
        mask = labels == i
        if mask.sum() > 0:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[colors[i]], label=class_names[i],
                       s=18, alpha=0.7, edgecolors='none')
    ax.legend(fontsize=9, markerscale=2.5, loc='best',
              framealpha=0.9, edgecolor='none')
    if not paper_mode:
        title = f"GNN Embeddings - Epoch {epoch}"
        if acc is not None:
            title += f" (acc={acc:.1%}, f1={f1:.1%})"
        ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    dpi = 300 if paper_mode else 150
    fig.savefig(output_dir / f'embeddings_epoch_{epoch:03d}.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, num_classes, output_dir: Path, class_names: List[str]):
    """Plot confusion matrix."""
    present_classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    labels = [class_names[i] if i < len(class_names) else str(i) for i in present_classes]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=labels, yticklabels=labels)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Counts)')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=labels, yticklabels=labels)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close()

    return cm


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Train HeteroGATv2 for MechCAD Graph Classification")
    parser.add_argument("--graphs_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_file", type=str, default=None,
                        help="Optional split JSON. If not provided, generates stratified split.")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes (auto-detected if not set)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--pool_face_only", action="store_true",
                        help="Only pool face nodes (ignore coedge/vertex in readout)")

    args = parser.parse_args()

    # Reproducibility
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    graphs_dir = Path(args.graphs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load class map
    class_map_path = graphs_dir / 'class_map.json'
    if class_map_path.exists():
        with open(class_map_path) as f:
            class_map = json.load(f)
        class_names = sorted(class_map.keys(), key=lambda x: class_map[x])
        num_classes = len(class_map)
    else:
        class_names = None
        num_classes = None

    print(f"\n{'='*70}")
    print("TRAIN HETERO-GATv2 FOR MECHCAD GRAPH CLASSIFICATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model: GATv2 (dynamic attention) + Global Pooling")
    print(f"Hidden: {args.hidden_dim}, Heads: {args.num_heads}, Layers: {args.num_layers}")
    print(f"Effective batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"Label smoothing: {args.label_smoothing}")

    # Load graphs
    print("\nLoading graphs...")
    all_graphs = []
    graph_names = {}

    for pt_file in sorted(graphs_dir.glob("*.pt")):
        if pt_file.name in ['all_graphs.pt', 'graph_stats.json']:
            continue
        try:
            g = torch.load(pt_file, weights_only=False)
            model_name = pt_file.stem
            if model_name.endswith('_graph'):
                model_name = model_name[:-6]
            g.model_name = model_name
            all_graphs.append(g)
            graph_names[model_name] = g
        except:
            continue

    print(f"Loaded {len(all_graphs)} graphs")

    # Filter to labeled graphs only
    all_graphs = [g for g in all_graphs if has_labels(g)]
    graph_names = {g.model_name: g for g in all_graphs}
    print(f"With labels: {len(all_graphs)}")

    # Auto-detect num_classes
    if num_classes is None:
        all_labels = [g.y.item() for g in all_graphs]
        num_classes = max(all_labels) + 1
    if args.num_classes is not None:
        num_classes = args.num_classes

    if class_names is None:
        class_names = [f'class_{i}' for i in range(num_classes)]

    print(f"Classes ({num_classes}): {', '.join(class_names)}")

    # Print class distribution
    label_dist = Counter(g.y.item() for g in all_graphs)
    print(f"\nClass distribution:")
    for i, name in enumerate(class_names):
        print(f"  [{i}] {name}: {label_dist.get(i, 0)}")

    # Verify relation types
    if all_graphs:
        g0 = all_graphs[0]
        print(f"\nRelation types in graphs:")
        for et in g0.edge_types:
            n_edges = g0[et].edge_index.shape[1] if hasattr(g0[et], 'edge_index') else 0
            print(f"  {et}: {n_edges} edges")

    # Fix NaN/Inf
    fixed = fix_nan_inf_in_graphs(all_graphs)
    if fixed > 0:
        print(f"WARNING: Fixed NaN/Inf in {fixed} graphs")

    # Get dims
    detected_node_types = detect_node_types(all_graphs)
    middle_type = detected_node_types[1]
    face_dim, middle_dim, vertex_dim = get_feature_dims(all_graphs)
    print(f"Graph topology: {'/'.join(detected_node_types)}")
    print(f"Feature dims: Face={face_dim}, {middle_type.capitalize()}={middle_dim}, Vertex={vertex_dim}")

    # Split
    if args.split_file:
        print(f"\nLoading split from: {args.split_file}")
        train_names, val_names, test_names = load_split(Path(args.split_file), graph_names)
    else:
        print(f"\nGenerating stratified split (train={args.train_ratio}, val={args.val_ratio})")
        train_names, val_names, test_names = generate_stratified_split(
            all_graphs, args.train_ratio, args.val_ratio,
            1.0 - args.train_ratio - args.val_ratio, args.seed
        )
        # Save the generated split
        split_data = {'train': train_names, 'val': val_names, 'test': test_names}
        with open(output_dir / 'split.json', 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Split saved to {output_dir / 'split.json'}")

    train_graphs = [graph_names[n] for n in train_names if n in graph_names]
    val_graphs = [graph_names[n] for n in val_names if n in graph_names]
    test_graphs = [graph_names[n] for n in test_names if n in graph_names]

    print(f"Split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

    # Normalize features
    if not args.no_normalize:
        print("\nNormalizing features...")
        stats = compute_feature_statistics(train_graphs)
        train_graphs = normalize_graphs(train_graphs, stats)
        val_graphs = normalize_graphs(val_graphs, stats)
        test_graphs = normalize_graphs(test_graphs, stats)

        stats_np = {k: (v[0].tolist(), v[1].tolist()) for k, v in stats.items()}
        with open(output_dir / 'feature_stats.json', 'w') as f:
            json.dump(stats_np, f)
        print("Feature statistics saved")

    # DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    # Class weights
    class_weights = compute_class_weights(train_graphs, num_classes)
    print(f"Class weights: {class_weights.numpy().round(2)}")

    # Model
    # Collect edge types from first graph
    graph_edge_types = list(all_graphs[0].edge_types)
    print(f"Edge types ({len(graph_edge_types)}): {graph_edge_types}")

    model = HeteroGATv2GraphClassifier(
        face_dim=face_dim,
        middle_dim=middle_dim,
        vertex_dim=vertex_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_classes=num_classes,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        residual=True,
        pool_all_nodes=not args.pool_face_only,
        edge_types=graph_edge_types,
        node_types=detected_node_types,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # Training
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []}
    best_val_f1 = 0
    patience_counter = 0

    print(f"\n{'='*70}")
    print("Training...")
    print(f"{'='*70}\n")

    pbar = tqdm(range(args.epochs), desc="Training")

    for epoch in pbar:
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, num_classes,
            grad_accum_steps=args.grad_accum
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device, num_classes)

        lr = optimizer.param_groups[0]['lr']
        #scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(lr)

        pbar.set_postfix({
            'loss': f'{train_loss:.4f}',
            'acc': f'{val_acc:.3f}',
            'f1': f'{val_f1:.3f}',
            'best': f'{best_val_f1:.3f}'
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_f1': val_f1,
                'config': vars(args),
                'class_names': class_names,
                'num_classes': num_classes,
            }, output_dir / 'best_model.pt')
        else:
            patience_counter += 1

        # Embedding t-SNE at epoch 0, 1, every 10 epochs, and on new best
        embed_plot_dir = output_dir / 'embedding_progress'
        embed_plot_dir.mkdir(exist_ok=True)
        if epoch in (0, 1) or (epoch + 1) % 10 == 0 or (patience_counter == 0 and val_f1 > 0):
            try:
                plot_embedding_tsne(
                    model, val_loader, device, class_names, num_classes,
                    epoch, embed_plot_dir, acc=val_acc, f1=val_f1
                )
            except Exception as e:
                print(f"\n  [WARN] Embedding plot failed at epoch {epoch}: {e}")

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f)

    plot_training_curves(history, output_dir)

    # Load best and test
    ckpt = torch.load(output_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, num_classes
    )

    # Final t-SNE on test set with best model
    try:
        embed_plot_dir = output_dir / 'embedding_progress'
        embed_plot_dir.mkdir(exist_ok=True)
        plot_embedding_tsne(
            model, test_loader, device, class_names, num_classes,
            epoch=999, output_dir=embed_plot_dir, acc=test_acc, f1=test_f1
        )
        # Rename to something clear
        src = embed_plot_dir / 'embeddings_epoch_999.png'
        dst = output_dir / 'embeddings_test_final.png'
        if src.exists():
            src.rename(dst)
            print(f"Final embedding plot: {dst}")
    except Exception as e:
        print(f"  [WARN] Final embedding plot failed: {e}")

    print(f"\n{'='*70}")
    print("TEST RESULTS (Graph Classification)")
    print(f"{'='*70}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {test_f1:.4f}")

    if test_preds:
        print(f"\nClassification Report:")
        present = sorted(set(test_labels) | set(test_preds))
        names = [class_names[i] if i < len(class_names) else f'class_{i}' for i in present]
        print(classification_report(test_labels, test_preds, labels=present,
                                   target_names=names, zero_division=0))

        plot_confusion_matrix(test_labels, test_preds, num_classes, output_dir, class_names)

    # Per-class F1
    per_class_f1 = {}
    if test_preds:
        from sklearn.metrics import f1_score as f1_fn
        present = sorted(set(test_labels) | set(test_preds))
        f1_per = f1_fn(test_labels, test_preds, labels=present, average=None, zero_division=0)
        for idx, cls_idx in enumerate(present):
            name = class_names[cls_idx] if cls_idx < len(class_names) else f'class_{cls_idx}'
            per_class_f1[name] = float(f1_per[idx])

    # Save results
    results = {
        'model_type': 'HeteroGATv2_MechCAD_GraphClassification',
        'task': 'classification',
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'best_val_f1': float(best_val_f1),
        'num_classes': num_classes,
        'class_names': class_names,
        'per_class_f1': per_class_f1,
        'config': vars(args),
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

# With external split:
# python train_gatv2_mechcad.py --graphs_dir /home/konstantinos/Downloads/mechcad/output/graphs_coedge --output_dir ./mechcad/models_gatv2 --split_file ./mechcad_split.json --device cuda
