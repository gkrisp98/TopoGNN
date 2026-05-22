"""
Train Heterogeneous GNN (GATv2) for Fusion 360 Dataset
======================================================

IMPROVEMENTS over previous version:
1. GATv2Conv instead of GATConv (dynamic attention - much better!)
2. Feature normalization (crucial for neural networks)
3. Better architecture with skip connections
4. Gradient accumulation for effective larger batch sizes
5. Label smoothing
6. Mixup augmentation option

GATv2 vs GAT:
- GAT: attention = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))  <- STATIC
- GATv2: attention = softmax(a^T LeakyReLU(W [h_i || h_j])) <- DYNAMIC
GATv2 can learn more expressive attention patterns.

Relations (from build_fusion360_graphs_v3.py):
- ('face', 'uses_fwd', 'edge')      / ('edge', 'used_by_fwd', 'face')
- ('face', 'uses_rev', 'edge')      / ('edge', 'used_by_rev', 'face')
- ('edge', 'next_in_loop', 'edge')  — sequential curves in boundary loop
- ('edge', 'has', 'vertex')         / ('vertex', 'belongs_to', 'edge')
- ('face', 'adjacent_to', 'face')   — shared-edge adjacency

Author: Konstantinos (TUM)
"""

import numpy as np
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear, LayerNorm
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_CLASSES = 8

CLASS_NAMES = [
    'ExtrudeSide', 'ExtrudeEnd', 'CutSide', 'CutEnd',
    'Fillet', 'Chamfer', 'RevolveSide', 'RevolveEnd'
]


# =============================================================================
# FEATURE NORMALIZATION
# =============================================================================

def compute_feature_statistics(graphs: List[HeteroData]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute mean and std for each node type's features."""
    features = {'face': [], 'edge': [], 'vertex': []}
    
    for g in graphs:
        for node_type in ['face', 'edge', 'vertex']:
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
        for node_type in ['face', 'edge', 'vertex']:
            if node_type in g.node_types and g[node_type].x.shape[0] > 0:
                mean, std = stats[node_type]
                mean_t = torch.from_numpy(mean).to(g[node_type].x.device)
                std_t = torch.from_numpy(std).to(g[node_type].x.device)
                g[node_type].x = (g[node_type].x - mean_t) / std_t
    return graphs


def fix_nan_inf_in_graphs(graphs: List[HeteroData]) -> int:
    """Fix NaN/Inf values in graph features. Returns count of fixed graphs."""
    fixed_count = 0
    for g in graphs:
        has_issue = False
        for node_type in ['face', 'edge', 'vertex']:
            if node_type in g.node_types and g[node_type].x.shape[0] > 0:
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
# GATv2 MODEL — MATCHED TO build_fusion360_graphs_v3.py RELATIONS
# =============================================================================

class HeteroGATv2(nn.Module):
    """
    Heterogeneous Graph Attention Network v2 for Fusion 360 B-rep graphs.
    
    Relations (must match build_fusion360_graphs_v3.py):
      face <-> edge:   uses_fwd / used_by_fwd, uses_rev / used_by_rev
      edge <-> edge:   next_in_loop (boundary curve sequencing)
      edge <-> vertex: has / belongs_to (canonical endpoints)
      face <-> face:   adjacent_to (shared-edge adjacency)
    
    Architecture:
    - GATv2Conv with dynamic attention
    - Pre-layer normalization (transformer-style)
    - Residual connections
    - Deep MLP classifier
    """
    
    def __init__(
        self,
        face_dim: int,
        edge_dim: int,
        vertex_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.residual = residual
        
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        
        # Input projections with layer norm
        self.face_input = nn.Sequential(
            Linear(face_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.edge_input = nn.Sequential(
            Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vertex_input = nn.Sequential(
            Linear(vertex_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # GATv2 layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            conv = HeteroConv({
                # ---- Face <-> Face: shared-edge adjacency ----
                ('face', 'adjacent_to', 'face'): GATv2Conv(
                    hidden_dim, self.head_dim, heads=num_heads,
                    dropout=attention_dropout, add_self_loops=False, concat=True,
                    share_weights=False,
                ),
                # ---- Face <-> Edge: directed by curve orientation ----
                ('face', 'uses_fwd', 'edge'): GATv2Conv(
                    (hidden_dim, hidden_dim), self.head_dim, heads=num_heads,
                    dropout=attention_dropout, add_self_loops=False, concat=True,
                ),
                ('edge', 'used_by_fwd', 'face'): GATv2Conv(
                    (hidden_dim, hidden_dim), self.head_dim, heads=num_heads,
                    dropout=attention_dropout, add_self_loops=False, concat=True,
                ),
                ('face', 'uses_rev', 'edge'): GATv2Conv(
                    (hidden_dim, hidden_dim), self.head_dim, heads=num_heads,
                    dropout=attention_dropout, add_self_loops=False, concat=True,
                ),
                ('edge', 'used_by_rev', 'face'): GATv2Conv(
                    (hidden_dim, hidden_dim), self.head_dim, heads=num_heads,
                    dropout=attention_dropout, add_self_loops=False, concat=True,
                ),
                # ---- Edge <-> Edge: boundary loop sequencing ----
                ('edge', 'next_in_loop', 'edge'): GATv2Conv(
                    hidden_dim, self.head_dim, heads=num_heads,
                    dropout=attention_dropout, add_self_loops=False, concat=True,
                    share_weights=False,
                ),
                # ---- Edge <-> Vertex: canonical endpoints ----
                ('edge', 'has', 'vertex'): GATv2Conv(
                    (hidden_dim, hidden_dim), self.head_dim, heads=num_heads,
                    dropout=attention_dropout, add_self_loops=False, concat=True,
                ),
                ('vertex', 'belongs_to', 'edge'): GATv2Conv(
                    (hidden_dim, hidden_dim), self.head_dim, heads=num_heads,
                    dropout=attention_dropout, add_self_loops=False, concat=True,
                ),
            }, aggr='mean')
            
            self.convs.append(conv)
            
            # Pre-norm style (like in transformers)
            self.norms.append(nn.ModuleDict({
                'face': nn.LayerNorm(hidden_dim),
                'edge': nn.LayerNorm(hidden_dim),
                'vertex': nn.LayerNorm(hidden_dim),
            }))
        
        self.dropout = nn.Dropout(dropout)
        
        # Deeper classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
        x_dict = {
            'face': self.face_input(data['face'].x),
        }
        
        if 'edge' in data.node_types and data['edge'].x.shape[0] > 0:
            x_dict['edge'] = self.edge_input(data['edge'].x)
        else:
            x_dict['edge'] = torch.zeros((0, self.hidden_dim), device=device)
        
        if 'vertex' in data.node_types and data['vertex'].x.shape[0] > 0:
            x_dict['vertex'] = self.vertex_input(data['vertex'].x)
        else:
            x_dict['vertex'] = torch.zeros((0, self.hidden_dim), device=device)
        
        # GATv2 message passing
        for conv, norm in zip(self.convs, self.norms):
            # Store for residual
            x_dict_residual = {k: v.clone() for k, v in x_dict.items() if v.shape[0] > 0}
            
            # Pre-norm
            x_dict_normed = {}
            for key in x_dict:
                if x_dict[key].shape[0] > 0:
                    x_dict_normed[key] = norm[key](x_dict[key])
                else:
                    x_dict_normed[key] = x_dict[key]
            
            # Only pass edge types that exist in this graph
            edge_index_dict = {et: data.edge_index_dict[et] for et in data.edge_types 
                              if et in data.edge_index_dict}
            
            x_dict_new = conv(x_dict_normed, edge_index_dict)
            
            # Residual + activation
            for key in x_dict:
                if key in x_dict_new and x_dict_new[key].shape[0] > 0:
                    if self.residual and key in x_dict_residual:
                        x_dict[key] = x_dict_residual[key] + self.dropout(x_dict_new[key])
                    else:
                        x_dict[key] = self.dropout(F.gelu(x_dict_new[key]))
        
        return self.classifier(x_dict['face'])


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def has_labels(graph: HeteroData) -> bool:
    """Check if graph has valid labels."""
    if not hasattr(graph['face'], 'y') or graph['face'].y is None:
        return False
    return (graph['face'].y >= 0).any().item()


def get_feature_dims(graphs: List[HeteroData]) -> Tuple[int, int, int]:
    """Get feature dimensions from graphs."""
    for g in graphs:
        face_dim = g['face'].x.shape[1]
        edge_dim = g['edge'].x.shape[1] if 'edge' in g.node_types and g['edge'].x.shape[0] > 0 else 64
        vertex_dim = g['vertex'].x.shape[1] if 'vertex' in g.node_types and g['vertex'].x.shape[0] > 0 else 8
        return face_dim, edge_dim, vertex_dim
    return 64, 64, 8


def compute_class_weights(graphs: List[HeteroData], num_classes: int) -> torch.Tensor:
    """Compute inverse frequency class weights."""
    all_labels = []
    for g in graphs:
        if has_labels(g):
            labels = g['face'].y.numpy()
            all_labels.extend(labels[(labels >= 0) & (labels < num_classes)])
    
    if not all_labels:
        return torch.ones(num_classes)
    
    counts = Counter(all_labels)
    total = sum(counts.values())
    
    weights = torch.ones(num_classes)
    for c, count in counts.items():
        if c < num_classes:
            weights[c] = total / (count * num_classes)
    
    # Cap extreme weights
    weights = torch.clamp(weights, min=0.1, max=50.0)
    
    return weights


def train_epoch(model, loader, optimizer, criterion, device, num_classes, 
                grad_accum_steps: int = 1, max_grad_norm: float = 1.0):
    """Train for one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    optimizer.zero_grad()
    
    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        
        out = model(data)
        labels = data['face'].y
        
        mask = (labels >= 0) & (labels < num_classes)
        if mask.sum() == 0:
            continue
        
        loss = criterion(out[mask], labels[mask])
        
        if torch.isnan(loss):
            continue
        
        # Scale loss for gradient accumulation
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
    """Evaluate model."""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            labels = data['face'].y
            
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
        return 0.0, 0.0, 0.0, 0.0, [], []

    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels)

    accuracy = np.mean(all_preds_np == all_labels_np)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Mean IoU: per-class intersection / union, then average
    present_classes = sorted(set(all_labels) | set(all_preds))
    ious = []
    for c in present_classes:
        pred_c = all_preds_np == c
        true_c = all_labels_np == c
        intersection = (pred_c & true_c).sum()
        union = (pred_c | true_c).sum()
        if union > 0:
            ious.append(intersection / union)
    mean_iou = np.mean(ious) if ious else 0.0

    return total_loss / max(total_samples, 1), accuracy, f1_macro, mean_iou, all_preds, all_labels


# =============================================================================
# SPLIT LOADING
# =============================================================================

def normalize_model_name(name: str) -> str:
    if name.endswith('_graph'):
        name = name[:-6]
    return name.lower()


def load_split(split_file: Path, graph_names: Dict[str, HeteroData]) -> Tuple[List, List, List]:
    """Load train/val/test split."""
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
    
    print(f"\nSplit: Train={len(train_matched)}, Val={len(val_matched)}, Test={len(test_matched)}")
    
    return train_matched, val_matched, test_matched


# =============================================================================
# PLOTTING
# =============================================================================

def plot_training_curves(history: Dict, output_dir: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
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
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, history['lr'], 'purple', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('LR')
    axes[0, 2].set_title('Learning Rate')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)

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

    if 'val_iou' in history and history['val_iou']:
        axes[1, 1].plot(epochs, history['val_iou'], '#E8833A', linewidth=2)
        best_iou = max(history['val_iou'])
        best_iou_epoch = history['val_iou'].index(best_iou) + 1
        axes[1, 1].axhline(y=best_iou, color='r', linestyle='--', alpha=0.7,
                           label=f'Best: {best_iou:.3f} @ epoch {best_iou_epoch}')
        axes[1, 1].scatter([best_iou_epoch], [best_iou], color='r', s=100, zorder=5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Mean IoU')
        axes[1, 1].set_title('Validation Mean IoU')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].set_visible(False)

    axes[1, 2].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()


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
    
    parser = argparse.ArgumentParser(description="Train HeteroGATv2 for Fusion 360")
    parser.add_argument("--graphs_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_normalize", action="store_true",
                        help="Skip feature normalization")
    
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
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    
    graphs_dir = Path(args.graphs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print("TRAIN HETERO-GATv2 FOR FUSION 360")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model: GATv2 (dynamic attention)")
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
    
    # Verify relation types from first graph
    if all_graphs:
        g0 = all_graphs[0]
        print(f"\nRelation types in graphs:")
        for et in g0.edge_types:
            n_edges = g0[et].edge_index.shape[1] if hasattr(g0[et], 'edge_index') else 0
            print(f"  {et}: {n_edges} edges")
    
    # Fix NaN/Inf BEFORE normalization
    fixed = fix_nan_inf_in_graphs(all_graphs)
    if fixed > 0:
        print(f"WARNING: Fixed NaN/Inf in {fixed} graphs")
    
    # Get dims
    face_dim, edge_dim, vertex_dim = get_feature_dims(all_graphs)
    print(f"Feature dims: Face={face_dim}, Edge={edge_dim}, Vertex={vertex_dim}")
    
    # Load split
    train_names, val_names, test_names = load_split(Path(args.split_file), graph_names)
    
    train_graphs = [graph_names[n] for n in train_names]
    val_graphs = [graph_names[n] for n in val_names]
    test_graphs = [graph_names[n] for n in test_names]
    
    print(f"Split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")
    
    # Normalize features
    if not args.no_normalize:
        print("\nNormalizing features...")
        stats = compute_feature_statistics(train_graphs)  # Compute on TRAIN only!
        train_graphs = normalize_graphs(train_graphs, stats)
        val_graphs = normalize_graphs(val_graphs, stats)
        test_graphs = normalize_graphs(test_graphs, stats)
        
        stats_np = {k: (v[0].tolist(), v[1].tolist()) for k, v in stats.items()}
        with open(output_dir / 'feature_stats.json', 'w') as f:
            json.dump(stats_np, f)
        print("Feature statistics saved")
    
    # Filter labeled
    train_graphs = [g for g in train_graphs if has_labels(g)]
    val_graphs = [g for g in val_graphs if has_labels(g)]
    test_graphs = [g for g in test_graphs if has_labels(g)]
    
    print(f"Labeled: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")
    
    # DataLoaders (seeded generator for reproducible shuffling)
    g_train = torch.Generator()
    g_train.manual_seed(args.seed)
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=g_train)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    
    # Class weights
    class_weights = compute_class_weights(train_graphs, args.num_classes)
    print(f"Class weights: {class_weights.numpy().round(2)}")
    
    # Model
    model = HeteroGATv2(
        face_dim=face_dim,
        edge_dim=edge_dim,
        vertex_dim=vertex_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_classes=args.num_classes,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        residual=True,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=20, T_mult=2, eta_min=1e-6
    # )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.0, last_epoch=-1)
    
    # Training
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_iou': [], 'lr': []}
    best_val_f1 = 0
    patience_counter = 0
    
    print(f"\n{'='*70}")
    print("Training...")
    print(f"{'='*70}\n")
    
    pbar = tqdm(range(args.epochs), desc="Training")
    
    for epoch in pbar:
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, args.num_classes,
            grad_accum_steps=args.grad_accum
        )
        val_loss, val_acc, val_f1, val_iou, _, _ = evaluate(model, val_loader, criterion, device, args.num_classes)
        
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_iou'].append(val_iou)
        history['lr'].append(lr)

        pbar.set_postfix({
            'loss': f'{train_loss:.4f}',
            'acc': f'{val_acc:.3f}',
            'f1': f'{val_f1:.3f}',
            'iou': f'{val_iou:.3f}',
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
            }, output_dir / 'best_model.pt')
        else:
            patience_counter += 1
        
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
    
    test_loss, test_acc, test_f1, test_iou, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, args.num_classes
    )

    print(f"\n{'='*70}")
    print("TEST RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Macro F1:  {test_f1:.4f}")
    print(f"Mean IoU:  {test_iou:.4f}")
    
    if test_preds:
        print(f"\nClassification Report:")
        present = sorted(set(test_labels) | set(test_preds))
        names = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}' for i in present]
        print(classification_report(test_labels, test_preds, labels=present, 
                                   target_names=names, zero_division=0))
        
        plot_confusion_matrix(test_labels, test_preds, args.num_classes, output_dir, CLASS_NAMES)
    
    # Save results
    results = {
        'model_type': 'HeteroGATv2_Fusion360',
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_iou': float(test_iou),
        'best_val_f1': float(best_val_f1),
        'config': vars(args),
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

# python train_gatv2_fusion360.py --graphs_dir /home/konstantinos/Downloads/s2.0.0/breps/graphs --output_dir ./models_gatv2 --split_file ./train_test.json --hidden_dim 256 --num_heads 8 --num_layers 4 --batch_size 32 --grad_accum 2 --epochs 200

#python train_gatv2_fusion360.py --graphs_dir /home/konstantinos/Downloads/s2.0.0/breps/graphs_neuronurbs --output_dir ./models_gatv2_neuronurbs --split_file ./train_test.json --hidden_dim 256 --num_heads 8 --num_layers 4 --batch_size 32 --grad_accum 2 --epochs 200

#python train_gatv2_fusion360.py --graphs_dir /home/konstantinos/Downloads/s2.0.0/breps/graphs_separate --output_dir ./models_gatv2_separate --split_file ./train_test.json --hidden_dim 256 --num_heads 8 --num_layers 4 --batch_size 32 --grad_accum 2 --epochs 200