"""
Multi-seed experiment runner for statistically significant results.
===================================================================

Runs GATv2 training N times with different random seeds (same data split),
collects test accuracy/F1, and reports mean ± std suitable for journal tables.

Usage:
    python run_experiment.py \
        --graphs_dir /home/konstantinos/Downloads/mechcad/output/graphs_coedge \
        --output_dir ./mechcad/experiment \
        --split_file ./mechcad_split.json \
        --n_runs 5 --device cuda
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import from training script
from train_gatv2_mechcad import (
    HeteroGATv2GraphClassifier,
    FocalLoss,
    has_labels,
    get_feature_dims,
    detect_node_types,
    fix_nan_inf_in_graphs,
    compute_feature_statistics,
    normalize_graphs,
    compute_class_weights,
    train_epoch,
    evaluate,
    load_split,
    plot_confusion_matrix,
)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def single_run(run_id: int, seed: int, train_graphs, val_graphs, test_graphs,
               face_dim, middle_dim, vertex_dim, num_classes, class_names,
               args, device, run_dir: Path,
               edge_types=None, node_types=None) -> Dict:
    """Execute a single training run with given seed. Returns metrics dict."""

    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Normalize features (fit on train only)
    stats = compute_feature_statistics(train_graphs)
    train_norm = normalize_graphs([g.clone() for g in train_graphs], stats)
    val_norm = normalize_graphs([g.clone() for g in val_graphs], stats)
    test_norm = normalize_graphs([g.clone() for g in test_graphs], stats)

    train_loader = DataLoader(train_norm, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_norm, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_norm, batch_size=args.batch_size, shuffle=False)

    class_weights = compute_class_weights(train_norm, num_classes)

    # Fresh model each run
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
        pool_all_nodes=True,
        edge_types=edge_types,
        node_types=node_types,
    ).to(device)

    criterion = FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_acc': [], 'val_f1': []}

    pbar = tqdm(range(args.epochs), desc=f"Run {run_id+1} (seed={seed})", leave=False)

    for epoch in pbar:
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, num_classes,
            grad_accum_steps=args.grad_accum
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device, num_classes)

        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        pbar.set_postfix({'acc': f'{val_acc:.3f}', 'f1': f'{val_f1:.3f}', 'best': f'{best_val_f1:.3f}'})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_f1': val_f1,
                'seed': seed,
            }, run_dir / 'best_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    pbar.close()

    # Load best model and evaluate on test
    ckpt = torch.load(run_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, num_classes
    )

    # Per-class F1
    per_class_f1 = {}
    if test_preds:
        from sklearn.metrics import f1_score as f1_fn
        present = sorted(set(test_labels) | set(test_preds))
        f1_per = f1_fn(test_labels, test_preds, labels=present, average=None, zero_division=0)
        for idx, cls_idx in enumerate(present):
            name = class_names[cls_idx] if cls_idx < len(class_names) else f'class_{cls_idx}'
            per_class_f1[name] = float(f1_per[idx])

    result = {
        'run_id': run_id,
        'seed': seed,
        'test_accuracy': float(test_acc),
        'test_f1_macro': float(test_f1),
        'best_val_f1': float(best_val_f1),
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'per_class_f1': per_class_f1,
    }

    # Save per-run results
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Multi-seed experiment for MechCAD GATv2")
    parser.add_argument("--graphs_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./mechcad/experiment")
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--n_runs", type=int, default=5, help="Number of runs (default: 5)")
    parser.add_argument("--seeds", type=int, nargs='+', default=None,
                        help="Explicit seeds (overrides n_runs). E.g. --seeds 42 123 456 789 1024")

    # Model hyperparams (same defaults as train_gatv2_mechcad.py)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
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
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Seeds
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = [42, 123, 456, 789, 1024][:args.n_runs]

    n_runs = len(seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = Path(args.graphs_dir)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"MULTI-SEED EXPERIMENT ({n_runs} runs)")
    print(f"{'='*70}")
    print(f"Seeds: {seeds}")
    print(f"Device: {device}")
    print(f"Model: GATv2 h={args.hidden_dim} heads={args.num_heads} layers={args.num_layers}")
    print(f"Split: {args.split_file}")

    # Load class map
    class_map_path = graphs_dir / 'class_map.json'
    with open(class_map_path) as f:
        class_map = json.load(f)
    class_names = sorted(class_map.keys(), key=lambda x: class_map[x])
    num_classes = len(class_map)

    # Load all graphs once
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

    all_graphs = [g for g in all_graphs if has_labels(g)]
    graph_names = {g.model_name: g for g in all_graphs}
    print(f"Loaded {len(all_graphs)} labeled graphs, {num_classes} classes")

    # Fix NaN/Inf
    fix_nan_inf_in_graphs(all_graphs)

    # Get dims
    detected_node_types = detect_node_types(all_graphs)
    middle_type = detected_node_types[1]
    face_dim, middle_dim, vertex_dim = get_feature_dims(all_graphs)
    graph_edge_types = list(all_graphs[0].edge_types)
    print(f"Graph topology: {'/'.join(detected_node_types)}")
    print(f"Feature dims: face={face_dim}, {middle_type}={middle_dim}, vertex={vertex_dim}")

    # Load split (same for all runs)
    train_names, val_names, test_names = load_split(Path(args.split_file), graph_names)
    train_graphs = [graph_names[n] for n in train_names if n in graph_names]
    val_graphs = [graph_names[n] for n in val_names if n in graph_names]
    test_graphs = [graph_names[n] for n in test_names if n in graph_names]
    print(f"Split: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}")

    # Class distribution
    label_dist = Counter(g.y.item() for g in train_graphs)
    print(f"\nTrain class distribution:")
    for i, name in enumerate(class_names):
        print(f"  [{i}] {name}: {label_dist.get(i, 0)}")

    # Run experiments
    print(f"\n{'='*70}")
    all_results = []
    t0 = time.time()

    for run_id, seed in enumerate(seeds):
        print(f"\n--- Run {run_id+1}/{n_runs} (seed={seed}) ---")
        run_dir = output_dir / f'run_{run_id}_seed{seed}'

        result = single_run(
            run_id, seed, train_graphs, val_graphs, test_graphs,
            face_dim, middle_dim, vertex_dim, num_classes, class_names,
            args, device, run_dir,
            edge_types=graph_edge_types, node_types=detected_node_types
        )
        all_results.append(result)
        print(f"  Test acc={result['test_accuracy']:.4f}, F1={result['test_f1_macro']:.4f} "
              f"(best_epoch={result['best_epoch']}, epochs={result['total_epochs']})")

    elapsed = time.time() - t0

    # Aggregate results
    accs = [r['test_accuracy'] for r in all_results]
    f1s = [r['test_f1_macro'] for r in all_results]

    acc_mean, acc_std = np.mean(accs), np.std(accs)
    f1_mean, f1_std = np.mean(f1s), np.std(f1s)

    # Per-class aggregation
    per_class_means = {}
    per_class_stds = {}
    for name in class_names:
        vals = [r['per_class_f1'].get(name, 0.0) for r in all_results]
        per_class_means[name] = float(np.mean(vals))
        per_class_stds[name] = float(np.std(vals))

    print(f"\n{'='*70}")
    print(f"EXPERIMENT RESULTS ({n_runs} runs)")
    print(f"{'='*70}")
    print(f"Test Accuracy: {100*acc_mean:.2f} ± {100*acc_std:.2f}%")
    print(f"Test Macro F1: {100*f1_mean:.2f} ± {100*f1_std:.2f}%")
    print(f"\nPer-run results:")
    for r in all_results:
        print(f"  Seed {r['seed']:>5d}: acc={100*r['test_accuracy']:.2f}%, "
              f"F1={100*r['test_f1_macro']:.2f}%, best_epoch={r['best_epoch']}")
    print(f"\nPer-class F1 (mean ± std):")
    for name in class_names:
        print(f"  {name:>10s}: {100*per_class_means[name]:.2f} ± {100*per_class_stds[name]:.2f}%")
    print(f"\nTotal time: {elapsed/60:.1f} min ({elapsed/n_runs/60:.1f} min/run)")

    # Save aggregate results
    summary = {
        'n_runs': n_runs,
        'seeds': seeds,
        'test_accuracy_mean': float(acc_mean),
        'test_accuracy_std': float(acc_std),
        'test_f1_macro_mean': float(f1_mean),
        'test_f1_macro_std': float(f1_std),
        'per_class_f1_mean': per_class_means,
        'per_class_f1_std': per_class_stds,
        'per_run': all_results,
        'config': vars(args),
        'elapsed_seconds': elapsed,
        'latex_accuracy': f"${100*acc_mean:.2f} \\pm {100*acc_std:.2f}$",
        'latex_f1': f"${100*f1_mean:.2f} \\pm {100*f1_std:.2f}$",
    }
    with open(output_dir / 'experiment_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_dir / 'experiment_results.json'}")

    # Generate summary plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar plot of per-run accuracy
    x = range(n_runs)
    axes[0].bar(x, [100*a for a in accs], color='steelblue', alpha=0.8)
    axes[0].axhline(y=100*acc_mean, color='red', linestyle='--', label=f'Mean: {100*acc_mean:.2f}%')
    axes[0].fill_between([-0.5, n_runs-0.5], 100*(acc_mean-acc_std), 100*(acc_mean+acc_std),
                         alpha=0.15, color='red')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f's={s}' for s in seeds], fontsize=8)
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title(f'Accuracy: {100*acc_mean:.2f} ± {100*acc_std:.2f}%')
    axes[0].legend(fontsize=8)

    # Per-class F1 with error bars
    y_pos = range(num_classes)
    means = [100*per_class_means[n] for n in class_names]
    stds = [100*per_class_stds[n] for n in class_names]
    axes[1].barh(y_pos, means, xerr=stds, color='steelblue', alpha=0.8, capsize=3)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(class_names, fontsize=8)
    axes[1].set_xlabel('F1 Score (%)')
    axes[1].set_title(f'Per-class F1 (macro: {100*f1_mean:.2f} ± {100*f1_std:.2f}%)')

    plt.tight_layout()
    plt.savefig(output_dir / 'experiment_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary plot: {output_dir / 'experiment_summary.png'}")

    # Print LaTeX-ready table row
    print(f"\n--- LaTeX table row ---")
    print(f"Ours (HeteroGATv2) & ${100*acc_mean:.2f} \\pm {100*acc_std:.2f}$ & "
          f"${100*f1_mean:.2f} \\pm {100*f1_std:.2f}$ \\\\")


if __name__ == "__main__":
    main()

# Usage:
# python run_experiment.py \
#     --graphs_dir /home/konstantinos/Downloads/mechcad/output/graphs_coedge \
#     --output_dir ./mechcad/experiment \
#     --split_file ./mechcad_split.json \
#     --n_runs 5 --device cuda
