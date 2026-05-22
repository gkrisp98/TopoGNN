"""
Generate Stratified Train/Val/Test Split for MechCAD
=====================================================

Scans /mechcad/<class>/*.step folder structure, generates a stratified
split, and saves it as JSON. Use this ONCE before the entire pipeline
so the same split is used for both encoder training and GAT training.

Model names are prefixed with class: "bearing__part001" to avoid
collisions when different classes have files with the same stem.

Usage:
    python generate_mechcad_split.py \
        --data_dir /path/to/mechcad \
        --output mechcad_split.json \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --seed 42
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit


def make_unique_name(class_idx: int, stem: str) -> str:
    """Create a unique model name: classIdx__stem (class-blind)."""
    return f"{class_idx}__{stem}"


def main():
    parser = argparse.ArgumentParser(description="Generate stratified split for MechCAD")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root dir with class subfolders (e.g. /mechcad)")
    parser.add_argument("--output", type=str, default="mechcad_split.json",
                        help="Output JSON path")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    # Discover classes
    class_dirs = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    class_map = {name: idx for idx, name in enumerate(class_dirs)}

    # Collect all STEP files with unique names: class__stem
    names = []
    labels = []
    model_classes = {}

    for class_name in class_dirs:
        class_dir = data_dir / class_name
        for ext in ['*.stp', '*.step', '*.STEP', '*.STP']:
            for f in class_dir.glob(ext):
                unique_name = make_unique_name(class_map[class_name], f.stem)
                names.append(unique_name)
                labels.append(class_map[class_name])
                model_classes[unique_name] = class_name

    # Verify uniqueness
    assert len(names) == len(set(names)), \
        f"Still have duplicates! {len(names)} names but {len(set(names))} unique"

    print(f"Found {len(names)} models across {len(class_map)} classes:")
    class_counts = defaultdict(int)
    for l in labels:
        class_counts[l] += 1
    for name, idx in class_map.items():
        print(f"  [{idx}] {name}: {class_counts[idx]}")

    # Check minimum class size
    min_count = min(class_counts.values())
    if min_count < 3:
        print(f"\nWARNING: Smallest class has only {min_count} samples.")

    # Split: train+val vs test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=args.seed)
    trainval_idx, test_idx = next(sss1.split(names, labels))

    # Split: train vs val
    trainval_labels = [labels[i] for i in trainval_idx]
    relative_val = args.val_ratio / (args.train_ratio + args.val_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=args.seed)
    train_sub_idx, val_sub_idx = next(sss2.split(trainval_idx, trainval_labels))

    train_names = sorted([names[trainval_idx[i]] for i in train_sub_idx])
    val_names = sorted([names[trainval_idx[i]] for i in val_sub_idx])
    test_names = sorted([names[test_idx[i]] for i in range(len(test_idx))])

    # Verify no overlap
    assert len(set(train_names) & set(val_names)) == 0, "Train/val overlap!"
    assert len(set(train_names) & set(test_names)) == 0, "Train/test overlap!"
    assert len(set(val_names) & set(test_names)) == 0, "Val/test overlap!"
    assert len(train_names) + len(val_names) + len(test_names) == len(names), "Missing models!"

    split = {
        "train": train_names,
        "val": val_names,
        "test": test_names,
        "class_map": class_map,
        "model_classes": model_classes,
    }

    with open(args.output, 'w') as f:
        json.dump(split, f, indent=2)

    print(f"\nSplit ({args.train_ratio}/{args.val_ratio}/{test_ratio:.2f}):")
    print(f"  Train: {len(train_names)}")
    print(f"  Val:   {len(val_names)}")
    print(f"  Test:  {len(test_names)}")
    print(f"\nNaming convention: classIdx__stem (e.g. '{train_names[0]}')")
    print(f"  Class index is opaque — label info is NOT in the filename")
    print(f"Saved to: {args.output}")

    # Per-class breakdown
    for subset_name, subset in [("Train", train_names), ("Val", val_names), ("Test", test_names)]:
        dist = defaultdict(int)
        for n in subset:
            dist[model_classes[n]] += 1
        print(f"\n  {subset_name} class distribution:")
        for cls_name in class_dirs:
            print(f"    {cls_name}: {dist[cls_name]}")


if __name__ == "__main__":
    main()

# Usage:
# python generate_mechcad_split.py --data_dir /home/konstantinos/Downloads/mechcad/data --output mechcad_split.json --seed 42