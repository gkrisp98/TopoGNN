"""
Merge Embeddings from Separate VAE Training
=============================================

Combines per-model face, edge, and vertex embeddings (from the three
separate VAE scripts) into a single per-model embedding pickle file
compatible with build_fusion360_graphs.py.

Usage:
    python merge_embeddings.py \\
        --face_dir ./output_face_vae/embeddings \\
        --edge_dir ./output_edge_vae/embeddings \\
        --vertex_dir ./output_vertex_vae/embeddings \\
        --output_dir ./output_merged/embeddings
"""

import pickle
from pathlib import Path
from tqdm import tqdm

import numpy as np


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge separate VAE embeddings")
    parser.add_argument("--face_dir", type=str, required=True,
                        help="Dir with *_face_embeddings.pkl")
    parser.add_argument("--edge_dir", type=str, required=True,
                        help="Dir with *_edge_embeddings.pkl")
    parser.add_argument("--vertex_dir", type=str, required=True,
                        help="Dir with *_vertex_embeddings.pkl")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output dir for merged *_embeddings.pkl")

    args = parser.parse_args()

    face_dir = Path(args.face_dir)
    edge_dir = Path(args.edge_dir)
    vertex_dir = Path(args.vertex_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all model names from face embeddings
    face_files = sorted(face_dir.glob("*_face_embeddings.pkl"))
    model_names = [f.stem.replace("_face_embeddings", "") for f in face_files]

    print(f"Found {len(model_names)} models with face embeddings")
    print(f"Face dir:   {face_dir}")
    print(f"Edge dir:   {edge_dir}")
    print(f"Vertex dir: {vertex_dir}")
    print(f"Output dir: {output_dir}\n")

    merged = 0
    missing_edge = 0
    missing_vertex = 0

    for model_name in tqdm(model_names, desc="Merging"):
        face_file = face_dir / f"{model_name}_face_embeddings.pkl"
        edge_file = edge_dir / f"{model_name}_edge_embeddings.pkl"
        vertex_file = vertex_dir / f"{model_name}_vertex_embeddings.pkl"

        # Load face (required)
        with open(face_file, "rb") as f:
            face_data = pickle.load(f)

        face_embeddings = face_data["face_embeddings"]
        face_indices = face_data.get("face_indices", None)
        embed_dim = face_embeddings.shape[1] if len(face_embeddings) > 0 else 64

        # Load edge (optional)
        if edge_file.exists():
            with open(edge_file, "rb") as f:
                edge_data = pickle.load(f)
            edge_embeddings = edge_data["edge_embeddings"]
        else:
            edge_embeddings = np.zeros((0, embed_dim), dtype=np.float32)
            missing_edge += 1

        # Load vertex (optional)
        if vertex_file.exists():
            with open(vertex_file, "rb") as f:
                vertex_data = pickle.load(f)
            vertex_embeddings = vertex_data["vertex_embeddings"]
        else:
            vertex_embeddings = np.zeros((0, embed_dim), dtype=np.float32)
            missing_vertex += 1

        # Merge into format expected by build_fusion360_graphs.py
        out_data = {
            "face_embeddings": face_embeddings,
            "edge_embeddings": edge_embeddings,
            "vertex_embeddings": vertex_embeddings,
            "model_name": model_name,
            "n_faces": len(face_embeddings),
            "n_edges": len(edge_embeddings),
            "n_vertices": len(vertex_embeddings),
            "face_indices": face_indices,
        }

        out_file = output_dir / f"{model_name}_embeddings.pkl"
        with open(out_file, "wb") as f:
            pickle.dump(out_data, f)

        merged += 1

    print(f"\nMerged {merged} models")
    if missing_edge > 0:
        print(f"  WARNING: {missing_edge} models missing edge embeddings")
    if missing_vertex > 0:
        print(f"  WARNING: {missing_vertex} models missing vertex embeddings")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

#python merge_embeddings.py --face_dir /home/konstantinos/Downloads/s2.0.0/breps/embeddings --edge_dir /home/konstantinos/Downloads/s2.0.0/breps/embeddings --vertex_dir /home/konstantinos/Downloads/s2.0.0/breps/embeddings --output_dir /home/konstantinos/Downloads/s2.0.0/breps/embeddings_merged
