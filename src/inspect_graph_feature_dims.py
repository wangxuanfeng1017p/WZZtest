"""Inspect DGL graph feature dimensions.

Usage:
  python -u src/inspect_graph_feature_dims.py --graphs a,b,c

Prints number of nodes/edges and feature dims for ndata/edata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dgl.data.utils import load_graphs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--graphs",
        type=str,
        required=True,
        help="Comma-separated graph file paths (relative to repo root or absolute).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    paths = [s.strip() for s in str(args.graphs).split(",") if s.strip()]
    for s in paths:
        p = Path(s)
        if not p.is_absolute():
            p = repo_root / p
        if not p.exists():
            print(f"{p} NOT FOUND")
            continue
        gs, _ = load_graphs(str(p))
        g = gs[0]
        edim = g.edata["feature"].shape[1] if "feature" in g.edata else None
        ndim = g.ndata["feature"].shape[1] if "feature" in g.ndata else None
        print(f"{p}")
        print(f"  nodes={g.num_nodes()} edges={g.num_edges()} edim={edim} ndim={ndim}")


if __name__ == "__main__":
    main()
