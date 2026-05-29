"""Inspect a flow-node DGL dataset saved by build_flow_node_graph.py.

Usage:
  python src/inspect_flow_graph.py --path datasets/edge_labels/enp0s3-merged-els
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from dgl.data.utils import load_graphs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, required=True)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    path = Path(args.path)
    if not path.is_absolute():
        path = repo_root / path

    graphs, labels = load_graphs(str(path))
    print("path:", path)
    print("num_graphs:", len(graphs), "label_dict_keys:", list(labels.keys()))

    g = graphs[0]
    print("nodes:", g.num_nodes(), "edges:", g.num_edges())
    print("ndata:", list(g.ndata.keys()))
    print("edata:", list(g.edata.keys()))

    x = g.ndata.get("feature")
    if x is not None:
        print("feature:", tuple(x.shape), x.dtype)

    y = g.ndata.get("node_label")
    if y is not None:
        y_cpu = y.detach().cpu().flatten()
        vals, cnts = torch.unique(y_cpu, return_counts=True)
        print("node_label counts:", {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())})

    sid = g.ndata.get("source_id")
    if sid is not None:
        sid_cpu = sid.detach().cpu().flatten().numpy()
        uniq, cnt = np.unique(sid_cpu, return_counts=True)
        print("source_id counts:", {int(u): int(c) for u, c in zip(uniq, cnt)})


if __name__ == "__main__":
    main()
