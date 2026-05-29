"""Inspect the local DGL dataset file for `weibo-els`.

This repo stores datasets as serialized DGLGraphs. This script prints:
- number of graphs
- graph sizes
- available node/edge data fields
- feature shape/dtype
- node/edge label distributions

Run from repo root:
    python src/inspect_weibo_els.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from dgl.data.utils import load_graphs


def _tensor_value_stats(x: torch.Tensor, max_unique: int = 20) -> dict:
    x_cpu = x.detach().cpu()
    stats = {
        "shape": tuple(x_cpu.shape),
        "dtype": str(x_cpu.dtype),
    }
    if x_cpu.numel() == 0:
        stats["numel"] = 0
        return stats

    # numeric summary if possible
    if x_cpu.dtype.is_floating_point or x_cpu.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        flat = x_cpu.flatten()
        if x_cpu.dtype.is_floating_point:
            stats["min"] = float(torch.nan_to_num(flat).min().item())
            stats["max"] = float(torch.nan_to_num(flat).max().item())
            stats["mean"] = float(torch.nan_to_num(flat).mean().item())
        else:
            stats["min"] = int(flat.min().item())
            stats["max"] = int(flat.max().item())

    # label-like unique counts (only when small)
    try:
        uniq = torch.unique(x_cpu)
        if uniq.numel() <= max_unique:
            counts = {int(v.item()): int((x_cpu == v).sum().item()) for v in uniq}
            stats["unique_counts"] = counts
    except Exception:
        pass

    return stats


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "datasets" / "edge_labels" / "weibo-els"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    graphs, label_dict = load_graphs(str(data_path))

    print("=== weibo-els (serialized DGLGraphs) ===")
    print("path:", data_path)
    print("num_graphs:", len(graphs))
    print("label_dict_keys:", list(label_dict.keys()))

    # sizes summary
    n_nodes = [g.num_nodes() for g in graphs]
    n_edges = [g.num_edges() for g in graphs]
    print("nodes: min/median/max =", int(np.min(n_nodes)), int(np.median(n_nodes)), int(np.max(n_nodes)))
    print("edges: min/median/max =", int(np.min(n_edges)), int(np.median(n_edges)), int(np.max(n_edges)))

    # inspect first graph fields
    g0 = graphs[0]
    print("\n--- graph[0] fields ---")
    print("num_nodes:", g0.num_nodes(), "num_edges:", g0.num_edges())
    print("ndata keys:", list(g0.ndata.keys()))
    print("edata keys:", list(g0.edata.keys()))

    if "feature" in g0.ndata:
        x = g0.ndata["feature"]
        print("feature stats:", _tensor_value_stats(x, max_unique=10))

    # label distributions aggregated across ALL graphs (important when graph_list)
    def agg_label_counter(get_label_tensor):
        c = Counter()
        total = 0
        for g in graphs:
            t = get_label_tensor(g)
            if t is None:
                continue
            t_cpu = t.detach().cpu().flatten()
            total += int(t_cpu.numel())
            # assume integer labels
            vals, cnts = torch.unique(t_cpu, return_counts=True)
            for v, k in zip(vals.tolist(), cnts.tolist()):
                c[int(v)] += int(k)
        return total, c

    print("\n--- aggregated label distributions ---")
    if "node_label" in g0.ndata:
        total, c = agg_label_counter(lambda g: g.ndata.get("node_label"))
        print("node_label total:", total, "counts:", dict(c))

    if "edge_label" in g0.edata:
        total, c = agg_label_counter(lambda g: g.edata.get("edge_label"))
        print("edge_label total:", total, "counts:", dict(c))

    # If label_dict has glabel, show its distribution
    if "glabel" in label_dict:
        gl = label_dict["glabel"]
        if torch.is_tensor(gl):
            gl_cpu = gl.detach().cpu().flatten()
            vals, cnts = torch.unique(gl_cpu, return_counts=True)
            print("graph_label(glab) shape:", tuple(gl_cpu.shape), "counts:", {int(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())})
        else:
            print("graph_label(glab) type:", type(gl))


if __name__ == "__main__":
    main()
