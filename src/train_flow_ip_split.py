"""Strong-alignment training script for flow IP-edge graphs.

Goal
----
You asked to *mirror the original UniGAD pipeline* as closely as possible while still
respecting a fixed day-based split.

This script intentionally reuses the same core components as `src/main.py`:
1) `utils.Dataset` for graph loading + (optional) subpooling matrix graphs.
2) `pretrain_models.GraphMAE` for self-supervised pretraining.
3) `e2e_models.UnifyMLPDetector` for supervised finetuning/evaluation with `cross_mode`.

Dataset split (strong alignment)
-------------------------------
- Pretrain graph: monday only (self-supervised, no labels used).
- Finetune (supervised): monday + a *small labeled fraction* of the other days.
    This is necessary because monday may contain no anomalies (all-0 edge labels), which
    would make supervised edge classification degenerate.
- Test: the remaining graphs (by default: the full 3 other days), evaluated the same
    way UniGAD reports metrics (AUROC/AUPRC/MacroF1 + optional CM).

Important
---------
GraphMAE in this repo reconstructs **node features** (`g.ndata['feature']`). For our
IP-node graphs, most information is on edges (`g.edata['feature']`), so we derive node
features by aggregating edge features (out-mean and in-mean, then concatenation).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import dgl
from dgl.data.utils import load_graphs, save_graphs

from sklearn.feature_selection import chi2

from utils import Dataset, set_seed
from pretrain_models import GraphMAE
from e2e_models import UnifyMLPDetector


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--graph_dir", type=str, default="datasets/edge_labels")

    # graph file names (under graph_dir)
    p.add_argument("--monday_graph", type=str, default="enp0s3-monday-ip-els")
    p.add_argument(
        "--other_graphs",
        type=str,
        default="enp0s3-ip.test1-ip-els,enp0s3-ip.test2-ip-els,enp0s3-ip.test3-ip-els",
        help="Comma-separated graph names used as the 3 'other days'.",
    )
    p.add_argument(
        "--other_graphs_list",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional space-separated list variant of other graphs. "
            "If provided, it overrides --other_graphs. Useful on Windows shells."
        ),
    )
    p.add_argument(
        "--other_split_mode",
        type=str,
        default="graph",
        choices=["graph", "edge"],
        help=(
            "How to split the 3 other_graphs into finetune-pool and test: "
            "'graph' = select whole graphs (original behavior); "
            "'edge' = treat the 3 graphs as one pooled set of edges, then split edges by ratio "
            "into val/test graphs (two new graphs will be created)."
        ),
    )
    p.add_argument(
        "--train_edge_ratio",
        type=float,
        default=0.0,
        help=(
            "When other_split_mode=edge: fraction of pooled other-day edges used for finetune training. "
            "If <=0, it will be set to 1 - val_edge_ratio - test_edge_ratio (and clipped)."
        ),
    )
    p.add_argument(
        "--val_edge_ratio",
        type=float,
        default=0.33,
        help="When other_split_mode=edge: fraction of pooled other-day edges used for validation.",
    )
    p.add_argument(
        "--test_edge_ratio",
        type=float,
        default=0.67,
        help="When other_split_mode=edge: fraction of pooled other-day edges used for test (rest unused).",
    )
    p.add_argument(
        "--save_split_graphs",
        action="store_true",
        help="When other_split_mode=edge: save the generated pooled val/test graphs under graph_dir.",
    )
    p.add_argument(
        "--split_suffix",
        type=str,
        default=".edgepool",
        help="Suffix used when saving pooled val/test graphs (only for other_split_mode=edge).",
    )
    p.add_argument(
        "--finetune_take_ratio",
        type=float,
        default=0.10,
        help="Fraction of each 'other day' graph to mix into finetune set (graph-level mix).",
    )
    p.add_argument(
        "--finetune_seed",
        type=int,
        default=42,
        help="Seed for choosing which graphs go to finetune from the other days.",
    )

    p.add_argument(
        "--val_pos_ratio",
        type=float,
        default=0.0,
        help=(
            "Ensure validation has anomalies by moving a small ratio of 'other-day' graphs into val. "
            "This only affects the graph-level train/val split inside the finetune pool. 0 disables."
        ),
    )

    p.add_argument(
        "--min_val_pos_graphs",
        type=int,
        default=1,
        help=(
            "Force validation split to contain at least this many graphs that have positive edge_label. "
            "If feasible, graphs are moved from train->val (only from other-day finetune graphs). 0 disables."
        ),
    )

    p.add_argument("--device", type=str, default="cuda")

    # model / train hyperparams (keep aligned with main.py defaults)
    p.add_argument("--hid_dim", type=int, default=32)
    p.add_argument("--num_layer_pretrain", type=int, default=2)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--replace_ratio", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--act", type=str, default="leakyrelu")
    p.add_argument("--act_ft", type=str, default="ReLU", help="Activation for finetune MLP (align with main.py)")
    p.add_argument("--norm", type=str, default="")
    p.add_argument("--residual", action="store_true", default=False)
    p.add_argument("--concat", action="store_true", default=False, help="Keep for arg-compat; used by some modes")

    # align naming with original main.py args where possible
    p.add_argument("--epoch_pretrain", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--l2", type=float, default=0.0)

    p.add_argument("--epoch_ft", type=int, default=200)
    p.add_argument("--lr_ft", type=float, default=0.003)
    p.add_argument("--l2_ft", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=1)

    # UniGAD e2e
    p.add_argument("--cross_mode", type=str, default="e2e", help="e.g. e2e, ne2ne, n2n")
    p.add_argument("--khop", type=int, default=1)
    p.add_argument("--sp_type", type=str, default="star+norm")
    p.add_argument("--metric", type=str, default="AUROC")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--stitch_mlp_layers", type=int, default=1)
    p.add_argument("--final_mlp_layers", type=int, default=2)
    p.add_argument("--node_loss_weight", type=float, default=1.0)
    p.add_argument("--edge_loss_weight", type=float, default=1.0)
    p.add_argument("--graph_loss_weight", type=float, default=1.0)

    # debug/prints (same flags as utils.get_args)
    p.add_argument('--print_cm', action='store_true', help='Print confusion matrix (best threshold) during evaluation')
    p.add_argument('--log_debug', action='store_true', help='Save evaluation debug logs under results/debug_logs/')
    p.add_argument('--debug_summary_every', type=int, default=10)

    # self-supervised contrastive learning (SimCLR / NT-Xent) on finetune batches
    p.add_argument(
        '--lambda_contrast',
        type=float,
        default=0.0,
        help=(
            'Weight of self-supervised contrastive loss added during finetune. '
            '0 disables contrastive learning.'
        ),
    )
    p.add_argument(
        '--contrast_tau',
        type=float,
        default=0.2,
        help='Temperature tau for NT-Xent (contrastive) loss.',
    )
    p.add_argument(
        '--contrast_view_mode',
        type=str,
        default='dropout',
        choices=['dropout'],
        help=(
            'How to create two views for self-supervised contrastive learning. '
            'Currently only uses model dropout stochasticity.'
        ),
    )
    p.add_argument(
        '--contrast_on',
        type=str,
        default='e',
        choices=['e', 'n', 'g'],
        help='Which route embedding to apply contrastive loss on (default: edge embeddings).',
    )
    p.add_argument(
        '--max_contrast_samples',
        type=int,
        default=2048,
        help=(
            'Max number of samples used in NT-Xent per batch (subsampled) to avoid O(N^2) memory blow-up. '
            'Set <=0 to disable subsampling (not recommended on CPU).'
        ),
    )

    # MisStage print limiting
    p.add_argument('--mis_stage_max_print', type=int, default=30, help='Max FP/FN samples to print in MisStage per category.')

    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--skip_pretrain", action="store_true")
    p.add_argument("--encoder", type=str, default="gcn", help="Encoder type for GraphMAE (gcn/gin/bwgnn/gat/graphsage)")
    p.add_argument("--decoder", type=str, default="gcn", help="Decoder type for GraphMAE (gcn/gin/mlp)")

    p.add_argument(
        "--edge_drop_ratio",
        type=float,
        default=0.0,
        help=(
            "Randomly drop this fraction of edges (structure augmentation) before deriving node features. "
            "Applied to all graphs (monday/finetune/test). 0 disables."
        ),
    )

    p.add_argument(
        "--save_augmented_graphs",
        action="store_true",
        help="If set, save derived node features and weak node/graph labels back to disk (new files).",
    )
    p.add_argument(
        "--aug_suffix",
        type=str,
        default=".aug",
        help="Suffix inserted into saved augmented graph names (before optional extension).",
    )

    # chi2 feature selection (edge features)
    p.add_argument(
        "--chi2_topk",
        type=int,
        default=0,
        help=(
            "If >0, select top-k edge feature dimensions by chi-square score computed on the finetune pool "
            "(using edge_label). Applied consistently to monday/finetune/test graphs before deriving node features."
        ),
    )
    return p.parse_args()


def _apply_edge_feature_select(g: dgl.DGLGraph, keep_idx: torch.Tensor) -> dgl.DGLGraph:
    g = g.local_var()
    if "feature" not in g.edata:
        return g
    x = g.edata["feature"]
    if not torch.is_tensor(x) or x.ndim != 2:
        return g
    g.edata["feature"] = x[:, keep_idx]
    return g


def _compute_chi2_keep_idx(graphs: Sequence[dgl.DGLGraph], topk: int) -> torch.Tensor | None:
    """Compute chi2 keep indices for edge features from a list of graphs.

    Notes:
    - chi2 requires non-negative features, so we shift each feature dim by its global min.
    - Only edges with available edge_label are used.
    """
    k = int(topk)
    if k <= 0:
        return None

    xs = []
    ys = []
    for g in graphs:
        if "feature" not in g.edata or "edge_label" not in g.edata:
            continue
        x = g.edata["feature"].detach().cpu()
        y = g.edata["edge_label"].detach().cpu().reshape(-1)
        if x.ndim != 2 or y.numel() != x.shape[0]:
            continue
        xs.append(x)
        ys.append(y)

    if not xs:
        print("[chi2] skip: no graphs/labels available for chi2 selection")
        return None

    X = torch.cat(xs, dim=0).float().numpy()
    y = torch.cat(ys, dim=0).long().numpy()

    if np.unique(y).size < 2:
        print("[chi2] skip: only one class in finetune pool; chi2 undefined")
        return None

    # make non-negative
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mins = X.min(axis=0, keepdims=True)
    X = X - mins
    X[X < 0] = 0.0

    scores, _ = chi2(X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    d = scores.shape[0]
    k = min(k, d)
    keep = np.argsort(scores)[-k:]
    keep = np.sort(keep)
    print(f"[chi2] select topk={k}/{d} (computed on finetune pool edges)")
    return torch.tensor(keep, dtype=torch.long)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _graph_path(graph_dir: str, name: str) -> Path:
    p = Path(graph_dir)
    if not p.is_absolute():
        p = _repo_root() / p
    return p / name


def _load_graph(graph_file: Path) -> dgl.DGLGraph:
    graphs, _ = load_graphs(str(graph_file))
    return graphs[0]


def _concat_graph_edges(graphs: Sequence[dgl.DGLGraph]) -> dgl.DGLGraph:
    """Concatenate edges of multiple IP-edge graphs into one graph.

    Assumptions:
    - Graphs use the same node ID space / semantics (this holds for our saved datasets).
    - Edge feature dims and edata keys are consistent.
    """
    if len(graphs) == 1:
        return graphs[0]
    bg = dgl.batch(graphs)
    g = dgl.to_homogeneous(bg)
    # The above loses original ndata/edata keys; instead we manually build using concatenation.
    # We'll do a simpler and safe concat: create a disjoint-union then relabel nodes back by max N.
    # For our use-case, we want to pool edges, not nodes, so we require same num_nodes.
    n = int(max(int(gg.num_nodes()) for gg in graphs))
    src_all = []
    dst_all = []
    for gg in graphs:
        s, d = gg.edges()
        src_all.append(s)
        dst_all.append(d)
    src = torch.cat(src_all, dim=0)
    dst = torch.cat(dst_all, dim=0)
    out = dgl.graph((src, dst), num_nodes=n)
    # concat all edata keys that exist in any graph; missing keys will be filled with zeros
    all_keys = set()
    for gg in graphs:
        all_keys.update(list(gg.edata.keys()))
    for k in sorted(all_keys):
        vals = []
        ref_shape = None
        ref_dtype = None
        for gg in graphs:
            if k in gg.edata:
                v = gg.edata[k]
                if ref_shape is None:
                    ref_shape = v.shape[1:]
                    ref_dtype = v.dtype
                vals.append(v)
            else:
                # fill zeros for missing key
                if ref_shape is None:
                    continue
                z = torch.zeros((gg.num_edges(),) + tuple(ref_shape), dtype=ref_dtype)
                vals.append(z)
        if ref_shape is None:
            continue
        out.edata[k] = torch.cat(vals, dim=0)
    return out


def _split_pooled_other_edges(
    other_graphs: Sequence[dgl.DGLGraph],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,

) -> tuple[dgl.DGLGraph, dgl.DGLGraph, dgl.DGLGraph]:
    """Pool edges from 3 other graphs then split edges into (train_graph, val_graph, test_graph).

    We keep num_nodes unchanged and split only edges + all edata.
    """
    g_pool = _concat_graph_edges(other_graphs)
    n_e = int(g_pool.num_edges())
    if n_e == 0:
        return g_pool, g_pool

    rr = float(train_ratio)
    vr = float(val_ratio)
    tr = float(test_ratio)
    rr = max(0.0, min(1.0, rr))
    vr = max(0.0, min(1.0, vr))
    tr = max(0.0, min(1.0, tr))

    # If train_ratio not specified, use the remainder.
    if rr <= 0.0:
        rr = max(0.0, 1.0 - vr - tr)

    # If still invalid, fall back to a safe default.
    s = rr + vr + tr
    if s <= 0:
        rr, vr, tr = 0.0, 0.33, 0.67
        s = 1.0

    # Normalize if sums > 1.
    if s > 1.0:
        rr /= s
        vr /= s
        tr /= s

    rng = np.random.default_rng(int(seed) + 778)
    perm = rng.permutation(n_e)

    n_train = int(round(n_e * rr))
    n_val = int(round(n_e * vr))
    n_test = int(round(n_e * tr))

    # Make sure val/test are non-empty for threshold selection + evaluation.
    n_val = max(1, min(n_e - 2, n_val))
    n_test = max(1, min(n_e - n_val - 1, n_test))
    # Train can be 0 (meaning: monday-only supervised training).
    n_train = max(0, min(n_e - n_val - n_test, n_train))

    idx_train = perm[:n_train]
    idx_val = perm[n_train : n_train + n_val]
    idx_test = perm[n_train + n_val : n_train + n_val + n_test]

    def _subgraph_by_eid(eids: np.ndarray) -> dgl.DGLGraph:
        e = torch.as_tensor(eids, dtype=torch.long)
        sg = dgl.edge_subgraph(g_pool, e, relabel_nodes=False, store_ids=False)
        # edge_subgraph keeps edata for selected edges
        return sg

    g_train = _subgraph_by_eid(idx_train) if n_train > 0 else dgl.edge_subgraph(g_pool, torch.zeros((0,), dtype=torch.long), relabel_nodes=False, store_ids=False)
    g_val = _subgraph_by_eid(idx_val)
    g_test = _subgraph_by_eid(idx_test)
    return g_train, g_val, g_test


def _maybe_drop_edges(g: dgl.DGLGraph, drop_ratio: float, seed: int) -> dgl.DGLGraph:
    """Optionally drop edges while keeping corresponding edata aligned.

    This is a simple DropEdge-like augmentation. We drop a random subset of edges
    and keep all edata fields for the remaining edges.
    """
    r = float(drop_ratio)
    if r <= 0:
        return g
    if r >= 1:
        # keep at least one edge to avoid empty-graph corner cases
        r = 0.999

    g = g.local_var()
    m = g.num_edges()
    if m <= 1:
        return g

    k_drop = int(round(m * r))
    k_drop = max(0, min(m - 1, k_drop))
    if k_drop == 0:
        return g

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(m, generator=gen)
    keep_eids = perm[k_drop:]
    # DGL 1.0.2 signature: edge_subgraph(graph, edges, *, relabel_nodes=True, store_ids=True, ...)
    # We want to keep original node IDs and preserve all nodes. So we:
    # 1) take an edge-induced subgraph with relabel_nodes=False
    # 2) rebuild a new graph with num_nodes=g.num_nodes() to keep isolated nodes
    sg = dgl.edge_subgraph(g, keep_eids, relabel_nodes=False, store_ids=True)

    parent_eids = sg.edata.get(dgl.EID, keep_eids)
    src, dst = sg.edges()
    new_g = dgl.graph((src, dst), num_nodes=g.num_nodes())

    # Copy node data (all nodes preserved)
    for k, v in g.ndata.items():
        new_g.ndata[k] = v

    # Copy edge data for kept edges
    for k, v in g.edata.items():
        new_g.edata[k] = v[parent_eids]

    return new_g


def _ensure_node_features_from_edges(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """Derive `g.ndata['feature']` from `g.edata['feature']` for IP-node graphs.

    GraphMAE in this repo reconstructs node features. Our flow graphs store information
    on edges, so we aggregate edge features to nodes.

    For anomaly detection, a pure mean can dilute sparse abnormal bursts, so we use
    both mean and max for in/out edges:

        x_node = concat(out_mean, out_max, in_mean, in_max)

    If `g.ndata['feature']` already exists and is non-empty, we keep it.
    """
    if "feature" in g.ndata:
        x = g.ndata["feature"]
        if torch.is_tensor(x) and x.numel() > 0 and x.shape[1] > 0:
            # If it's not all-zero, keep it. If all-zero, we still regenerate.
            if torch.any(x != 0):
                return g

    if "feature" not in g.edata:
        raise KeyError("Graph must have g.edata['feature'] to derive node features.")

    g = g.local_var()
    ef = g.edata["feature"].float()
    src, dst = g.edges()
    n = g.num_nodes()
    d = ef.shape[1]

    # out mean / max
    out_sum = torch.zeros((n, d), dtype=ef.dtype)
    out_cnt = torch.zeros((n, 1), dtype=ef.dtype)
    out_sum.index_add_(0, src, ef)
    out_cnt.index_add_(0, src, torch.ones((ef.shape[0], 1), dtype=ef.dtype))
    out_mean = out_sum / (out_cnt + 1e-12)

    out_max = torch.full((n, d), -float("inf"), dtype=ef.dtype)
    out_max.index_reduce_(0, src, ef, reduce="amax")
    out_max = torch.nan_to_num(out_max, neginf=0.0)

    # in mean / max
    in_sum = torch.zeros((n, d), dtype=ef.dtype)
    in_cnt = torch.zeros((n, 1), dtype=ef.dtype)
    in_sum.index_add_(0, dst, ef)
    in_cnt.index_add_(0, dst, torch.ones((ef.shape[0], 1), dtype=ef.dtype))
    in_mean = in_sum / (in_cnt + 1e-12)

    in_max = torch.full((n, d), -float("inf"), dtype=ef.dtype)
    in_max.index_reduce_(0, dst, ef, reduce="amax")
    in_max = torch.nan_to_num(in_max, neginf=0.0)

    g.ndata["feature"] = torch.cat([out_mean, out_max, in_mean, in_max], dim=1)
    return g


def _attach_weak_node_graph_labels(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """Create weak node/graph labels from edge labels if missing.

    - node_label[v] = 1 if any incident edge has edge_label==1 else 0
    - graph_label = 1 if any edge has edge_label==1 else 0
    """
    if "edge_label" not in g.edata:
        return g

    g = g.local_var()
    el = g.edata["edge_label"].long().reshape(-1)
    src, dst = g.edges()
    n = g.num_nodes()

    node_any = torch.zeros((n,), dtype=torch.long)
    inc = (el > 0).long()
    # mark endpoints
    node_any.index_put_((src,), torch.maximum(node_any[src], inc), accumulate=False)
    node_any.index_put_((dst,), torch.maximum(node_any[dst], inc), accumulate=False)

    if "node_label" not in g.ndata:
        g.ndata["node_label"] = node_any
    return g


@dataclass
class _SplitFiles:
    pretrain: Path
    finetune: List[Path]
    test: List[Path]


def _choose_finetune_and_test(other_files: Sequence[Path], take_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = np.random.default_rng(int(seed))
    others = list(other_files)
    if len(others) == 0:
        return [], []
    idx = np.arange(len(others))
    rng.shuffle(idx)
    k = int(max(1, round(len(others) * float(take_ratio)))) if take_ratio > 0 else 0
    finetune = [others[i] for i in idx[:k]]
    test = [others[i] for i in idx[k:]]
    if len(test) == 0:
        # keep at least one test graph if possible
        test = finetune[-1:]
        finetune = finetune[:-1]
    return finetune, test


def _save_augmented(graph_file: Path, g: dgl.DGLGraph, aug_suffix: str) -> Path:
    """Save a single augmented graph to a new file next to the original."""
    out = graph_file.with_name(graph_file.name + aug_suffix)
    save_graphs(str(out), [g])
    return out


def _collect_labels_for_dataset(ds: Dataset) -> None:
    """Populate ds.node_label / ds.edge_label lists similar to Dataset.prepare_dataset.

    We don't call `prepare_dataset()` because it performs stratified splits based on
    graph-level labels, which we don't have for these flow graphs.
    """
    ds.node_label = []
    ds.edge_label = []
    for g in ds.graph_list:
        g.ndata["feature"] = g.ndata["feature"].float()
        if hasattr(ds, "labels_have") and "n" in getattr(ds, "labels_have", ""):
            if "node_label" in g.ndata:
                ds.node_label.append(g.ndata["node_label"])
        if hasattr(ds, "labels_have") and "e" in getattr(ds, "labels_have", ""):
            if "edge_label" in g.edata:
                ds.edge_label.append(g.edata["edge_label"])


def _graph_has_pos_edge(g: dgl.DGLGraph) -> bool:
    """Return True if graph contains any positive edge label."""
    if "edge_label" not in g.edata:
        return False
    el = g.edata["edge_label"].detach().reshape(-1)
    if el.numel() == 0:
        return False
    return bool((el > 0).any().item())


def _ensure_val_has_positives(
    mix_graphs: Sequence[dgl.DGLGraph],
    idx_train: List[int],
    idx_val: List[int],
    candidate_idx: Sequence[int],
    min_val_pos_graphs: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Force the val split to include at least `min_val_pos_graphs` positive graphs.

    We only move graphs from `idx_train` to `idx_val` (never touch test), limited to
    indices in `finetune_candidate_idx` (typically other-day graphs, excluding monday).
    """
    m = int(min_val_pos_graphs)
    if m <= 0:
        return idx_train, idx_val

    idx_train = list(idx_train)
    idx_val = list(idx_val)

    val_pos = [i for i in idx_val if _graph_has_pos_edge(mix_graphs[i])]
    need = m - len(val_pos)
    if need <= 0:
        return idx_train, idx_val

    cand = [i for i in candidate_idx if i in idx_train and _graph_has_pos_edge(mix_graphs[i])]
    if not cand:
        # We can't manufacture positives if none exist in the candidate pool.
        return idx_train, idx_val

    rng = np.random.default_rng(int(seed) + 2026)
    rng.shuffle(cand)
    pick = cand[:need]
    if not pick:
        return idx_train, idx_val

    idx_train = [i for i in idx_train if i not in pick]
    idx_val = sorted(set(idx_val).union(pick))

    # keep non-empty train (at least one graph)
    if len(idx_train) == 0:
        idx_train = idx_val[-1:]
        idx_val = idx_val[:-1]
        if len(idx_val) == 0:
            idx_val = idx_train

    return idx_train, idx_val


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)

    graph_dir = Path(args.graph_dir)
    if not graph_dir.is_absolute():
        graph_dir = _repo_root() / graph_dir

    monday_fp = graph_dir / args.monday_graph
    if not monday_fp.exists():
        raise FileNotFoundError(f"monday graph file not found: {monday_fp}")

    if getattr(args, "other_graphs_list", None):
        other_names = [str(s).strip() for s in args.other_graphs_list if str(s).strip()]
    else:
        other_names = [s.strip() for s in str(args.other_graphs).split(",") if s.strip()]
    other_files = [graph_dir / nm for nm in other_names]
    for fp in other_files:
        if not fp.exists():
            raise FileNotFoundError(f"other graph file not found: {fp}")

    # Decide how to split other_graphs.
    other_split_mode = str(getattr(args, "other_split_mode", "graph"))
    finetune_files: list[Path] = []
    test_files: list[Path] = []
    pooled_train_graph: dgl.DGLGraph | None = None
    pooled_val_graph: dgl.DGLGraph | None = None
    pooled_test_graph: dgl.DGLGraph | None = None

    if other_split_mode == "edge":
        # We'll load all other graphs, pool edges, then split edges into two NEW graphs:
        # one for validation and one for test.
        print("[split] other_split_mode=edge: pooling edges from other_graphs then splitting by ratio")
        other_raw = [_load_graph(fp) for fp in other_files]
        g_tr_p, g_val_p, g_test_p = _split_pooled_other_edges(
            other_raw,
            train_ratio=float(getattr(args, "train_edge_ratio", 0.0)),
            val_ratio=float(getattr(args, "val_edge_ratio", 0.33)),
            test_ratio=float(getattr(args, "test_edge_ratio", 0.67)),
            seed=int(args.finetune_seed),
        )
        pooled_train_graph = g_tr_p
        pooled_val_graph = g_val_p
        pooled_test_graph = g_test_p
        if bool(getattr(args, "save_split_graphs", False)):
            suf = str(getattr(args, "split_suffix", ".edgepool"))
            train_name = f"__pooled_other_train{suf}"
            val_name = f"__pooled_other_val{suf}"
            test_name = f"__pooled_other_test{suf}"
            save_graphs(str(graph_dir / train_name), [pooled_train_graph], {})
            save_graphs(str(graph_dir / val_name), [pooled_val_graph], {})
            save_graphs(str(graph_dir / test_name), [pooled_test_graph], {})
            print(f"[split] saved pooled train graph: {train_name}")
            print(f"[split] saved pooled val graph: {val_name}")
            print(f"[split] saved pooled test graph: {test_name}")
    else:
        finetune_files, test_files = _choose_finetune_and_test(
            other_files, take_ratio=float(args.finetune_take_ratio), seed=int(args.finetune_seed)
        )
        if len(finetune_files) == 0:
            print(
                "[warn] finetune set contains only monday (no other-day graphs were mixed in). "
                "If monday has no anomalies, edge supervised training may degenerate."
            )

    # ------------------------------------------------------------------
    # (A) Load + augment graphs (derive node features from edge features)
    # ------------------------------------------------------------------
    print("[1/3] Loading graphs and deriving node features from edge features...")
    g_monday = _load_graph(monday_fp)
    g_monday = _maybe_drop_edges(g_monday, float(args.edge_drop_ratio), int(args.seed) + 1001)
    g_monday = _attach_weak_node_graph_labels(g_monday)

    g_finetune_others = []
    for fp in finetune_files:
        gg = _load_graph(fp)
        gg = _maybe_drop_edges(gg, float(args.edge_drop_ratio), int(args.seed) + 2000 + len(g_finetune_others))
        gg = _attach_weak_node_graph_labels(gg)
        g_finetune_others.append(gg)

    g_test_list = []
    if other_split_mode == "edge":
        assert pooled_test_graph is not None and pooled_val_graph is not None and pooled_train_graph is not None
        # Edge pooled split creates three disjoint graphs:
        # - pooled_train_graph goes into finetune pool (train split)
        # - pooled_val_graph is used ONLY for validation (threshold selection)
        # - pooled_test_graph is used ONLY for final test
        g_finetune_others = []
        if pooled_train_graph.num_edges() > 0:
            gtr = _maybe_drop_edges(pooled_train_graph, float(args.edge_drop_ratio), int(args.seed) + 2000)
            gtr = _attach_weak_node_graph_labels(gtr)
            g_finetune_others.append(gtr)

        # We'll pass val/test via the dataset masks (val inside tmp dataset), and explicitly via g_test_list (other-only).
        gt = _maybe_drop_edges(pooled_test_graph, float(args.edge_drop_ratio), int(args.seed) + 3000)
        gt = _attach_weak_node_graph_labels(gt)
        # pooled_test_graph will also be placed into Dataset (scheme A), but we keep g_test_list empty here.
        pooled_test_graph = gt
        g_test_list = []

        # IMPORTANT: pooled_val_graph should NOT be part of finetune_others; it must only be used for validation.
        # We'll append it later into the tmp dataset as a dedicated val graph, but we must still
        # derive its node features so GraphMAE/UniGAD can consume it.

        gv = _maybe_drop_edges(pooled_val_graph, float(args.edge_drop_ratio), int(args.seed) + 2500)
        gv = _attach_weak_node_graph_labels(gv)
        pooled_val_graph = gv
    else:
        for fp in test_files:
            gg = _load_graph(fp)
            gg = _maybe_drop_edges(gg, float(args.edge_drop_ratio), int(args.seed) + 3000 + len(g_test_list))
            gg = _attach_weak_node_graph_labels(gg)
            g_test_list.append(gg)

    # Optional chi2 feature selection on edge features, computed from finetune pool edges.
    keep_idx = _compute_chi2_keep_idx([g_monday] + g_finetune_others, topk=int(getattr(args, "chi2_topk", 0)))
    if keep_idx is not None:
        g_monday = _apply_edge_feature_select(g_monday, keep_idx)
        g_finetune_others = [_apply_edge_feature_select(gg, keep_idx) for gg in g_finetune_others]
        g_test_list = [_apply_edge_feature_select(gg, keep_idx) for gg in g_test_list]
        if other_split_mode == "edge":
            assert pooled_val_graph is not None
            pooled_val_graph = _apply_edge_feature_select(pooled_val_graph, keep_idx)
            assert pooled_test_graph is not None
            pooled_test_graph = _apply_edge_feature_select(pooled_test_graph, keep_idx)

    # Derive node features from (possibly selected) edge features
    g_monday = _ensure_node_features_from_edges(g_monday)
    g_finetune_others = [_ensure_node_features_from_edges(gg) for gg in g_finetune_others]
    g_test_list = [_ensure_node_features_from_edges(gg) for gg in g_test_list]
    if other_split_mode == "edge":
        assert pooled_val_graph is not None
        pooled_val_graph = _ensure_node_features_from_edges(pooled_val_graph)
        assert pooled_test_graph is not None
        pooled_test_graph = _ensure_node_features_from_edges(pooled_test_graph)

    # Ensure all graphs share the same node feature dimensionality.
    # This matters especially when mixing different graph sources (e.g., monday tfaug vs other tf)
    # or when chi2_topk changes the edge feature dim, which then affects derived node features.
    base_dim = int(g_monday.ndata["feature"].shape[1])

    def _clip_node_feat_dim(gg: dgl.DGLGraph, dim: int) -> dgl.DGLGraph:
        x = gg.ndata.get("feature")
        if x is None:
            return gg
        if int(x.shape[1]) == int(dim):
            return gg
        if int(x.shape[1]) < int(dim):
            raise ValueError(
                f"node feature dim mismatch: got {int(x.shape[1])} < base_dim={int(dim)}"
            )
        gg.ndata["feature"] = x[:, :dim].contiguous()
        return gg

    g_finetune_others = [_clip_node_feat_dim(gg, base_dim) for gg in g_finetune_others]
    g_test_list = [_clip_node_feat_dim(gg, base_dim) for gg in g_test_list]
    if other_split_mode == "edge":
        assert pooled_val_graph is not None
        pooled_val_graph = _clip_node_feat_dim(pooled_val_graph, base_dim)
        assert pooled_test_graph is not None
        pooled_test_graph = _clip_node_feat_dim(pooled_test_graph, base_dim)

    if args.save_augmented_graphs:
        print("[save_augmented_graphs] saving augmented graphs (node features) to disk...")
        _save_augmented(monday_fp, g_monday, args.aug_suffix)
        for fp, gg in zip(finetune_files, g_finetune_others):
            _save_augmented(fp, gg, args.aug_suffix)
        for fp, gg in zip(test_files, g_test_list):
            _save_augmented(fp, gg, args.aug_suffix)

    in_dim = int(g_monday.ndata["feature"].shape[1])
    print(f"  monday nodes={g_monday.num_nodes()} edges={g_monday.num_edges()} in_dim={in_dim}")

    # ------------------------------------------------------------------
    # (B) Pretrain GraphMAE on monday only (same as main.py pretrain)
    # ------------------------------------------------------------------
    print("[2/3] Pretraining GraphMAE on monday graph only...")
    pretrain_model = GraphMAE(
        in_dim=in_dim,
        hid_dim=args.hid_dim,
        num_layer=args.num_layer_pretrain,
        drop_ratio=args.dropout,
        act=args.act,
        norm=args.norm,
        residual=args.residual,
        mask_ratio=args.mask_ratio,
        encoder_type=args.encoder,
        decoder_type=args.decoder,
        replace_ratio=args.replace_ratio,
    ).to(args.device)

    if args.skip_pretrain:
        print("[skip_pretrain] Using randomly initialized GraphMAE encoder.")
    else:
        optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=args.lr, weight_decay=args.l2)
        g = g_monday.to(args.device)
        pretrain_model.train()
        for ep in range(int(args.epoch_pretrain)):
            loss, loss_dict = pretrain_model(g, g.ndata["feature"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  [pretrain] epoch={ep+1}/{args.epoch_pretrain} loss={float(loss.item()):.4f} {loss_dict}")
        g_monday = g_monday.to("cpu")

    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # (C) Finetune + evaluate using the original UnifyMLPDetector
    # ------------------------------------------------------------------
    print("[3/3] Finetuning with UnifyMLPDetector (aligned with main.py)...")

    # Build a temporary multi-graph dataset file on disk so we can reuse Dataset.
    # Dataset expects `load_graphs(prefix + name)` to return a list of graphs and a label dict.
    tmp_name = f"__tmp_flowip_mix_{np.random.randint(0, 1_000_000)}"
    tmp_fp = graph_dir / tmp_name

    # In edge pooled split mode, we keep roles strictly separated but still place
    # pooled_test inside Dataset so UnifyMLPDetector can run a proper FinalTest.
    #   mix_graphs = [monday] + [pooled_train?] + [pooled_val] + [pooled_test]
    if other_split_mode == "edge":
        assert pooled_val_graph is not None and pooled_test_graph is not None
        mix_graphs = [g_monday] + g_finetune_others + [pooled_val_graph, pooled_test_graph]
    else:
        mix_graphs = [g_monday] + g_finetune_others + g_test_list

    # Graph-level label for g2g: mark a graph anomalous if it contains any anomalous edges.
    # This keeps strong alignment while enabling graph-level evaluation.
    glabel = []
    for gg in mix_graphs:
        if "edge_label" in gg.edata:
            glabel.append(int(bool((gg.edata["edge_label"].reshape(-1) > 0).any().item())))
        else:
            glabel.append(0)
    glabel = torch.tensor(glabel, dtype=torch.long)
    save_graphs(str(tmp_fp), mix_graphs, {"glabel": glabel})

    # IMPORTANT: `Dataset` loads graphs by `load_graphs(prefix + name)`.
    # Use an OS-correct directory prefix ending with the current separator.
    dataset_prefix = os.fspath(graph_dir) + os.sep
    dataset = Dataset(
        tmp_name,
        prefix=dataset_prefix,
        labels_have="neg",
        sp_type=args.sp_type,
    )

    # Override splits to match our fixed split:
    # - train/val: monday + finetune_others
    # - test: test_list
    # We implement it by directly setting masks over graphs.
    n_monday = 1
    n_ft = len(g_finetune_others)
    if other_split_mode == "edge":
        # pooled_val + pooled_test
        n_test = 1
        total = n_monday + n_ft + 2
    else:
        n_test = len(g_test_list)
        total = n_monday + n_ft + n_test

    # manual preparation (skip prepare_dataset which needs graph labels for stratified splits)
    dataset.is_single_graph = False
    _collect_labels_for_dataset(dataset)
    dataset.make_sp_matrix_graph_list(khop=int(args.khop), load_kg=False)

    dataset.graph_train_masks = torch.zeros((total, 1), dtype=torch.bool)
    dataset.graph_val_masks = torch.zeros((total, 1), dtype=torch.bool)
    dataset.graph_test_masks = torch.zeros((total, 1), dtype=torch.bool)

    # train/val split within the finetune pool
    # IMPORTANT: in edge pooled split mode, we already constructed:
    #   mix_graphs = [monday] + [pooled_train?] + [pooled_val]
    # and we must keep pooled_val strictly for validation only.
    if other_split_mode == "edge":
        # monday + pooled_train graphs (if any) => train
        idx_train = list(range(0, 1 + n_ft))
        # last two graphs are pooled_val, pooled_test
        idx_val = [total - 2]
        idx_test = [total - 1]
    else:
        # IMPORTANT: keep monday (index 0) always in train.
        # We split at GRAPH level only: each "other" day is an indivisible unit.
        other_idx = list(range(1, 1 + n_ft))
        rng = np.random.default_rng(int(args.finetune_seed))
        rng.shuffle(other_idx)

        # target ~20% other-day graphs for val, but keep at least 1 when possible
        n_val_other = 0
        if len(other_idx) > 0:
            n_val_other = int(round(len(other_idx) * 0.2))
            n_val_other = max(1, min(len(other_idx), n_val_other))
        idx_val = other_idx[:n_val_other]
        idx_train = [0] + other_idx[n_val_other:]
        idx_test = list(range(n_monday + n_ft, total)) if n_test > 0 else []

    if other_split_mode != "edge":
        # If we can, ensure val contains at least one positive graph by swapping from train.
        if len(other_idx) > 0:
            val_has_pos = any(_graph_has_pos_edge(mix_graphs[i]) for i in idx_val)
            if not val_has_pos:
                train_pos = [i for i in idx_train if i != 0 and _graph_has_pos_edge(mix_graphs[i])]
                if train_pos:
                    swap_in = train_pos[0]
                    swap_out = idx_val[0]
                    idx_train = [i for i in idx_train if i != swap_in] + [swap_out]
                    idx_val = [swap_in] + [i for i in idx_val if i != swap_out]

        # keep non-empty val and train
        if len(idx_val) == 0:
            idx_val = idx_train[-1:]
            idx_train = idx_train[:-1]
            if len(idx_train) == 0:
                idx_train = idx_val

    dataset.graph_train_masks[idx_train, 0] = True
    dataset.graph_val_masks[idx_val, 0] = True
    if len(idx_test) > 0:
        dataset.graph_test_masks[idx_test, 0] = True

    # If monday is all-benign, the random 80/20 split can yield an all-benign val.
    # Optionally force some other-day graphs into validation to make threshold selection meaningful.
    # (Disabled for edge pooled split mode because pooled_val is already dedicated.)
    if other_split_mode != "edge" and float(getattr(args, "val_pos_ratio", 0.0)) > 0 and n_ft > 0:
        # candidates are finetune other-day graphs: indices [1, 1+n_ft)
        # IMPORTANT: only move graphs that actually contain positives; otherwise val may remain all-benign.
        cand_all = list(range(1, 1 + n_ft))
        cand_pos = [i for i in cand_all if _graph_has_pos_edge(mix_graphs[i])]
        if len(cand_pos) == 0:
            print("[split][warn] val_pos_ratio enabled but no positive graphs exist in finetune other-day pool")
        else:
            rng2 = np.random.default_rng(int(args.finetune_seed) + 999)
            rng2.shuffle(cand_pos)
            k = int(max(1, round(len(cand_pos) * float(args.val_pos_ratio))))
            pick_for_val = set(cand_pos[:k])

            # move picked graphs from train->val if needed
            idx_train = [i for i in idx_train if i not in pick_for_val]
            idx_val = sorted(set(idx_val).union(pick_for_val))

        # keep non-empty train
        if len(idx_train) == 0:
            idx_train = idx_val[-1:]
            idx_val = idx_val[:-1]
            if len(idx_val) == 0:
                idx_val = idx_train

            dataset.graph_train_masks[:, 0] = False
            dataset.graph_val_masks[:, 0] = False
            dataset.graph_train_masks[idx_train, 0] = True
            dataset.graph_val_masks[idx_val, 0] = True

    # Hard guarantee: ensure val contains at least N positive graphs if possible.
    # This is crucial for stable threshold selection (otherwise val may be all-benign).
    if int(getattr(args, "min_val_pos_graphs", 0)) > 0:
        # candidates: finetune other-day graphs only (exclude monday), because monday can be all-benign.
        cand_idx = list(range(1, 1 + n_ft))
        before_val_pos = sum(1 for i in idx_val if _graph_has_pos_edge(mix_graphs[i]))
        idx_train, idx_val = _ensure_val_has_positives(
            mix_graphs=mix_graphs,
            idx_train=list(idx_train),
            idx_val=list(idx_val),
            candidate_idx=cand_idx,
            min_val_pos_graphs=int(args.min_val_pos_graphs),
            seed=int(args.finetune_seed),
        )
        after_val_pos = sum(1 for i in idx_val if _graph_has_pos_edge(mix_graphs[i]))
        if after_val_pos < int(args.min_val_pos_graphs):
            print(
                f"[split][warn] min_val_pos_graphs={int(args.min_val_pos_graphs)} not satisfied "
                f"(before={before_val_pos}, after={after_val_pos}). "
                "Likely no positive graphs exist in the finetune pool."
            )

    dataset.graph_train_masks[:, 0] = False
    dataset.graph_val_masks[:, 0] = False
    dataset.graph_train_masks[idx_train, 0] = True
    dataset.graph_val_masks[idx_val, 0] = True

    # Split sanity logs (graph-level and edge-level)
    def _edge_pos_count(i: int) -> int:
        gg = mix_graphs[i]
        if "edge_label" not in gg.edata:
            return 0
        el = gg.edata["edge_label"].detach().reshape(-1)
        if el.numel() == 0:
            return 0
        return int((el > 0).sum().item())

    n_train_pos_graph = sum(1 for i in idx_train if _graph_has_pos_edge(mix_graphs[i]))
    n_val_pos_graph = sum(1 for i in idx_val if _graph_has_pos_edge(mix_graphs[i]))
    n_train_pos_edge = sum(_edge_pos_count(i) for i in idx_train)
    n_val_pos_edge = sum(_edge_pos_count(i) for i in idx_val)
    if other_split_mode == "edge":
        # pooled_test is part of mix_graphs (last graph)
        gg = mix_graphs[total - 1]
        if "edge_label" in gg.edata:
            n_test_pos_edge = int((gg.edata["edge_label"].reshape(-1) > 0).sum().item())
        else:
            n_test_pos_edge = 0
    else:
        n_test_pos_edge = sum(
            int((gg.edata.get("edge_label", torch.zeros(0)).reshape(-1) > 0).sum().item())
            if "edge_label" in gg.edata else 0
            for gg in g_test_list
        )

    # Clarify what is being split.
    # - graph mode: other_graphs are split as whole graphs.
    # - edge mode: other_graphs are pooled then split into two NEW graphs (val/test).
    def _name_of(i: int) -> str:
        if i == 0:
            return str(args.monday_graph)
        j = i - 1
        if 0 <= j < len(finetune_files):
            return str(finetune_files[j].name)
        j2 = j - len(finetune_files)
        if 0 <= j2 < len(test_files):
            return str(test_files[j2].name)
        return f"idx={i}"

    if other_split_mode == "edge":
        print(
            f"[split] EDGE-level pooled split on other_graphs: "
            f"train_edges_ratio={float(getattr(args, 'train_edge_ratio', 0.0)):.3f}, "
            f"val_edges_ratio={float(getattr(args, 'val_edge_ratio', 0.33)):.3f}, "
            f"test_edges_ratio={float(getattr(args, 'test_edge_ratio', 0.67)):.3f}. "
            f"monday is always in train."
        )
    else:
        print(
            f"[split] GRAPH-level split on other_graphs: finetune_others={n_ft}, test_others={n_test}. "
            f"monday is always in train."
        )
    print(
        f"[split] train_graphs(total_in_tmp)={len(idx_train)} (pos_graphs={n_train_pos_graph}, pos_edges={n_train_pos_edge}) | "
        f"val_graphs(total_in_tmp)={len(idx_val)} (pos_graphs={n_val_pos_graph}, pos_edges={n_val_pos_edge}) | "
        f"test_graphs(other_only)={n_test} (pos_edges={n_test_pos_edge})"
    )
    print(f"[split] train_names={[_name_of(i) for i in idx_train]}")
    print(f"[split] val_names={[_name_of(i) for i in idx_val]}")
    if other_split_mode == "edge":
        print("[split] val_names=['__pooled_other_val']")
        print("[split] test_names=['__pooled_other_test']")
    else:
        print(f"[split] test_names={[fp.name for fp in test_files]}")

    train_dl, val_dl, test_dl = dataset.get_graph_and_sp_dataloaders(batch_size=int(args.batch_size), trial_id=0)

    # launch e2e training (same class used by main.py)
    score_test = UnifyMLPDetector(
        pretrain_model,
        dataset,
        (train_dl, val_dl, test_dl),
        cross_mode=str(args.cross_mode),
        args=args,
    ).train()

    print("\n[Result] Test scores (aligned reporting):")
    print(score_test)

    # cleanup temp file
    try:
        tmp_fp.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass


if __name__ == "__main__":
    main()
