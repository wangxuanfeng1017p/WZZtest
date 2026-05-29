"""Build a DGL graph dataset from one or more CIC-style flow CSVs.

This repo originally uses `weibo-els` and other edge-labeled datasets.
For encrypted traffic / flow logs (CICFlowMeter-like), we often start from
`*_Flow.csv` where each row is one network flow.

This script supports two graph types:

1) flow-node (legacy)
     - Node = one flow record
     - Node feature = flow numeric features (log1p + standardize) + one-hot(Protocol)
     - Node label = 1 if Stage != 'Benign' else 0
     - Edge = connect flows that share Src IP or share Dst IP within a time window

2) ip-node (your requested format)
     - Node = one IP address
     - Edge = one directed edge per flow: Src IP -> Dst IP
     - Edge feature = flow numeric features (log1p + standardize) + one-hot(Protocol)
     - Edge label = 1 if Stage != 'Benign' else 0

Output
------
Saves a single DGLGraph to: datasets/edge_labels/<out>

Examples
--------
IP graph (recommended):
    python src/build_flow_node_graph.py --csv enp0s3-monday.pcap_Flow.csv --out enp0s3-monday-ip-els --graph_type ip

Merge 4 CSVs:
    python src/build_flow_node_graph.py --csv a.csv,b.csv,c.csv,d.csv --out enp0s3-4days-ip-els --graph_type ip

Notes
-----
- Required columns: Src IP, Dst IP, Timestamp, Stage
- Protocol is optional.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

import dgl
from dgl.data.utils import save_graphs


DEFAULT_NUMERIC_COLS = [
    # time and size
    "Flow Duration",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
    "Flow Bytes/s",
    "Flow Packets/s",
    # IAT
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    # flags
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWR Flag Count",
    "ECE Flag Count",
    # activity
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


def _encode_str_list_to_tensors(items: list[str]) -> dict[str, torch.Tensor]:
    """Encode a list[str] into tensors for save_graphs label_dict.

    DGL's save_graphs persists label_dict as tensors; Python objects (like list[str])
    are dropped in some versions.

    Returns a dict with:
      - 'bytes': uint8 tensor of concatenated utf-8 bytes
      - 'offsets': int64 tensor of start offsets, length = len(items)+1
    """
    bs_list = [s.encode('utf-8', errors='replace') for s in items]
    offsets = [0]
    total = 0
    for b in bs_list:
        total += len(b)
        offsets.append(total)
    if total == 0:
        data = torch.zeros((0,), dtype=torch.uint8)
    else:
        data = torch.from_numpy(np.frombuffer(b''.join(bs_list), dtype=np.uint8)).clone()
    return {
        'bytes': data.to(dtype=torch.uint8),
        'offsets': torch.tensor(offsets, dtype=torch.int64),
    }


def _existing_cols(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    # flow csv sometimes has inf/NaN; clip to finite
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    return np.log1p(x)


def _standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-12
    return (x - mu) / std


def _parse_timestamp_to_unix_seconds(ts_series: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts_series, errors="coerce", utc=False)
    if ts.isna().any():
        ts_num = pd.to_numeric(ts_series, errors="coerce")
        if ts_num.isna().any():
            bad = int(ts.isna().sum())
            raise ValueError(f"Failed to parse Timestamp for {bad} rows. Please share a sample format.")
        ts = pd.to_datetime(ts_num, unit="s", errors="coerce")
    return (ts.astype("int64") // 10**9).to_numpy()


def _get_flow_features(df: pd.DataFrame, required_cols: list[str]) -> tuple[torch.Tensor, list[str]]:
    num_cols = _existing_cols(df, DEFAULT_NUMERIC_COLS)
    if len(num_cols) == 0:
        excluded = {"Src Port", "Dst Port"}
        excluded.update(required_cols)
        num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in excluded]

    num_x = df[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    num_x = _standardize(_safe_log1p(num_x))

    if "Protocol" in df.columns:
        proto = df["Protocol"].astype(str).fillna("UNK")
        proto_vals = sorted(proto.unique().tolist())
        proto_map = {v: i for i, v in enumerate(proto_vals)}
        proto_idx = proto.map(proto_map).to_numpy()
        proto_oh = np.zeros((len(df), len(proto_vals)), dtype=np.float64)
        proto_oh[np.arange(len(df)), proto_idx] = 1.0
        feat = np.concatenate([num_x, proto_oh], axis=1)
        feat_cols = num_cols + [f"Protocol={v}" for v in proto_vals]
    else:
        feat = num_x
        feat_cols = num_cols

    return torch.tensor(feat, dtype=torch.float32), feat_cols


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path(s) to *_Flow.csv. Use comma to pass multiple files.",
    )
    p.add_argument("--out", type=str, required=True, help="Output dataset name (saved under datasets/edge_labels)")
    p.add_argument(
        "--graph_type",
        type=str,
        default="ip",
        choices=["flow", "ip"],
        help="Graph type: 'flow' (node=flow) or 'ip' (node=ip, edge=flow)",
    )

    # flow-node options
    p.add_argument("--time_window_sec", type=int, default=60, help="(flow graph) connect flows within this many seconds")
    p.add_argument("--max_degree", type=int, default=50, help="(flow graph) cap neighbors per node (per key)")

    # ip-node options
    p.add_argument(
        "--max_edges_per_pair",
        type=int,
        default=0,
        help="(ip graph) cap number of flow edges per (src_ip,dst_ip) pair. 0 disables.",
    )

    p.add_argument("--limit", type=int, default=-1, help="Optional: only use first N rows (smoke test)")

    # time-frequency feature augmentation (for edge detection)
    p.add_argument(
        "--tf_augment",
        action="store_true",
        help=(
            "If set, augment edge features with time-frequency statistics computed per (Src IP, Dst IP) pair "
            "using STFT + wavelet (DWT). Requires a parsable Timestamp column."
        ),
    )
    p.add_argument(
        "--tf_key",
        type=str,
        default="pair",
        choices=["pair"],
        help="Time-series grouping key. Currently only supports pair=(Src IP, Dst IP).",
    )
    p.add_argument(
        "--tf_bin_sec",
        type=int,
        default=1,
        help="Time bin size in seconds when building per-pair time series.",
    )
    p.add_argument(
        "--tf_nperseg",
        type=int,
        default=32,
        help="STFT window length (nperseg) on binned series.",
    )
    p.add_argument(
        "--tf_noverlap",
        type=int,
        default=16,
        help="STFT overlap (noverlap) on binned series.",
    )
    p.add_argument(
        "--tf_wavelet",
        type=str,
        default="db2",
        help="Wavelet name for DWT (pywt).",
    )
    p.add_argument(
        "--tf_level",
        type=int,
        default=3,
        help="DWT decomposition level.",
    )
    return p.parse_args()


def _tf_stats_from_series(
    s: np.ndarray,
    *,
    nperseg: int,
    noverlap: int,
    wavelet: str,
    level: int,
) -> np.ndarray:
    """Compute compact STFT + DWT summary stats from a 1D series.

    Returns a fixed-length vector. Designed to be cheap and robust.
    """
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0:
        return np.zeros((12,), dtype=np.float32)

    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    if np.all(s == s[0]):
        return np.zeros((12,), dtype=np.float32)

    # Lazy import to avoid heavy SciPy import cost when tf augmentation isn't used.
    from scipy import signal
    import pywt

    # STFT summaries
    nper = int(max(4, min(int(nperseg), int(s.size))))
    nov = int(max(0, min(int(noverlap), nper - 1)))
    try:
        _, _, Zxx = signal.stft(s, nperseg=nper, noverlap=nov, boundary=None, padded=False)
        pxx = (np.abs(Zxx) ** 2).astype(np.float64)
        freq_energy = pxx.sum(axis=1)
        total_e = float(freq_energy.sum())
        if total_e <= 0:
            stft = np.zeros((6,), dtype=np.float64)
        else:
            p = freq_energy / (total_e + 1e-12)
            spec_entropy = float(-(p * np.log(p + 1e-12)).sum())
            peak_f = int(np.argmax(freq_energy))
            stft = np.array(
                [
                    total_e,
                    float(freq_energy.mean()),
                    float(freq_energy.std()),
                    float(freq_energy.max()),
                    float(peak_f),
                    spec_entropy,
                ],
                dtype=np.float64,
            )
    except Exception:
        stft = np.zeros((6,), dtype=np.float64)

    # DWT summaries
    try:
        lev = int(max(1, level))
        coeffs = pywt.wavedec(s, wavelet=wavelet, level=lev)
        details = coeffs[1:]
        if len(details) == 0:
            dwt = np.zeros((6,), dtype=np.float64)
        else:
            energies = np.array([float(np.sum(np.square(d))) for d in details], dtype=np.float64)
            dwt = np.array(
                [
                    float(energies.sum()),
                    float(energies.mean()),
                    float(energies.std()),
                    float(energies.max()),
                    float(np.max([np.max(np.abs(d)) for d in details])),
                    float(len(details)),
                ],
                dtype=np.float64,
            )
    except Exception:
        dwt = np.zeros((6,), dtype=np.float64)

    out = np.concatenate([stft, dwt], axis=0)
    return out.astype(np.float32)


def _augment_ip_edges_with_timefreq(
    df: pd.DataFrame,
    base_feat: torch.Tensor,
    t_sec: np.ndarray,
    *,
    bin_sec: int,
    nperseg: int,
    noverlap: int,
    wavelet: str,
    level: int,
) -> tuple[torch.Tensor, list[str]]:
    """Augment each edge (flow row) with TF stats of its (src,dst) pair series."""
    if len(df) == 0:
        return base_feat, []

    src = df["Src IP"].astype(str).to_numpy()
    dst = df["Dst IP"].astype(str).to_numpy()
    pair_key = (pd.Series(src) + "->" + pd.Series(dst)).to_numpy()

    t = np.asarray(t_sec, dtype=np.int64)
    b = int(max(1, bin_sec))
    t0 = int(t.min())
    bin_id = ((t - t0) // b).astype(np.int64)

    x = base_feat.detach().cpu().numpy().astype(np.float32)
    proxy = np.linalg.norm(x, axis=1).astype(np.float64)

    groups: dict[str, list[int]] = {}
    for i, k in enumerate(pair_key):
        groups.setdefault(str(k), []).append(int(i))

    tf_dim = 12
    tf_feat = np.zeros((len(df), tf_dim), dtype=np.float32)
    for _, idx_list in groups.items():
        idx = np.asarray(idx_list, dtype=np.int64)
        bins = bin_id[idx]
        p = proxy[idx]
        bmin = int(bins.min())
        bmax = int(bins.max())
        series = np.zeros((bmax - bmin + 1,), dtype=np.float64)
        np.add.at(series, bins - bmin, p)
        stats = _tf_stats_from_series(series, nperseg=nperseg, noverlap=noverlap, wavelet=wavelet, level=level)
        tf_feat[idx, :] = stats[None, :]

    out = torch.cat([base_feat, torch.from_numpy(tf_feat)], dim=1)
    cols = [
        "stft_total_e",
        "stft_e_mean",
        "stft_e_std",
        "stft_e_max",
        "stft_peak_f",
        "stft_spec_entropy",
        "dwt_total_e",
        "dwt_e_mean",
        "dwt_e_std",
        "dwt_e_max",
        "dwt_max_abs",
        "dwt_n_detail",
    ]
    return out, cols


def _build_flow_node_graph(df: pd.DataFrame, feat_t: torch.Tensor, y_t: torch.Tensor, t_sec: np.ndarray, args) -> dgl.DGLGraph:
    src_ip = df["Src IP"].astype(str).to_numpy()
    dst_ip = df["Dst IP"].astype(str).to_numpy()

    order = np.argsort(t_sec)

    def add_edges_for_key(key_arr: np.ndarray) -> tuple[list[int], list[int]]:
        buckets: dict[str, list[int]] = {}
        src_list: list[int] = []
        dst_list: list[int] = []

        for idx in order:
            k = key_arr[idx]
            buf = buckets.setdefault(k, [])

            t_i = t_sec[idx]
            while buf and (t_i - t_sec[buf[0]]) > args.time_window_sec:
                buf.pop(0)

            if buf:
                neigh = buf[-args.max_degree :]
                for j in neigh:
                    src_list.append(j)
                    dst_list.append(idx)
                    src_list.append(idx)
                    dst_list.append(j)

            buf.append(idx)

        return src_list, dst_list

    s1, d1 = add_edges_for_key(src_ip)
    s2, d2 = add_edges_for_key(dst_ip)

    src_edges = np.array(s1 + s2, dtype=np.int64)
    dst_edges = np.array(d1 + d2, dtype=np.int64)

    mask = src_edges != dst_edges
    src_edges = src_edges[mask]
    dst_edges = dst_edges[mask]

    g = dgl.graph((torch.from_numpy(src_edges), torch.from_numpy(dst_edges)), num_nodes=len(df))
    g.ndata["feature"] = feat_t
    g.ndata["node_label"] = y_t
    g.ndata["source_id"] = torch.tensor(df["__source_id"].to_numpy(), dtype=torch.int64)
    g.edata["edge_label"] = torch.zeros(g.num_edges(), dtype=torch.int64)
    return g


def _build_ip_node_graph(df: pd.DataFrame, feat_t: torch.Tensor, y_t: torch.Tensor, t_sec: np.ndarray, args) -> dgl.DGLGraph:
    src_ip_s = df["Src IP"].astype(str)
    dst_ip_s = df["Dst IP"].astype(str)

    all_ips = pd.Index(pd.concat([src_ip_s, dst_ip_s], ignore_index=True).unique())
    ip_to_id = {ip: i for i, ip in enumerate(all_ips.tolist())}

    u = src_ip_s.map(ip_to_id).to_numpy(dtype=np.int64)
    v = dst_ip_s.map(ip_to_id).to_numpy(dtype=np.int64)

    src_sid = df["__source_id"].to_numpy(dtype=np.int64)

    # Persist per-edge stage info for post-hoc error analysis.
    # We store stage as integer IDs to keep graph compact, and also keep an id->name mapping.
    stage_s = df.get("Stage", pd.Series(["" for _ in range(len(df))])).fillna("").astype(str)
    stage_cat = pd.Categorical(stage_s)
    stage_id = stage_cat.codes.astype(np.int64)
    stage_id2name = [str(x) for x in stage_cat.categories.tolist()]

    if args.max_edges_per_pair and args.max_edges_per_pair > 0:
        pair = pd.Series(u.astype(str) + "->" + v.astype(str))
        order = np.argsort(t_sec)
        kept = np.zeros(len(df), dtype=bool)
        counts: dict[str, int] = {}
        for idx in order:
            k = pair.iat[idx]
            c = counts.get(k, 0)
            if c < args.max_edges_per_pair:
                kept[idx] = True
                counts[k] = c + 1

        u = u[kept]
        v = v[kept]
        t_sec = t_sec[kept]
        src_sid = src_sid[kept]
        stage_id = stage_id[kept]
        feat_t = feat_t[torch.from_numpy(kept)]
        y_t = y_t[torch.from_numpy(kept)]

    g = dgl.graph((torch.from_numpy(u), torch.from_numpy(v)), num_nodes=len(all_ips))

    # node feature placeholder (same dim as edge feature for convenience)
    g.ndata["feature"] = torch.zeros((g.num_nodes(), feat_t.shape[1]), dtype=torch.float32)
    g.ndata["ip_id"] = torch.arange(g.num_nodes(), dtype=torch.int64)

    g.edata["feature"] = feat_t
    g.edata["edge_label"] = y_t
    g.edata["timestamp"] = torch.tensor(t_sec[: g.num_edges()], dtype=torch.int64)
    g.edata["source_id"] = torch.tensor(src_sid[: g.num_edges()], dtype=torch.int64)
    g.edata["stage_id"] = torch.tensor(stage_id[: g.num_edges()], dtype=torch.int64)

    # Save mapping for stage_id -> original string.
    # DGL will persist graph-level attributes when using save_graphs.
    try:
        g.graph_data = getattr(g, "graph_data", {})
        g.graph_data["stage_id2name"] = stage_id2name
        # Some DGL versions prefer `g.gdata` for graph-level metadata.
        g.gdata = getattr(g, "gdata", {})
        g.gdata["stage_id2name"] = stage_id2name
    except Exception:
        pass
    return g


def main() -> None:
    args = _parse_args()

    csv_paths = [Path(p.strip()) for p in args.csv.split(",") if p.strip()]
    if not csv_paths:
        raise ValueError("--csv is empty")
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "datasets" / "edge_labels" / args.out

    dfs = []
    for i, p in enumerate(csv_paths):
        dfi = pd.read_csv(p)
        dfi["__source_id"] = i
        dfi["__source_name"] = p.name
        dfs.append(dfi)
    df = pd.concat(dfs, ignore_index=True)
    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit].copy()

    required = ["Src IP", "Dst IP", "Timestamp", "Stage"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    stage = df["Stage"].astype(str)
    y_bin = (stage.str.lower() != "benign").astype(np.int64).to_numpy()
    y_t = torch.tensor(y_bin, dtype=torch.int64)

    t_sec = _parse_timestamp_to_unix_seconds(df["Timestamp"])

    feat_t, feat_cols = _get_flow_features(df, required_cols=required)

    # Optional: time-frequency augmentation for edge detection (ip graph)
    if bool(getattr(args, "tf_augment", False)):
        if args.graph_type != "ip":
            raise ValueError("--tf_augment currently supports only --graph_type ip")
        feat_t, tf_cols = _augment_ip_edges_with_timefreq(
            df,
            feat_t,
            t_sec,
            bin_sec=int(args.tf_bin_sec),
            nperseg=int(args.tf_nperseg),
            noverlap=int(args.tf_noverlap),
            wavelet=str(args.tf_wavelet),
            level=int(args.tf_level),
        )
        feat_cols = feat_cols + tf_cols

    if args.graph_type == "flow":
        g = _build_flow_node_graph(df, feat_t=feat_t, y_t=y_t, t_sec=t_sec, args=args)
    else:
        g = _build_ip_node_graph(df, feat_t=feat_t, y_t=y_t, t_sec=t_sec, args=args)

    # Persist stage_id -> Stage string mapping via save_graphs label_dict.
    # DGL's save_graphs may drop non-tensor objects, so we encode strings as bytes+offset tensors.
    label_dict = {}
    try:
        stage_s = df.get("Stage", pd.Series(["" for _ in range(len(df))])).fillna("").astype(str)
        stage_cat = pd.Categorical(stage_s)
        stage_id2name = [str(x) for x in stage_cat.categories.tolist()]
        enc = _encode_str_list_to_tensors(stage_id2name)
        label_dict["stage_id2name__bytes"] = enc["bytes"]
        label_dict["stage_id2name__offsets"] = enc["offsets"]
    except Exception:
        label_dict = {}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_graphs(str(out_path), [g], label_dict)

    print("Saved:", out_path)
    print("Sources:", [p.name for p in csv_paths])
    print("graph_type:", args.graph_type)
    print("Nodes:", g.num_nodes(), "Edges:", g.num_edges())
    if args.graph_type == "flow":
        print("Node feature dim:", int(g.ndata["feature"].shape[1]))
        pos = int(g.ndata["node_label"].sum().item())
        print("Node label: pos", pos, "neg", g.num_nodes() - pos)
    else:
        print("Edge feature dim:", int(g.edata["feature"].shape[1]))
        pos = int(g.edata["edge_label"].sum().item())
        print("Edge label: pos", pos, "neg", g.num_edges() - pos)
        print("Feature columns example:", feat_cols[: min(10, len(feat_cols))], "...")


if __name__ == "__main__":
    main()
