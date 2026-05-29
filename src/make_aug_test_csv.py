"""Augment CICFlowMeter-style CSVs by oversampling anomalous rows.

This is intentionally simple: it *copies* anomalous rows until the overall
positive rate reaches a target ratio (e.g. 5%).

It writes new CSV files and never overwrites originals.

Example (PowerShell):
    python src\make_aug_test_csv.py --target_pos 0.05 --suffix aug5p --files a.csv b.csv

Notes:
- Label column is auto-detected by case-insensitive name == 'label'.
- If label values are strings: anything != 'BENIGN' is treated as anomaly.
- If label values are numeric: >0 is treated as anomaly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _detect_label_col(df: pd.DataFrame) -> str:
    # Common variants across flow exporters
    candidates = [
        "label",
        "Label",
        "class",
        "Class",
        "attack",
        "Attack",
        "activity",
        "Activity",
        "stage",
        "Stage",
    ]
    cols_lut = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        k = str(cand).strip().lower()
        if k in cols_lut:
            return cols_lut[k]
    raise ValueError(
        "No label-like column found. Tried %s. Columns=%s"
        % (candidates, list(df.columns)[:30])
    )


def _is_pos(series: pd.Series) -> np.ndarray:
    if series.dtype == object:
        s = series.astype(str).str.strip().str.upper()
        # Common benign tokens
        benign = {"BENIGN", "NORMAL", "BENIGN ", "NORMAL "}
        return (~s.isin(benign)).to_numpy()
    # numeric-ish
    y = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    return (y > 0).to_numpy()


def augment_file(in_fp: Path, target_pos: float, suffix: str) -> Path:
    df = pd.read_csv(in_fp)
    label_col = _detect_label_col(df)
    pos_mask = _is_pos(df[label_col])

    n_total = int(df.shape[0])
    n_pos = int(pos_mask.sum())
    if n_pos <= 0:
        raise ValueError(f"No positive/anomalous rows found in {in_fp} (label_col={label_col}).")

    # how many positives we want in the ORIGINAL base-size
    desired_pos = int(np.ceil(float(target_pos) * n_total))
    if desired_pos <= n_pos:
        out_fp = in_fp.with_name(in_fp.stem + f".{suffix}" + in_fp.suffix)
        df.to_csv(out_fp, index=False)
        print(f"[skip] {in_fp.name}: pos_rate={n_pos/n_total:.4f} already >= target={target_pos:.3f} -> {out_fp.name}")
        return out_fp

    add = desired_pos - n_pos
    pos_df = df.loc[pos_mask]
    reps = int(np.ceil(add / n_pos))
    pos_rep = pd.concat([pos_df] * reps, ignore_index=True).iloc[:add]
    out = pd.concat([df, pos_rep], ignore_index=True)

    out_fp = in_fp.with_name(in_fp.stem + f".{suffix}" + in_fp.suffix)
    out.to_csv(out_fp, index=False)

    new_total = int(out.shape[0])
    new_pos = n_pos + add
    print(
        f"[aug] {in_fp.name}: label_col={label_col} n={n_total} pos={n_pos} "
        f"-> n={new_total} pos={new_pos} pos_rate={new_pos/new_total:.4f} (target~{target_pos:.3f}) -> {out_fp.name}"
    )
    return out_fp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_pos", type=float, default=0.05, help="Target positive ratio (e.g., 0.05 for 5%).")
    ap.add_argument("--suffix", type=str, default="aug", help="Suffix inserted before .csv")
    ap.add_argument("--files", nargs="+", required=True, help="One or more input CSV files")
    args = ap.parse_args()

    for f in args.files:
        augment_file(Path(f), float(args.target_pos), str(args.suffix))


if __name__ == "__main__":
    main()
