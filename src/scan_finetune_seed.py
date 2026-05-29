"""Scan finetune_seed to find a desired graph-split.

We avoid PowerShell quoting issues by putting the scan logic in a .py file.

Usage (example):
  python src/scan_finetune_seed.py --want_test dataset0209-public-thursday-ip-els

The script runs a light config (epoch_ft=0, epoch_pretrain=1) just to print split.
"""

from __future__ import annotations

import argparse
import random
import sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--want_test", type=str, required=True)
    p.add_argument("--max_seed", type=int, default=200)
    p.add_argument("--print_first", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Keep consistent with the command you run in train_flow_ip_split.py.
    monday = "dataset0209-monday-ip-els"
    others = [
        "dataset0209-public-thursday-ip-els",
        "dataset0209-public-tuesday-ip-els",
        "dataset0209-public-wednesday-ip-els",
        "dataset0209-tcpdump-friday-ip-els",
    ]

    # Assumption (matches current default behavior in train_flow_ip_split.py for graph split):
    # - train = monday + 2 graphs
    # - val = 1 graph
    # - test = 1 graph
    # This is the only feasible 3-way split for 4 other graphs.
    def split_by_seed(seed: int):
        r = random.Random(int(seed))
        o = others.copy()
        r.shuffle(o)
        # order: [g0, g1, g2, g3]
        train = [monday, o[0], o[1]]
        val = [o[2]]
        test = [o[3]]
        return train, val, test

    want_test = str(args.want_test).strip()

    for s in range(1, int(args.max_seed) + 1):
        _, val, test = split_by_seed(s)
        v0 = val[0] if val else "N/A"
        t0 = test[0] if test else "N/A"
        if s <= int(args.print_first):
            print(f"seed={s} val={v0} test={t0}")
        if t0 == want_test:
            print(f"FOUND seed={s} val={v0} test={t0}")
            return

    print("NOT_FOUND")


if __name__ == "__main__":
    main()
