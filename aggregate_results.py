"""
aggregate_results.py
--------------------
Gộp kết quả nhiều seed của ``train_fedmarl.py`` thành bảng mean ± std.

Quy ước tên file (xem README/guide): seed 0 → ``a_<het|hom>_{base,fl}_summary.csv``,
seed k ≥ 1 → ``s<k>_<het|hom>_{base,fl}_summary.csv`` trong ``results/``.

Ví dụ:
    python aggregate_results.py            # in bảng + ghi results/multiseed_summary.csv
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SETTINGS = ("het", "hom")
PARTS = ("base", "fl")
METRICS = ("mean_queue", "worst_queue", "throughput", "violation_rate")
SEED_PREFIX = {0: "a"}          # seed 0 dùng tiền tố "a"; seed k dùng "s<k>"
MODE_ORDER = ("fixed_time", "greedy_queue", "independent", "fedavg", "fedprox")


def summary_path(seed: int, setting: str, part: str) -> Path:
    """Đường dẫn file summary của một seed / thiết lập / nhóm mode."""
    prefix = SEED_PREFIX.get(seed, f"s{seed}")
    return RESULTS_DIR / f"{prefix}_{setting}_{part}_summary.csv"


def load(seeds: list[int]) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Đọc mọi summary có sẵn → {(setting, mode): {metric: [giá trị theo seed]}}."""
    data: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for seed in seeds:
        for setting in SETTINGS:
            for part in PARTS:
                path = summary_path(seed, setting, part)
                if not path.exists():
                    continue
                with path.open(encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        for metric in METRICS:
                            data[(setting, row["mode"])][metric].append(float(row[metric]))
    return data


def main() -> None:
    """In bảng mean ± std theo mode và ghi ``multiseed_summary.csv``."""
    seeds = sorted(
        {0}
        | {int(p.name.split("_")[0][1:]) for p in RESULTS_DIR.glob("s[0-9]*_*_summary.csv")}
    )
    data = load(seeds)
    rows: list[dict] = []
    for setting in SETTINGS:
        print(f"\n=== {setting} (seeds: {seeds}) | mean_queue mean +/- std; lower is better ===")
        for mode in MODE_ORDER:
            m = data.get((setting, mode))
            if not m:
                continue
            q = np.array(m["mean_queue"])
            row = {"setting": setting, "mode": mode, "n_seeds": len(q)}
            for metric in METRICS:
                v = np.array(m[metric])
                row[f"{metric}_mean"] = float(v.mean())
                row[f"{metric}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            rows.append(row)
            print(
                f"{mode:13s} queue={row['mean_queue_mean']:7.1f} +/- {row['mean_queue_std']:5.1f}  "
                f"thr={row['throughput_mean']:6.0f}  viol={row['violation_rate_mean']:.3f}  (n={len(q)})"
            )
    if rows:
        out = RESULTS_DIR / "multiseed_summary.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote results/{out.name}")


if __name__ == "__main__":
    main()
