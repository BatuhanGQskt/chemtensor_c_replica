#!/usr/bin/env python3
"""Remove benchmark *.jsonl files that do not match active config files.

Keeps only:
  contraction_timings_{nsites}_{d}_{chi_max}.jsonl
  dmrg_singlesite_timings_{nsites}_{d}_{chi_max}_{num_sweeps}.jsonl
  dmrg_twosite_timings_{nsites}_{d}_{chi_max}_{num_sweeps}.jsonl
  profiling_{nsites}_{d}_{chi_max}.jsonl  (Mojo profiling output; same lattice as contractions)

Usage:
  clean_stale_perf_jsonl.py <perf_directory> <contraction_config.json> [dmrg_config.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "usage: clean_stale_perf_jsonl.py <perf_directory> <contraction_config.json> "
            "[dmrg_config.json]",
            file=sys.stderr,
        )
        return 2
    perf_dir = Path(sys.argv[1])
    config_paths = [Path(p) for p in sys.argv[2:]]
    if not perf_dir.is_dir():
        return 0
    if not any(p.is_file() for p in config_paths):
        print(
            "clean_stale_perf_jsonl: no config file found among: "
            + ", ".join(str(p) for p in config_paths),
            file=sys.stderr,
        )
        return 0
    keep: set[str] = set()
    for cfg in config_paths:
        if not cfg.is_file():
            continue
        try:
            with cfg.open(encoding="utf-8") as f:
                bench = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        ns = int(bench.get("nsites", 6))
        d = int(bench.get("d", 2))
        chi = int(bench.get("chi_max", 16))
        ss_ns = int(bench.get("dmrg_singlesite_nsites", ns))
        ss_d = int(bench.get("dmrg_singlesite_d", d))
        ss_chi = int(bench.get("dmrg_singlesite_chi_max", chi))
        ss_sw = int(bench.get("dmrg_singlesite_num_sweeps", 6))
        ts_ns = int(bench.get("dmrg_twosite_nsites", ns))
        ts_d = int(bench.get("dmrg_twosite_d", d))
        ts_chi = int(bench.get("dmrg_twosite_chi_max", chi))
        ts_sw = int(bench.get("dmrg_twosite_num_sweeps", 4))

        keep.update(
            {
                f"contraction_timings_{ns}_{d}_{chi}.jsonl",
                f"profiling_{ns}_{d}_{chi}.jsonl",
                f"dmrg_singlesite_timings_{ss_ns}_{ss_d}_{ss_chi}_{ss_sw}.jsonl",
                f"dmrg_twosite_timings_{ts_ns}_{ts_d}_{ts_chi}_{ts_sw}.jsonl",
            }
        )

    removed = 0
    for path in sorted(perf_dir.glob("*.jsonl")):
        if path.name not in keep:
            path.unlink()
            print(f"  removed stale perf file: {path.name}")
            removed += 1
    if removed:
        print(f"  ({removed} stale jsonl removed under {perf_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
