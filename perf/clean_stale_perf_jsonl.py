#!/usr/bin/env python3
"""Remove benchmark *.jsonl files that do not match ../../bench_config.json (from repo root).

Keeps only:
  contraction_timings_{nsites}_{d}_{chi_max}.jsonl
  dmrg_singlesite_timings_{...}.jsonl
  dmrg_twosite_timings_{...}.jsonl
  profiling_{nsites}_{d}_{chi_max}.jsonl  (Mojo profiling output; same lattice as contractions)

Usage:
  clean_stale_perf_jsonl.py <perf_directory> <path_to_bench_config.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: clean_stale_perf_jsonl.py <perf_directory> <bench_config.json>", file=sys.stderr)
        return 2
    perf_dir = Path(sys.argv[1])
    bench_path = Path(sys.argv[2])
    remove_files = sys.argv[3] if len(sys.argv) > 3 else False
    if remove_files:
        print("Removing files...")
    else:
        print("Not removing files...")
    if not perf_dir.is_dir():
        return 0
    if not bench_path.is_file():
        print(f"clean_stale_perf_jsonl: no config at {bench_path}, skip", file=sys.stderr)
        return 0
    with bench_path.open(encoding="utf-8") as f:
        bench = json.load(f)

    ns = int(bench.get("nsites", 6))
    d = int(bench.get("d", 2))
    chi = int(bench.get("chi_max", 16))
    ss_ns = int(bench.get("dmrg_singlesite_nsites", ns))
    ss_d = int(bench.get("dmrg_singlesite_d", d))
    ss_chi = int(bench.get("dmrg_singlesite_chi_max", chi))
    ts_ns = int(bench.get("dmrg_twosite_nsites", ns))
    ts_d = int(bench.get("dmrg_twosite_d", d))
    ts_chi = int(bench.get("dmrg_twosite_chi_max", chi))

    keep = {
        f"contraction_timings_{ns}_{d}_{chi}.jsonl",
        f"dmrg_singlesite_timings_{ss_ns}_{ss_d}_{ss_chi}.jsonl",
        f"dmrg_twosite_timings_{ts_ns}_{ts_d}_{ts_chi}.jsonl",
        f"profiling_{ns}_{d}_{chi}.jsonl",
    }

    removed = 0
    for path in sorted(perf_dir.glob("*.jsonl")):
        if path.name not in keep:
            if remove_files:
                path.unlink()
                print(f"  removed stale perf file: {path.name}")
            else:
                print(f"  kept stale perf file: {path.name}")
            removed += 1
    if removed:
        print(f"  ({removed} stale jsonl removed under {perf_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
