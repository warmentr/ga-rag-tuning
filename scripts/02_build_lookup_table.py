"""Step 2: score every valid config on tuning and test questions -> lookup table.

The one expensive step (~30-40 min on an RTX 2080 Ti, mostly embedding and
reranking). Resumable: finished chunkings are skipped on re-run.

  .venv/Scripts/python scripts/02_build_lookup_table.py            # full grid
  .venv/Scripts/python scripts/02_build_lookup_table.py --only 512/0 128/0   # subset, no merge
  .venv/Scripts/python scripts/02_build_lookup_table.py --force    # rebuild all parts
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragtune.grid import build_lookup_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="SIZE/OVERLAP", help="build only these chunkings")
    ap.add_argument("--force", action="store_true", help="rebuild parts even if up to date")
    ap.add_argument("--out", default="results/lookup_table", help="output directory")
    args = ap.parse_args()

    only = [tuple(int(x) for x in s.split("/")) for s in args.only] if args.only else None
    meta = build_lookup_table(ROOT, ROOT / args.out, only_chunkings=only, force=args.force,
                              log=lambda m: print(m, flush=True))
    if meta:
        s, c = meta["summary"], meta["compute"]
        wh = f"{c['gpu_wh_total']:.1f} Wh" if c["gpu_wh_total"] is not None else "n/a"
        print(f"Best (tuning): {s['best_tuning']}")
        print(f"Grid compute: {c['grid_seconds_total'] / 60:.1f} min, GPU energy {wh}")


if __name__ == "__main__":
    main()
