"""Step 1: download WixQA (if missing) and write the tuning/test split.

Run from the repo root:  .venv/Scripts/python scripts/01_prepare_data.py
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragtune.data import build_splits, download_raw, save_splits  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text(encoding="utf-8"))
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    splits_path = ROOT / cfg["paths"]["splits"]

    download_raw(raw_dir)

    splits = build_splits(raw_dir, cfg["split"]["tuning_ratio"], cfg["split"]["seed"])
    save_splits(splits, splits_path)
    print(f"Wrote {splits_path.relative_to(ROOT)}: {splits['counts']}")


if __name__ == "__main__":
    main()
