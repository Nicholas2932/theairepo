"""Validate the artifacts on disk without re-running anything (rules V-01..V-13).

    python validate.py
    python validate.py --artifacts out_dir --kb other_kb --queries other.json

Exit code 0 iff every rule passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag import config
from rag.validation import validate_artifacts


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifacts", type=Path, default=config.DEFAULT_ARTIFACTS_DIR)
    p.add_argument("--kb", type=Path, default=config.DEFAULT_KB_DIR)
    p.add_argument("--queries", type=Path, default=config.DEFAULT_QUERIES_PATH)
    args = p.parse_args(argv)

    report = validate_artifacts(args.artifacts, args.kb, args.queries)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
