"""Collect per-fold metrics into long and summary tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import limit_threads, parser, setup_logging  # noqa: E402

limit_threads()

import pandas as pd  # noqa: E402

KEYS = ["dataset", "variant", "fold"]


def collect(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.glob("*/*/fold*/metrics.json")):
        with open(path) as handle:
            rows.append(json.load(handle))
    if not rows:
        raise SystemExit(f"No metrics.json found under {run_dir}")
    table = pd.DataFrame(rows)
    # Runs evaluated before the divergence flag existed are finite by
    # construction (they would have crashed otherwise).
    if "diverged" not in table:
        table["diverged"] = 0
    table["diverged"] = table["diverged"].fillna(0).astype(int)
    return table


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    numeric = table.select_dtypes("number").columns.difference(["fold"])
    grouped = table.groupby(["dataset", "variant"])[list(numeric)]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def main() -> None:
    setup_logging()
    p = parser(__doc__)
    p.add_argument("--runs-dir", default="results/runs")
    p.add_argument("--out-dir", default="results/tables")
    opts = p.parse_args()
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = collect(Path(opts.runs_dir))
    table = table[KEYS + sorted(c for c in table.columns if c not in KEYS)]
    table.to_csv(out / "metrics_long.csv", index=False)
    summarise(table).to_csv(out / "metrics_summary.csv", index=False)
    print(table.groupby(["dataset", "variant"]).size().to_string())


if __name__ == "__main__":
    main()
