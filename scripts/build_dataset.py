"""Build a MOSA dataset (h5mu) from the figshare tables.

Example
-------
python scripts/build_dataset.py --name cell_lines --raw-dir data/raw/figshare_24420580 \
    --output data/processed/cell_lines.h5mu --top-variance transcriptomics=5000

``proteomics_paired`` also writes ``<stem>_truth.h5mu`` with the partner
laboratory's measurement of each cell line.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import limit_threads, parser, setup_logging  # noqa: E402

limit_threads()

from mosa_disentangle import data  # noqa: E402

BUILDERS = ("cell_lines", "proteomics_paired")


def parse_top_variance(items: list[str]) -> dict[str, int]:
    return {k: int(v) for k, v in (item.split("=") for item in items)}


def main() -> None:
    setup_logging()
    p = parser(__doc__)
    p.add_argument("--name", choices=sorted(BUILDERS), required=True)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-views", type=int, default=2)
    p.add_argument(
        "--top-variance",
        nargs="*",
        default=[],
        help="view=N pairs keeping the N most variable features",
    )
    opts = p.parse_args()
    top = parse_top_variance(opts.top_variance)
    if opts.name == "cell_lines":
        dataset = data.build_cell_lines(
            opts.raw_dir, top_variance=top, min_views=opts.min_views
        )
        data.write_dataset(dataset, opts.output)
        return
    dataset, truth = data.build_proteomics_paired(
        opts.raw_dir, top_variance=top.get("proteomics")
    )
    data.write_dataset(dataset, opts.output)
    output = Path(opts.output)
    data.write_dataset(truth, output.with_name(f"{output.stem}_truth{output.suffix}"))


if __name__ == "__main__":
    main()
