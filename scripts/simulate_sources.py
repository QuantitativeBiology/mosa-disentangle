"""Inject a simulated technical factor with ground truth into a dataset.

Writes ``<output>`` (observed, distorted) and ``<output stem>_truth.h5mu``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import limit_threads, parser, setup_logging  # noqa: E402

limit_threads()

from mosa_disentangle.batch_sim import BatchSimConfig, inject_source_effects  # noqa: E402
from mosa_disentangle.data import read_dataset, write_dataset  # noqa: E402


def truth_path(output: str | Path) -> Path:
    output = Path(output)
    return output.with_name(f"{output.stem}_truth{output.suffix}")


def main() -> None:
    setup_logging()
    p = parser(__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--confounding", type=float, default=0.5)
    p.add_argument("--shift-sd", type=float, default=0.8)
    p.add_argument("--scale-sd", type=float, default=0.2)
    p.add_argument("--programme-fraction", type=float, default=0.05)
    p.add_argument("--programme-shift", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    opts = p.parse_args()
    cfg = BatchSimConfig(
        confounding=opts.confounding,
        shift_sd=opts.shift_sd,
        scale_sd=opts.scale_sd,
        programme_fraction=opts.programme_fraction,
        programme_shift=opts.programme_shift,
        seed=opts.seed,
    )
    observed, truth = inject_source_effects(read_dataset(opts.input), cfg)
    write_dataset(observed, opts.output)
    write_dataset(truth, truth_path(opts.output))


if __name__ == "__main__":
    main()
