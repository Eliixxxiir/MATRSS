"""Entry point: `matrss run configs/baseline.yaml --out results/baseline [--jobs 0]`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .experiment import aggregate, build_providers, run_experiment

__all__ = ["aggregate", "build_providers", "main", "run_experiment"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="matrss")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run an experiment config")
    run.add_argument("config", type=Path)
    run.add_argument("--out", type=Path, default=None)
    run.add_argument("--jobs", type=int, default=1,
                     help="worker processes; 0 = one per CPU (results do not depend on it)")
    run.add_argument("--quiet", action="store_true", help="do not print the summary table")
    args = ap.parse_args(argv)

    cfg = ExperimentConfig.load(args.config)
    out = args.out or Path("results") / cfg.name
    df = run_experiment(cfg, out, jobs=args.jobs)
    agg = aggregate(df, cfg.reference, cfg.n_boot)
    agg.to_csv(out / "summary.csv", index=False)
    if not args.quiet:
        with pd.option_context("display.width", 160, "display.max_columns", 20,
                               "display.float_format", "{:.4f}".format):
            print(agg.to_string(index=False))
    print(f"\nwrote {out}/per_seed.csv, summary.csv"
          + (", trust trajectories" if cfg.trajectories else ""))


if __name__ == "__main__":
    main()
