"""Entry point: `matrss run configs/baseline.yaml --out results/baseline`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .behaviors import Behavior, build_behavior
from .config import ExperimentConfig
from .engine import simulate
from .environment import Environment
from .metrics import bootstrap_ci, summarize
from .rng import make_streams
from .selection import build_selection
from .trust import build_trust


def build_providers(specs: list[dict]) -> list[Behavior]:
    behaviors: list[Behavior] = []
    for spec in specs:
        spec = dict(spec)
        kind, count = spec.pop("type"), spec.pop("count", 1)
        behaviors += [build_behavior(kind, **spec) for _ in range(count)]
    return behaviors


def run_experiment(cfg: ExperimentConfig, out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    behaviors = build_providers(cfg.providers)
    rows = []
    for seed in cfg.seeds:
        env_rng, _ = make_streams(seed)
        env = Environment(behaviors, cfg.n_clients, cfg.n_rounds, env_rng)   # shared across conditions (CRN)
        for cond in cfg.conditions:
            _, decision_rng = make_streams(seed)          # same fresh decision stream per condition
            trust = build_trust(cfg.n_clients, env.n_providers, **cond.trust)
            policy = build_selection(**cond.selection)
            res = simulate(cond.name, seed, env, trust, policy, decision_rng)
            rows.append({"condition": cond.name, "seed": seed,
                         **summarize(res, behaviors, cfg.isolation_threshold,
                                     cfg.isolation_window)})
            if seed == cfg.seeds[0]:
                cols = [f"{label}_{j}" for j, label in enumerate(res.labels)]
                pd.DataFrame(res.mean_trust, columns=cols).to_csv(
                    out / f"trust_trajectory_{cond.name}_seed{seed}.csv", index_label="round")
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_seed.csv", index=False)
    return df


def aggregate(df: pd.DataFrame, reference: str | None) -> pd.DataFrame:
    metrics = [c for c in df.columns if c not in ("condition", "seed")]
    recs = []
    for cond, g in df.groupby("condition", sort=False):
        for m in metrics:
            x = g[m].to_numpy()
            lo, hi = bootstrap_ci(x)
            rec = {"condition": cond, "metric": m, "mean": x.mean(), "ci_low": lo, "ci_high": hi}
            if reference and cond != reference:          # paired difference under CRN
                ref = df[df.condition == reference].set_index("seed")[m]
                d = (g.set_index("seed")[m] - ref.loc[g.seed]).to_numpy()
                dlo, dhi = bootstrap_ci(d)
                rec |= {"diff_vs_ref": d.mean(), "diff_ci_low": dlo, "diff_ci_high": dhi}
            recs.append(rec)
    return pd.DataFrame(recs)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="matrss")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run an experiment config")
    run.add_argument("config", type=Path)
    run.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = ExperimentConfig.load(args.config)
    out = args.out or Path("results") / cfg.name
    df = run_experiment(cfg, out)
    agg = aggregate(df, cfg.reference)
    agg.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 160, "display.max_columns", 20,
                           "display.float_format", "{:.4f}".format):
        print(agg.to_string(index=False))
    print(f"\nwrote {out}/per_seed.csv, summary.csv, trust trajectories")


if __name__ == "__main__":
    main()
