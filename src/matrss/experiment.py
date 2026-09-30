"""Run an experiment config: scenarios x conditions x seeds, then aggregate with bootstrap CIs.

Within a (scenario, seed) cell every condition faces the same pre-drawn environment (CRN), so
conditions can be compared as paired samples. Cells are independent, so they can run in
worker processes; results are collected in a fixed order, independent of the worker count.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import pandas as pd

from .behaviors import Behavior, build_behavior
from .config import ExperimentConfig, Scenario
from .engine import simulate
from .environment import Environment
from .metrics import bootstrap_ci, summarize
from .rng import make_streams
from .selection import build_selection
from .trust import build_trust


def build_providers(specs: list[dict]) -> list[Behavior]:
    """`stagger: true` spreads the phases of `count` on-off providers evenly over their cycle."""
    behaviors: list[Behavior] = []
    for spec in specs:
        spec = dict(spec)
        kind, count = spec.pop("type"), spec.pop("count", 1)
        stagger = spec.pop("stagger", False)
        if stagger and kind != "onoff":
            raise ValueError("stagger only applies to onoff providers")
        for k in range(count):
            phase = {"phase": k / count} if stagger else {}
            behaviors.append(build_behavior(kind, **spec, **phase))
    return behaviors


def run_cell(cfg: ExperimentConfig, scenario: Scenario,
             seed: int) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """All conditions of one (scenario, seed): metric rows and, for the first seed, trajectories."""
    behaviors = build_providers(scenario.providers)
    env_rng, _ = make_streams(seed)
    env = Environment(behaviors, cfg.n_clients, cfg.n_rounds, env_rng)   # shared by conditions
    rows, trajectories = [], {}
    for cond in cfg.conditions:
        _, decision_rng = make_streams(seed)          # same fresh decision stream per condition
        trust = build_trust(cfg.n_clients, env.n_providers, **cond.trust)
        policy = build_selection(**cond.selection)
        res = simulate(cond.name, seed, env, trust, policy, decision_rng)
        rows.append({"scenario": scenario.name, "condition": cond.name, "seed": seed,
                     **summarize(res, behaviors, cfg.isolation_threshold, cfg.isolation_window,
                                 cfg.readmission_level)})
        if cfg.trajectories and seed == cfg.seeds[0]:
            cols = [f"{label}_{j}" for j, label in enumerate(res.labels)]
            prefix = "" if len(cfg.scenarios) == 1 else f"{scenario.name}_"
            trajectories[f"{prefix}{cond.name}_seed{seed}"] = pd.DataFrame(res.mean_trust,
                                                                           columns=cols)
    return rows, trajectories


def run_experiment(cfg: ExperimentConfig, out: Path, jobs: int = 1) -> pd.DataFrame:
    """Run every cell (jobs > 1: in that many processes; 0: one per CPU) and write per_seed.csv."""
    out.mkdir(parents=True, exist_ok=True)
    cells = [(sc, seed) for sc in cfg.scenarios for seed in cfg.seeds]
    jobs = jobs or os.cpu_count() or 1
    if jobs == 1 or len(cells) == 1:
        results = [run_cell(cfg, sc, seed) for sc, seed in cells]
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(cells))) as pool:
            results = list(pool.map(run_cell, repeat(cfg), *zip(*cells),
                                    chunksize=max(1, len(cells) // (4 * jobs))))
    rows = [row for cell_rows, _ in results for row in cell_rows]
    for _, trajectories in results:
        for key, frame in trajectories.items():
            frame.to_csv(out / f"trust_trajectory_{key}.csv", index_label="round")
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_seed.csv", index=False)
    return df


def aggregate(df: pd.DataFrame, reference: str | None, n_boot: int = 10_000) -> pd.DataFrame:
    """Mean and bootstrap CI per (scenario, condition, metric); paired differences vs reference."""
    keys = ["scenario", "condition", "seed"]
    metrics = [c for c in df.columns if c not in keys]
    recs = []
    for (scen, cond), g in df.groupby(["scenario", "condition"], sort=False):
        ref_rows = df[(df.scenario == scen) & (df.condition == reference)].set_index("seed")
        for m in metrics:
            x = g[m].dropna().to_numpy()
            if not x.size:                       # metric not defined in this scenario
                continue
            lo, hi = bootstrap_ci(x, n_boot)
            rec = {"scenario": scen, "condition": cond, "metric": m, "mean": x.mean(),
                   "ci_low": lo, "ci_high": hi}
            if reference and cond != reference:          # paired difference under CRN
                d = (g.set_index("seed")[m] - ref_rows[m].loc[g.seed]).dropna().to_numpy()
                if d.size:
                    dlo, dhi = bootstrap_ci(d, n_boot)
                    rec |= {"diff_vs_ref": d.mean(), "diff_ci_low": dlo, "diff_ci_high": dhi}
            recs.append(rec)
    return pd.DataFrame(recs)
