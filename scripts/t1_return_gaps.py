"""T1: how soon does a client come back to an on-off attacker after leaving it?

For each greedy and Thompson-sampling defender, re-simulates its worst on-off schedule and
records the gaps between a client's successive uses of the same attacker (gaps > 1 round, i.e.
returns after leaving). Relates exposure to this return delay and to the recovered provider's
readmission delay (technical-notes.md §11 R5).

Usage: python scripts/t1_return_gaps.py [results/t1_frontier]   (after scripts/t1_frontier.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from matrss.config import ExperimentConfig
from matrss.engine import simulate
from matrss.environment import Environment
from matrss.experiment import build_providers
from matrss.rng import make_streams
from matrss.selection import build_selection
from matrss.trust import build_trust

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/t1_frontier")
FAMILIES = ("add_greedy", "beta_greedy", "beta_ts")
SEEDS = range(10)


def return_gaps(cfg: ExperimentConfig, cond, scenario_name: str) -> np.ndarray:
    beh = build_providers(next(s for s in cfg.scenarios if s.name == scenario_name).providers)
    attackers = [j for j, b in enumerate(beh) if b.label == "onoff"]
    gaps: list[int] = []
    for seed in SEEDS:
        env_rng, dec_rng = make_streams(seed)
        env = Environment(beh, cfg.n_clients, cfg.n_rounds, env_rng)
        res = simulate(cond.name, seed, env, build_trust(cfg.n_clients, env.n_providers,
                                                         **cond.trust),
                       build_selection(**cond.selection), dec_rng)
        for client in range(cfg.n_clients):
            for j in attackers:
                used = np.flatnonzero(res.selections[:, client] == j)
                g = np.diff(used)
                gaps += list(g[g > 1])
    return np.array(gaps, dtype=float)


def main() -> None:
    cfg = ExperimentConfig.load("configs/t1_frontier.yaml")
    table = pd.read_csv(SRC / "frontier_table.csv", index_col="condition")
    conds = {c.name: c for c in cfg.conditions}
    rows = []
    for name in table.index[table.family.isin(FAMILIES)]:
        gaps = return_gaps(cfg, conds[name], f"onoff[{table.loc[name, 'worst_schedule']}]")
        rows.append({"condition": name, "family": table.loc[name, "family"],
                     "exposure_pct": 100 * table.loc[name, "onoff_worst"],
                     "recovery_delay": table.loc[name, "readmission"],
                     "return_gap_mean": gaps.mean() if gaps.size else np.nan,
                     "return_gap_median": np.median(gaps) if gaps.size else np.nan,
                     "n_returns": gaps.size})
    r = pd.DataFrame(rows).set_index("condition")
    r["exposure_x_gap"] = r.exposure_pct * r.return_gap_mean
    r["exposure_x_delay"] = r.exposure_pct * r.recovery_delay
    r.to_csv(SRC / "return_gaps.csv")
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print(r.to_string())
    for label, fams, col in (("greedy rules", ["add_greedy", "beta_greedy"], "return_gap_mean"),
                             ("Thompson sampling", ["beta_ts"], "recovery_delay")):
        g = r[r.family.isin(fams) & (r.n_returns > 20) & (r[col] < 1699)]
        slope = np.polyfit(np.log(g[col]), np.log(g.exposure_pct), 1)[0]
        corr = np.corrcoef(np.log(g[col]), np.log(g.exposure_pct))[0, 1]
        print(f"{label:<18} log exposure vs log {col}: slope {slope:+.2f}, r {corr:+.2f}, "
              f"n {len(g)}")
    print(f"wrote {SRC / 'return_gaps.csv'}")


if __name__ == "__main__":
    main()
