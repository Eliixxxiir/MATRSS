"""T2 analysis: why the saturating additive rule beats Thompson sampling with forgetting.

1. Exact long-run behaviour of the v1 rule + greedy (Markov chain) vs plain win-stay/lose-shift
   and vs simulation, for small stationary problems.
2. Exact finite-horizon share of the best provider (lock-in speed) vs Thompson sampling.
3. Steady-state regret of Thompson sampling with per-interaction forgetting as a function of
   lambda: bounded evidence keeps TS exploring forever, so its regret grows linearly.

Usage: python scripts/t2_mechanism.py [out_dir] [--plot-only]   (default results/t2_mechanism)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from matrss.behaviors import Stationary
from matrss.engine import simulate
from matrss.environment import Environment
from matrss.markov import AdditiveChain, wsls_shares
from matrss.rng import make_streams
from matrss.selection import Greedy, ThompsonSampling
from matrss.trust import AsymmetricAdditive, BetaReputation

_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(_ARGS[0]) if _ARGS else Path("results/t2_mechanism")
PROBLEMS = {                      # stationary success probabilities, best first
    "0.95 vs 0.80": [0.95, 0.80],
    "0.95 vs 0.90": [0.95, 0.90],
    "0.95 vs 3x0.80": [0.95, 0.80, 0.80, 0.80],
    "0.95, 0.80, 0.50": [0.95, 0.80, 0.50],
    "0.80 vs 0.70": [0.80, 0.70],
    "harsh 0.60 vs 0.30": [0.60, 0.30],        # below the 2/3 break-even
}


def simulate_shares(p, make_trust, policy, n_clients=2000, n_rounds=3000, seed=0):
    """Per-round share of each provider across many independent clients, shape (T, K)."""
    beh = [Stationary(f"p{j}", x) for j, x in enumerate(p)]
    env_rng, rng = make_streams(seed)
    env = Environment(beh, n_clients, n_rounds, env_rng)
    res = simulate("x", seed, env, make_trust(n_clients, len(p)), policy, rng)
    return np.stack([(res.selections == j).mean(axis=1) for j in range(len(p))], axis=1)


def long_run_table() -> pd.DataFrame:
    rows = []
    for name, p in PROBLEMS.items():
        p_arr = np.array(p)
        chain = AdditiveChain.from_rule(p)
        exact = chain.long_run_shares() if chain.n_states <= 6000 else None
        horizon = chain.transient(1000).mean(axis=0)                   # exact, 1000 rounds
        sim = simulate_shares(p, lambda c, k: AsymmetricAdditive(c, k), Greedy(),
                              n_rounds=1000).mean(axis=0)
        ts = {lam: simulate_shares(p, lambda c, k, lam=lam: BetaReputation(c, k, forgetting=lam),
                                   ThompsonSampling(), n_rounds=1000).mean(axis=0)
              for lam in (0.9, 1.0)}
        rows.append({
            "problem": name,
            "v1_exact_longrun_best%": 100 * exact[0] if exact is not None else np.nan,
            "wsls_longrun_best%": 100 * wsls_shares(p)[0],
            "v1_exact_T1000_best%": 100 * horizon[0],
            "v1_sim_T1000_best%": 100 * sim[0],
            "ts0.9_sim_T1000_best%": 100 * ts[0.9][0],
            "ts1.0_sim_T1000_best%": 100 * ts[1.0][0],
            "v1_T1000_regret": float(p_arr[0] - horizon @ p_arr),
            "ts0.9_T1000_regret": float(p_arr[0] - ts[0.9] @ p_arr),
            "ts1.0_T1000_regret": float(p_arr[0] - ts[1.0] @ p_arr),
        })
    return pd.DataFrame(rows)


def lock_in_curves(p=(0.95, 0.80, 0.80, 0.80), n_rounds=2000) -> pd.DataFrame:
    """Share of the best provider per round: v1 exact vs Thompson sampling (simulated)."""
    curves = {"v1_exact": AdditiveChain.from_rule(list(p)).transient(n_rounds)[:, 0]}
    for lam in (0.9, 0.98, 1.0):
        curves[f"ts_lambda{lam}"] = simulate_shares(
            p, lambda c, k, lam=lam: BetaReputation(c, k, forgetting=lam), ThompsonSampling(),
            n_rounds=n_rounds)[:, 0]
    return pd.DataFrame(curves).rename_axis("round")


def steady_state_regret(p=(0.95, 0.80), lams=(0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0),
                        n_rounds=6000, tail=2000) -> pd.DataFrame:
    """Per-round regret over the last `tail` rounds (TS: simulated; v1: exact Markov chain)."""
    p_arr = np.array(p)
    rows = []
    for lam in lams:
        sh = simulate_shares(p, lambda c, k, lam=lam: BetaReputation(c, k, forgetting=lam),
                             ThompsonSampling(), n_rounds=n_rounds)[-tail:].mean(axis=0)
        n_eff = np.inf if lam == 1.0 else 1 / (1 - lam)
        rows.append({"policy": "thompson", "lambda": lam, "effective_memory": n_eff,
                     "regret_per_round": float(p_arr[0] - sh @ p_arr)})
    chain = AdditiveChain.from_rule(list(p))
    tail_shares = chain.transient(n_rounds)[-tail:].mean(axis=0)
    rows.append({"policy": "v1_additive", "lambda": np.nan, "effective_memory": np.nan,
                 "regret_per_round": float(p_arr[0] - tail_shares @ p_arr)})
    return pd.DataFrame(rows)


def plot(out: Path) -> None:
    """SVG figures from the CSVs written by main(): lock-in curves and forgetting cost."""
    from figstyle import BLUE_RAMP, SERIES
    from svgplot import Figure

    ramp = {"0.9": BLUE_RAMP[0], "0.98": BLUE_RAMP[2], "1.0": BLUE_RAMP[4]}  # more memory = darker
    curves = pd.read_csv(out / "lock_in_curves.csv", index_col="round")
    rounds = curves.index.to_numpy() + 1
    fig = Figure(380, 250)
    ax = fig.axes((58, 18, 300, 180), (1, len(curves)), (0, 1), xscale="log",
                  xlabel="round (log scale)", ylabel="share to the best provider",
                  yticks=[0, 0.25, 0.5, 0.75, 1.0])
    for lam, color in ramp.items():
        ax.line(rounds, curves[f"ts_lambda{lam}"], color, label=f"Thompson sampling, λ = {lam}")
    ax.line(rounds, curves["v1_exact"], SERIES[1], label="v1 rule (exact Markov chain)")
    fig.legend(ax.legend_items, 200, 145)
    fig.save(out / "lock_in.svg")

    reg = pd.read_csv(out / "steady_state_regret.csv")
    ts = reg[(reg.policy == "thompson") & np.isfinite(reg.effective_memory)]
    ts = ts.sort_values("effective_memory")
    rule = float(reg.loc[reg.policy == "v1_additive", "regret_per_round"].iloc[0])
    fig = Figure(380, 250)
    ax = fig.axes((58, 18, 300, 180), (2, 2000), (1e-4, 0.1), xscale="log", yscale="log",
                  xlabel="remembered interactions, 1 / (1 − λ)",
                  ylabel="steady-state regret / round",
                  yticklabels=["0.0001", "0.001", "0.01", "0.1"])
    ax.hline(rule, SERIES[1], label="v1 rule (exact)")
    ax.line(ts.effective_memory, ts.regret_per_round, SERIES[0])
    ax.points(ts.effective_memory, ts.regret_per_round, SERIES[0],
              label="Thompson sampling with forgetting λ",
              titles=[f"λ = {lam}: {r:.4f}" for lam, r in zip(ts["lambda"], ts.regret_per_round)])
    fig.legend(ax.legend_items, 150, 30)
    fig.save(out / "forgetting_cost.svg")
    print(f"figures in {out}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--plot-only" not in sys.argv:
        table = long_run_table()
        table.to_csv(OUT / "long_run.csv", index=False)
        curves = lock_in_curves()
        curves.to_csv(OUT / "lock_in_curves.csv")
        regret = steady_state_regret()
        regret.to_csv(OUT / "steady_state_regret.csv", index=False)
        with pd.option_context("display.width", 220, "display.max_columns", 20,
                               "display.float_format", "{:.4f}".format):
            print("=== best-provider share and regret, 1000 rounds from init 0.5 ===")
            print(table.to_string(index=False))
            print("\n=== share of the best provider at selected rounds (0.95 vs 3 x 0.80) ===")
            print(curves.iloc[[0, 9, 49, 99, 249, 499, 999, 1999]].to_string())
            print("\n=== steady-state regret per round, 0.95 vs 0.80 (rounds 4000-6000) ===")
            print(regret.to_string(index=False))
    plot(OUT)


if __name__ == "__main__":
    main()
