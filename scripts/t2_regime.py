"""T2 regime map: where does the v1 rule beat (best-tuned) Thompson sampling?

Cells: one best provider at p_best, four at p_best - gap, five malicious (0.05). Variant
"sleepers" adds two providers slightly better than the best (p_best + 0.03, capped at 0.99)
that turn bad (0.05) at round 300. Per cell, the v1 rule (additive R=0.1, P=0.2, greedy) is
compared against Thompson sampling with its forgetting factor tuned for that cell in hindsight
(the minimum regret over lambda), which favours TS. Negative delta = v1 has lower regret.

Usage: python scripts/t2_regime.py [out_dir] [--jobs N]     (default results/t2_regime)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from matrss.config import Condition, ExperimentConfig, Scenario
from matrss.experiment import run_experiment
from matrss.metrics import bootstrap_ci

P_BEST = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
GAPS = [0.02, 0.05, 0.10, 0.20, 0.30]
LAMBDAS = [0.9, 0.95, 0.98, 0.995, 1.0]


def scenarios() -> list[Scenario]:
    out = []
    for variant in ("stationary", "sleepers"):
        for pb in P_BEST:
            for gap in GAPS:
                providers = [{"type": "honest", "count": 1, "p": pb},
                             {"type": "noisy", "count": 4, "p": round(pb - gap, 4)},
                             {"type": "malicious", "count": 5, "p": 0.05}]
                if variant == "sleepers":
                    providers.append({"type": "degrading", "count": 2,
                                      "p_start": min(0.99, pb + 0.03), "p_end": 0.05,
                                      "onset": 300, "rate": float("inf")})
                out.append(Scenario(f"{variant}|{pb}|{gap}", providers))
    return out


def conditions() -> list[Condition]:
    conds = [Condition("v1", {"type": "asymmetric", "reward": 0.1, "penalty": 0.2},
                       {"type": "greedy"})]
    conds += [Condition(f"ts[{lam}]", {"type": "beta", "forgetting": lam}, {"type": "thompson"})
              for lam in LAMBDAS]
    return conds


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[["variant", "p_best", "gap"]] = df.scenario.str.split("|", expand=True)
    df[["p_best", "gap"]] = df[["p_best", "gap"]].astype(float)
    rows = []
    for (variant, pb, gap, scen), g in df.groupby(["variant", "p_best", "gap", "scenario"]):
        reg = g.pivot(index="seed", columns="condition", values="mean_regret_per_task")
        ts_cols = [c for c in reg.columns if c.startswith("ts[")]
        best_ts = reg[ts_cols].mean().idxmin()                  # tuned in hindsight, per cell
        delta = (reg["v1"] - reg[best_ts]).to_numpy()           # paired under CRN
        lo, hi = bootstrap_ci(delta, 2000)
        rows.append({"variant": variant, "p_best": pb, "gap": gap, "v1_regret": reg["v1"].mean(),
                     "best_ts": best_ts, "best_ts_regret": reg[best_ts].mean(),
                     "delta": delta.mean(), "delta_ci_low": lo, "delta_ci_high": hi})
    out = pd.DataFrame(rows)
    out["winner"] = np.select([out.delta_ci_high < 0, out.delta_ci_low > 0],
                              ["v1", "TS"], "tie")
    return out


def plot(table: pd.DataFrame, out: Path) -> None:
    """Two heatmaps (stationary, sleepers): regret(v1) - regret(best-tuned TS), diverging."""
    from figstyle import INK, INK_2, SURFACE, diverging
    from svgplot import Figure

    scale = 0.05              # colour saturates at a regret difference of +-0.05 per task
    fig = Figure(560, 318)
    for k, (variant, title) in enumerate((("stationary", "stationary providers"),
                                          ("sleepers", "with two sleepers"))):
        g = table[table.variant == variant]
        ax = fig.axes((70 + k * 262, 26, 200, 216), (0, len(GAPS)), (0, len(P_BEST)),
                      title=title, xlabel="gap to the other providers",
                      ylabel="best provider's p" if k == 0 else "", grid=False,
                      xticks=[i + 0.5 for i in range(len(GAPS))],
                      xticklabels=[f"{x:g}" for x in GAPS],
                      yticks=[i + 0.5 for i in range(len(P_BEST))],
                      yticklabels=[f"{x:.2f}" for x in P_BEST])
        for _, row in g.iterrows():
            i, j = GAPS.index(row.gap), P_BEST.index(row.p_best)
            t = max(-1.0, min(1.0, row.delta / scale))
            ax.rect(i, j, i + 1, j + 1, diverging(t),
                    title=f"p={row.p_best}, gap={row.gap}: {1000 * row.delta:+.1f} x1e-3 "
                          f"(best TS {row.best_ts})")
            mark = "" if row.winner == "tie" else "*"
            ax.text(i + 0.5, j + 0.5, f"{1000 * row.delta:+.0f}{mark}", anchor="middle",
                    color=SURFACE if abs(t) > 0.6 else INK, size=9, dy=3)  # white on deep fills
        y_be = P_BEST.index(0.65) + 1                           # break-even 2/3 lies between rows
        ax.line((0, len(GAPS)), (y_be, y_be), INK, width=1.5)
        x_end, y_px = ax.px(len(GAPS), y_be)
        fig.extra.append(f'<text x="{x_end + 4:.1f}" y="{y_px + 3.5:.1f}" font-size="9" '
                         f'fill="{INK_2}">2/3</text>')
    # colour key, then the note on its own line
    for n, t in enumerate(np.linspace(-1, 1, 9)):
        fig.extra.append(f'<rect x="{200 + n * 22}" y="282" width="21" height="8" '
                         f'fill="{diverging(t)}"/>')
    fig.extra.append(f'<text x="195" y="289" text-anchor="end" font-size="10" fill="{INK_2}">'
                     f'v1 rule better</text>')
    fig.extra.append(f'<text x="403" y="289" font-size="10" fill="{INK_2}">TS better</text>')
    fig.extra.append(f'<text x="299" y="307" text-anchor="middle" font-size="10" '
                     f'fill="{INK_2}">cells: regret(v1) − regret(best-tuned TS), x1000; '
                     f'* = 95% CI excludes 0; line: v1 break-even 2/3</text>')
    fig.save(out / "regime_map.svg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", type=Path, default=Path("results/t2_regime"))
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        plot(pd.read_csv(args.out / "regime_table.csv"), args.out)
        return
    cfg = ExperimentConfig(name="t2_regime", n_rounds=1000, n_clients=20, seeds=list(range(20)),
                           scenarios=scenarios(), conditions=conditions(), trajectories=False)
    df = run_experiment(cfg, args.out, jobs=args.jobs)
    table = summarize(df)
    table.to_csv(args.out / "regime_table.csv", index=False)
    for variant, g in table.groupby("variant"):
        print(f"\n=== {variant}: regret(v1) - regret(best-tuned TS), x1000 "
              "(negative = v1 better; * = 95% CI excludes 0) ===")
        cell = g.assign(txt=[f"{1000 * d:+.1f}{'*' if w != 'tie' else ''}"
                             for d, w in zip(g.delta, g.winner)])
        print(cell.pivot(index="p_best", columns="gap", values="txt").iloc[::-1].to_string())
        print("tuned lambda per cell:")
        print(g.pivot(index="p_best", columns="gap", values="best_ts").iloc[::-1].to_string())
    plot(table, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
