"""T1 analysis: the recovery-manipulation frontier.

Reads results/t1_frontier/per_seed.csv (from `matrss run configs/t1_frontier.yaml`) and, per
defender, combines its scenarios into one row:

  sleeper_exposure   share of tasks sent to a sleeper after it turned bad        (lower better)
  recovery_share     share of post-recovery tasks sent to the recovered providers (higher better)
  readmission        rounds until recovered providers get half the traffic       (lower better)
  onoff_worst        share of tasks sent to on-off attackers in a bad phase, for the attacker's
                     best-response schedule (the worst over the grid)             (lower better)
  stationary_regret  per-task regret in a benign world                           (lower better)

Pareto-optimal = not dominated on (recovery_share, onoff_worst, sleeper_exposure).
Usage: python scripts/t1_frontier.py [results/t1_frontier]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from matrss.metrics import bootstrap_ci

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/t1_frontier")


def family(condition: str) -> str:
    """Defender family: the condition name without its grid tag, CUSUM split by sidedness."""
    base = condition.split("[", 1)[0]
    if base.startswith("cusum"):
        return f"{base}_{'down' if 'sides=down' in condition else 'both'}"
    return base


def per_defender(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per defender, plus the per-seed values behind each headline metric."""
    by = df.groupby(["scenario", "condition"], sort=False)

    def metric(scenario: str, column: str) -> pd.Series:
        return by[column].mean().xs(scenario, level="scenario")

    onoff = df[df.scenario.str.startswith("onoff")]
    onoff_means = onoff.groupby(["condition", "scenario"]).onoff_exposure_rate.mean()
    worst = onoff_means.groupby(level="condition").idxmax().map(lambda ix: ix[1])

    table = pd.DataFrame({
        "sleeper_exposure": metric("sleeper", "degraded_exposure_rate"),
        "sleeper_adaptation": metric("sleeper", "degraded_adaptation_rounds"),
        "recovery_share": metric("recovering", "recovered_share"),
        "readmission": metric("recovering", "readmission_rounds"),
        "stationary_regret": metric("stationary", "mean_regret_per_task"),
        "onoff_worst": onoff_means.groupby(level="condition").max(),
        "worst_schedule": worst.str.extract(r"\[(.*)\]")[0],
    })
    table.insert(0, "family", [family(c) for c in table.index])

    # per-seed values for CIs: the worst schedule is fixed per defender (the attacker commits
    # to one schedule), then its seed-to-seed spread is bootstrapped
    seeds = {}
    for cond, sched in worst.items():
        rows = onoff[(onoff.condition == cond) & (onoff.scenario == sched)]
        seeds[cond] = rows.set_index("seed").onoff_exposure_rate
    per_seed = pd.DataFrame(seeds)
    return table, per_seed


def pareto(table: pd.DataFrame, cols_max: list[str], cols_min: list[str]) -> np.ndarray:
    x = np.column_stack([-table[c].to_numpy() for c in cols_max]
                        + [table[c].to_numpy() for c in cols_min])
    dominated = np.array([np.any(np.all(x <= row, axis=1) & np.any(x < row, axis=1))
                          for row in x])
    return ~dominated


HIGHLIGHT = {"add_greedy": ("v1 additive rule, greedy (penalty P)", "penalty", "P"),
             "beta_ts": ("Beta + Thompson sampling (forgetting λ)", "forgetting", "λ")}


RIBBON = "#e3e2dc"                  # the Pareto front, drawn as a soft band under the marks
EXPOSURE_TICKS = [0.2, 0.5, 1, 2, 5, 10]


def plot(table: pd.DataFrame, out: Path) -> None:
    """frontier.svg: on-off exposure vs recovery share; exposure_delay.svg: exposure vs delay."""
    from figstyle import CONTEXT, INK_2, SERIES
    from svgplot import Figure

    t = table.assign(x=100 * table.recovery_share, y=100 * table.onoff_worst)
    ylim = (0.15, 15.0)
    fig = Figure(420, 330)
    ax = fig.axes((58, 14, 340, 210), (0, 100), ylim, yscale="log",
                  xlabel="recovery share: tasks to recovered providers (%)",
                  ylabel="on-off exposure, worst schedule (%)", yticks=EXPOSURE_TICKS)
    front = t[t.pareto_2d].sort_values("x")
    ax.line(front.x, front.y, RIBBON, width=9)
    ax.legend_items.append(("band", RIBBON, "Pareto front"))
    ax.points(t.x, t.y, CONTEXT, r=3.5, label="all 75 defenders",
              titles=[f"{c}: {x:.1f}%, {y:.2f}%" for c, x, y in zip(t.index, t.x, t.y)])
    for color, (fam, (label, param, symbol)) in zip(SERIES, HIGHLIGHT.items()):
        g = t[t.family == fam].copy()
        g["knob"] = g.index.str.extract(rf"{param}=([0-9.]+)", expand=False).astype(float)
        g = g.sort_values("knob")
        ax.line(g.x, g.y, color)
        ax.points(g.x, g.y, color, label=label,
                  titles=[f"{param}={k}: {x:.1f}%, {y:.2f}%" for k, x, y in zip(g.knob, g.x, g.y)])
        for _, row in g.iloc[[0, -1]].iterrows():               # label the ends of the knob
            ax.text(row.x, row.y, f"{symbol}={row.knob:g}", color=INK_2, size=9,
                    dx=7, dy=4)
    items = ax.legend_items
    fig.legend(items[:2], 60, 282)
    fig.legend(items[2:], 190, 282)
    fig.save(out / "frontier.svg")

    # exposure vs readmission delay, log-log; censored defenders (never readmitted) as rings
    censored = t.readmission >= 1699
    u = t[~censored]
    slope, icpt = np.polyfit(np.log(u.readmission), np.log(u.y), 1)
    r = np.corrcoef(np.log(u.readmission), np.log(u.y))[0, 1]
    fig = Figure(420, 322)
    ax = fig.axes((58, 14, 340, 210), (10, 2000), ylim, xscale="log", yscale="log",
                  xlabel="readmission delay of recovered providers (rounds)",
                  ylabel="on-off exposure, worst schedule (%)", yticks=EXPOSURE_TICKS,
                  xticks=[10, 30, 100, 300, 1000])
    ax.points(u.readmission, u.y, CONTEXT, r=3.5, label="defender",
              titles=[f"{c}: {d:.0f} rounds, {y:.2f}%" for c, d, y in
                      zip(u.index, u.readmission, u.y)])
    ax.points(np.full(censored.sum(), 1700.0), t.y[censored], CONTEXT, r=3.5, hollow=True,
              label="never readmitted (shown at the horizon)")
    xs = np.array([u.readmission.min(), u.readmission.max()])
    ax.line(xs, np.exp(icpt) * xs ** slope, INK_2, width=1.5,
            label=f"fit: exposure ∝ delay^{slope:.2f} (r = {r:.2f})")
    fig.legend(ax.legend_items, 60, 270)
    fig.save(out / "exposure_delay.svg")


def main() -> None:
    df = pd.read_csv(OUT / "per_seed.csv")
    table, onoff_seeds = per_defender(df)
    table["pareto_2d"] = pareto(table, ["recovery_share"], ["onoff_worst"])
    table["pareto_3d"] = pareto(table, ["recovery_share"], ["onoff_worst", "sleeper_exposure"])
    ci = {c: bootstrap_ci(onoff_seeds[c].to_numpy(), 2000) for c in table.index}
    table["onoff_ci_low"] = [ci[c][0] for c in table.index]
    table["onoff_ci_high"] = [ci[c][1] for c in table.index]
    table.to_csv(OUT / "frontier_table.csv", index_label="condition")

    pct = ["sleeper_exposure", "recovery_share", "onoff_worst", "stationary_regret"]
    show = table.copy()
    show[pct] = 100 * show[pct]
    cols = ["family", "recovery_share", "readmission", "onoff_worst", "worst_schedule",
            "sleeper_exposure", "stationary_regret", "pareto_3d"]
    with pd.option_context("display.width", 220, "display.max_rows", 200,
                           "display.float_format", "{:.2f}".format):
        print("=== 2-D front: recovery share (%) vs worst on-off exposure (%) ===")
        print(show[show.pareto_2d].sort_values("recovery_share")[cols].to_string())
        print("\n=== best point of each family on the 2-D trade-off (per family, sorted) ===")
        for fam, g in show.groupby("family", sort=False):
            front = g[pareto(g, ["recovery_share"], ["onoff_worst"])]
            pts = ", ".join(f"({r:.0f}%, {e:.2f}%)" for r, e in
                            front.sort_values("recovery_share")[["recovery_share",
                                                                  "onoff_worst"]].to_numpy())
            print(f"{fam:>16}: {pts}")
    plot(table, OUT)
    print(f"\n{len(table)} defenders, {table.pareto_3d.sum()} Pareto-optimal in 3-D; "
          f"wrote {OUT / 'frontier_table.csv'} and frontier.svg")


if __name__ == "__main__":
    main()
