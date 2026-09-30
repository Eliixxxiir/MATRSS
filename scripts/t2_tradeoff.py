"""T2 figure from the T1 sweep: cost in a benign world vs exposure to sleepers.

Every defender of configs/t1_frontier.yaml is one point: x = share of tasks sent to sleepers
after they turned bad (sleeper scenario), y = per-task regret when nobody changes (stationary
scenario). Thompson sampling must trade one for the other through its forgetting factor; greedy
rules with a short memory (the v1 additive rule, Beta with lambda <= 0.8) get both.

Usage: python scripts/t2_tradeoff.py [results/t1_frontier]   (run scripts/t1_frontier.py first)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from figstyle import CONTEXT, INK_2, SERIES
from svgplot import Figure

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/t1_frontier")
FAMILIES = {       # family -> (legend, knob, symbol, knob values to label, label y-offset)
    "add_greedy": ("v1 additive rule, greedy (P)", "penalty", "P", [0.2], -7),
    "beta_ts": ("Beta + Thompson sampling (λ)", "forgetting", "λ", [0.7, 1.0], 4),
    "beta_greedy": ("Beta, greedy (λ)", "forgetting", "λ", [0.7], 15),
}


def main() -> None:
    t = pd.read_csv(SRC / "frontier_table.csv", index_col="condition")
    t = t.assign(x=100 * t.sleeper_exposure, y=100 * t.stationary_regret)
    fig = Figure(420, 332)
    ax = fig.axes((58, 14, 340, 210), (0.1, 20), (0.5, 20), xscale="log", yscale="log",
                  xlabel="sleeper exposure: tasks sent to sleepers after they turned bad (%)",
                  ylabel="regret per task, stationary world (%)",
                  xticks=[0.1, 0.3, 1, 3, 10], yticks=[0.5, 1, 2, 5, 10, 20])
    ax.points(t.x, t.y, CONTEXT, r=3.5, label="all 75 defenders",
              titles=[f"{c}: {x:.2f}%, {y:.2f}%" for c, x, y in zip(t.index, t.x, t.y)])
    for color, (fam, (label, param, symbol, marked, dy)) in zip(SERIES, FAMILIES.items()):
        g = t[t.family == fam].copy()
        g["knob"] = g.index.str.extract(rf"{param}=([0-9.]+)", expand=False).astype(float)
        g = g.sort_values("knob")
        ax.line(g.x, g.y, color)
        ax.points(g.x, g.y, color, label=label,
                  titles=[f"{symbol}={k:g}: {x:.2f}%, {y:.2f}%" for k, x, y in
                          zip(g.knob, g.x, g.y)])
        for _, row in g[g.knob.isin(marked)].iterrows():
            ax.text(row.x, row.y, f"{symbol}={row.knob:g}", color=INK_2, size=9, dx=7, dy=dy)
    items = ax.legend_items
    fig.legend(items[:2], 60, 284)
    fig.legend(items[2:], 250, 284)
    fig.save(SRC.parent / "t2_mechanism" / "tradeoff_from_t1.svg")
    print(f"wrote {SRC.parent / 't2_mechanism' / 'tradeoff_from_t1.svg'}")


if __name__ == "__main__":
    main()
