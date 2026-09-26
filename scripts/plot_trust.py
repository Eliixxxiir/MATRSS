"""Plot client-averaged trust trajectories by behavior label.

Usage: python scripts/plot_trust.py results/baseline/trust_trajectory_asym_eps0.1_seed0.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

path = Path(sys.argv[1])
df = pd.read_csv(path, index_col="round")
labels = df.columns.str.rsplit("_", n=1).str[0]
fig, ax = plt.subplots(figsize=(8, 4.5))
for label in labels.unique():
    ax.plot(df.index, df.loc[:, labels == label].mean(axis=1), label=label)
ax.set(xlabel="round", ylabel="mean trust (clients x providers)", ylim=(-0.02, 1.02),
       title=path.stem.replace("trust_trajectory_", ""))
ax.legend()
fig.tight_layout()
out = path.with_suffix(".png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
