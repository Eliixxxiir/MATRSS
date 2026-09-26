"""Discrete-round simulation loop: select -> execute -> observe -> record.

Each round every client picks a provider from its own ledger, receives a binary outcome and
updates only its own ledger. All clients act simultaneously within a round, which is
vectorized: one NumPy step per round instead of one Python call per client.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import Environment
from .selection import SelectionStrategy
from .trust import TrustModel


@dataclass
class RunResult:
    condition: str
    seed: int
    labels: np.ndarray        # (P,) behavior label per provider
    p: np.ndarray             # (T, P) true success probabilities
    selections: np.ndarray    # (T, C) chosen provider
    successes: np.ndarray     # (T, C) bool outcome
    mean_trust: np.ndarray    # (T, P) trust score averaged over clients, after round t


def simulate(condition: str, seed: int, env: Environment, trust: TrustModel,
             policy: SelectionStrategy, rng: np.random.Generator) -> RunResult:
    if (trust.n_clients, trust.n_providers) != (env.n_clients, env.n_providers):
        raise ValueError("trust model shape does not match the environment")
    T, C, P = env.n_rounds, env.n_clients, env.n_providers
    selections = np.empty((T, C), dtype=np.int32)
    successes = np.empty((T, C), dtype=bool)
    mean_trust = np.empty((T, P))
    for t in range(T):
        chosen = policy.select(trust, rng)
        ok = env.outcomes(t, chosen)
        trust.update(chosen, ok)
        selections[t] = chosen
        successes[t] = ok
        mean_trust[t] = trust.scores().mean(axis=0)
    return RunResult(condition, seed, env.labels, env.p, selections, successes, mean_trust)
