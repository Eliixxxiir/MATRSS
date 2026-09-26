"""Environment with common random numbers (CRN).

All environment randomness is pre-drawn as U[t, c, j] ~ Uniform(0, 1). The outcome of client c
delegating to provider j in round t is U[t, c, j] < p_j(t). This makes the world's randomness
independent of which strategy is being evaluated, so every condition run with the same seed
faces an identical realisation, and between-condition differences can be analysed as paired.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .behaviors import Behavior


class Environment:
    def __init__(self, behaviors: Sequence[Behavior], n_clients: int, n_rounds: int,
                 rng: np.random.Generator) -> None:
        self.behaviors = list(behaviors)
        self.labels = np.array([b.label for b in self.behaviors])
        self.n_providers = len(self.behaviors)
        self.n_clients = n_clients
        self.n_rounds = n_rounds
        self.p = np.array([[b.p_success(t) for b in self.behaviors] for t in range(n_rounds)])
        self._u = rng.random((n_rounds, n_clients, self.n_providers))
        self._clients = np.arange(n_clients)

    def outcomes(self, t: int, providers: np.ndarray) -> np.ndarray:
        """Bool outcome for every client c delegating round t's task to providers[c]."""
        return self._u[t, self._clients, providers] < self.p[t, providers]
