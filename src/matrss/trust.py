"""Local trust models.

A model holds the trust ledgers of a whole client population as arrays of shape
(n_clients, n_providers): row c is client c's private ledger. Row c is written only from
client c's own observations and read only by client c's selection policy, so there is no
shared or global reputation. Keeping the rows in one array just lets NumPy update every
client in a single vectorized step instead of a Python loop over agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TrustModel(ABC):
    def __init__(self, n_clients: int, n_providers: int) -> None:
        self.n_clients = n_clients
        self.n_providers = n_providers
        self._rows = np.arange(n_clients)

    @abstractmethod
    def scores(self) -> np.ndarray:
        """Current point estimates, shape (n_clients, n_providers), in [0, 1]."""

    @abstractmethod
    def update(self, providers: np.ndarray, successes: np.ndarray) -> None:
        """Client c observed outcome successes[c] after delegating to providers[c]."""


class AsymmetricAdditive(TrustModel):
    """v1 MATRSS rule: T <- min(1, T + R) on success, max(0, T - P) on failure.

    Expected drift for a provider with success probability p is pR - (1 - p)P, so scores
    rise iff p > P / (R + P) (= 2/3 for R=0.1, P=0.2). The rule is therefore a threshold
    classifier, not a calibrated probability estimate.
    """

    def __init__(self, n_clients: int, n_providers: int, reward: float = 0.1,
                 penalty: float = 0.2, init: float = 0.5) -> None:
        super().__init__(n_clients, n_providers)
        if reward <= 0 or penalty <= 0:
            raise ValueError("reward and penalty must be positive")
        if not 0.0 <= init <= 1.0:
            raise ValueError("init must be in [0, 1]")
        self.reward, self.penalty = reward, penalty
        self._t = np.full((n_clients, n_providers), init, dtype=float)

    def scores(self) -> np.ndarray:
        return self._t

    def update(self, providers: np.ndarray, successes: np.ndarray) -> None:
        idx = (self._rows, providers)
        delta = np.where(successes, self.reward, -self.penalty)
        self._t[idx] = np.clip(self._t[idx] + delta, 0.0, 1.0)

    @property
    def break_even(self) -> float:
        return self.penalty / (self.reward + self.penalty)


class BetaReputation(TrustModel):
    """Beta reputation (Josang & Ismail, 2002) with exponential forgetting.

    Evidence r (successes) and s (failures) are discounted by lambda on each update of that
    provider; the score is the posterior mean (r + a0) / (r + s + a0 + b0). lambda = 1 is the
    stationary Bayesian estimate; lambda < 1 gives an effective memory of ~1 / (1 - lambda).
    """

    def __init__(self, n_clients: int, n_providers: int, prior_a: float = 1.0,
                 prior_b: float = 1.0, forgetting: float = 1.0) -> None:
        super().__init__(n_clients, n_providers)
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        if prior_a <= 0 or prior_b <= 0:
            raise ValueError("prior_a and prior_b must be positive")
        self.a0, self.b0, self.lam = prior_a, prior_b, forgetting
        self.r = np.zeros((n_clients, n_providers))
        self.s = np.zeros((n_clients, n_providers))

    def alpha(self) -> np.ndarray:
        return self.r + self.a0

    def beta(self) -> np.ndarray:
        return self.s + self.b0

    def scores(self) -> np.ndarray:
        a = self.alpha()
        return a / (a + self.beta())

    def update(self, providers: np.ndarray, successes: np.ndarray) -> None:
        idx = (self._rows, providers)
        self.r[idx] = self.lam * self.r[idx] + successes
        self.s[idx] = self.lam * self.s[idx] + ~successes


def build_trust(n_clients: int, n_providers: int, type: str, **params) -> TrustModel:
    models = {"asymmetric": AsymmetricAdditive, "beta": BetaReputation}
    if type not in models:
        raise ValueError(f"unknown trust model '{type}', expected one of {sorted(models)}")
    return models[type](n_clients, n_providers, **params)
