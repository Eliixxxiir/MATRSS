"""Provider selection strategies (the exploration/exploitation policy).

Each strategy returns one provider per client, shape (n_clients,). Client c's choice depends
only on row c of the trust model, i.e. on its own private ledger.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .trust import BetaReputation, TrustModel

_TIE_TOL = 1e-9   # scores this close count as tied (absorbs float error, e.g. 0.5+0.1-0.1)


def argmax_random_tiebreak(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-wise argmax with uniform tie-breaking; index-order tie-breaking biases early picks."""
    best = x >= x.max(axis=1, keepdims=True) - _TIE_TOL
    keys = np.where(best, rng.random(x.shape), -1.0)
    return keys.argmax(axis=1)


class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, trust: TrustModel, rng: np.random.Generator) -> np.ndarray: ...


class Greedy(SelectionStrategy):
    def select(self, trust: TrustModel, rng: np.random.Generator) -> np.ndarray:
        return argmax_random_tiebreak(trust.scores(), rng)


class EpsilonGreedy(SelectionStrategy):
    def __init__(self, epsilon: float = 0.1) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = epsilon

    def select(self, trust: TrustModel, rng: np.random.Generator) -> np.ndarray:
        choice = argmax_random_tiebreak(trust.scores(), rng)
        explore = rng.random(trust.n_clients) < self.epsilon
        choice[explore] = rng.integers(trust.n_providers, size=int(explore.sum()))
        return choice


class ThompsonSampling(SelectionStrategy):
    """Sample theta_cj ~ Beta(alpha_cj, beta_cj) and pick argmax; needs a Beta trust model."""

    def select(self, trust: TrustModel, rng: np.random.Generator) -> np.ndarray:
        if not isinstance(trust, BetaReputation):
            raise TypeError("ThompsonSampling requires a BetaReputation trust model")
        return rng.beta(trust.alpha(), trust.beta()).argmax(axis=1)


def build_selection(type: str, **params) -> SelectionStrategy:
    strategies = {"greedy": Greedy, "epsilon_greedy": EpsilonGreedy, "thompson": ThompsonSampling}
    if type not in strategies:
        raise ValueError(f"unknown selection '{type}', expected one of {sorted(strategies)}")
    return strategies[type](**params)
