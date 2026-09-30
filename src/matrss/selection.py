"""Provider selection strategies (the exploration/exploitation policy).

Each strategy returns one provider per client, shape (n_clients,). Client c's choice depends
only on row c of the trust model, i.e. on its own private ledger.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .trust import BetaEvidence, TrustModel

_TIE_TOL = 1e-9   # scores this close count as tied (absorbs float error, e.g. 0.5+0.1-0.1)


def argmax_random_tiebreak(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-wise argmax with uniform tie-breaking; index-order tie-breaking biases early picks."""
    best = x >= x.max(axis=1, keepdims=True) - _TIE_TOL
    keys = np.where(best, rng.random(x.shape), -1.0)
    return keys.argmax(axis=1)


def _require_beta(trust: TrustModel, policy: str) -> BetaEvidence:
    if not isinstance(trust, BetaEvidence):
        raise TypeError(f"{policy} requires a Beta-evidence trust model (beta, window, cusum)")
    return trust


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
        beta = _require_beta(trust, "ThompsonSampling")
        return rng.beta(beta.alpha(), beta.beta()).argmax(axis=1)


class UCB(SelectionStrategy):
    """Pick argmax of mean + c * sqrt(log N / n): n is a ledger entry's evidence, N the client's
    total. On discounted or windowed evidence this is D-UCB / SW-UCB (Garivier & Moulines,
    2011); on undiscounted evidence, UCB1. Providers without evidence are tried first.
    """

    def __init__(self, c: float = 1.0) -> None:
        if c < 0:
            raise ValueError("c must be >= 0")
        self.c = c

    def select(self, trust: TrustModel, rng: np.random.Generator) -> np.ndarray:
        beta = _require_beta(trust, "UCB")
        n = beta.counts()
        total = np.maximum(n.sum(axis=1, keepdims=True), 1.0)
        seen = n > 1e-12
        safe_n = np.where(seen, n, 1.0)
        index = beta.r / safe_n + self.c * np.sqrt(np.log(total) / safe_n)
        return argmax_random_tiebreak(np.where(seen, index, np.inf), rng)


def build_selection(type: str, **params) -> SelectionStrategy:
    strategies = {"greedy": Greedy, "epsilon_greedy": EpsilonGreedy,
                  "thompson": ThompsonSampling, "ucb": UCB}
    if type not in strategies:
        raise ValueError(f"unknown selection '{type}', expected one of {sorted(strategies)}")
    return strategies[type](**params)
