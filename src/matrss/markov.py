"""Exact analysis of the additive trust rule with greedy selection in a stationary world.

Under AsymmetricAdditive + Greedy one client's ledger is a Markov chain: the state is the vector
of trust levels, which stay on the grid spanned by reward, penalty and the initial score; the
client delegates to a uniformly random argmax provider j, whose level then moves up by R with
probability p_j and down by P otherwise, clipped to [0, 1]. Clients never interact, so one
client's chain describes the whole population. With K providers and L + 1 grid levels the
chain has (L + 1)^K states, so the analysis is exact but only practical for small K.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass
class AdditiveChain:
    p: np.ndarray            # (K,) success probability of each provider
    up: int                  # reward, in grid steps
    down: int                # penalty, in grid steps
    top: int                 # score 1.0, in grid steps (levels are 0..top)
    init: int                # initial score, in grid steps

    @classmethod
    def from_rule(cls, p, reward: float = 0.1, penalty: float = 0.2,
                  init: float = 0.5) -> AdditiveChain:
        fracs = [Fraction(x).limit_denominator(1000) for x in (reward, penalty, init)]
        scale = math.lcm(*(f.denominator for f in fracs))
        up, down, start = (int(f * scale) for f in fracs)
        return cls(np.asarray(p, dtype=float), up, down, scale, start)

    def __post_init__(self) -> None:
        k, levels = self.p.size, self.top + 1
        self.states = np.indices((levels,) * k).reshape(k, -1).T           # (S, K) levels
        best = self.states == self.states.max(axis=1, keepdims=True)
        self.choice = best / best.sum(axis=1, keepdims=True)                # P(pick j | s)
        # sparse transitions: pick j (prob choice[s, j]), then succeed or fail
        src, dst, prob = [], [], []
        idx = np.arange(len(self.states))
        for j in range(k):
            for ok, pj in ((True, self.p[j]), (False, 1.0 - self.p[j])):
                nxt = self.states.copy()
                nxt[:, j] = (np.minimum(self.top, nxt[:, j] + self.up) if ok
                             else np.maximum(0, nxt[:, j] - self.down))
                keep = self.choice[:, j] > 0
                src.append(idx[keep])
                dst.append(np.ravel_multi_index(nxt[keep].T, (levels,) * k))
                prob.append(self.choice[keep, j] * pj)
        self._src, self._dst = np.concatenate(src), np.concatenate(dst)
        self._prob = np.concatenate(prob)

    @property
    def n_states(self) -> int:
        return len(self.states)

    def start(self) -> np.ndarray:
        mu = np.zeros(self.n_states)
        mu[np.ravel_multi_index((self.init,) * self.p.size, (self.top + 1,) * self.p.size)] = 1.0
        return mu

    def step(self, mu: np.ndarray) -> np.ndarray:
        return np.bincount(self._dst, weights=mu[self._src] * self._prob,
                           minlength=self.n_states)

    def selection_shares(self, mu: np.ndarray) -> np.ndarray:
        """P(delegate to provider j) under the state distribution mu, shape (K,)."""
        return mu @ self.choice

    def transient(self, n_rounds: int) -> np.ndarray:
        """Exact P(delegate to j) in each of the first n_rounds rounds, shape (T, K)."""
        mu, out = self.start(), np.empty((n_rounds, self.p.size))
        for t in range(n_rounds):
            out[t] = self.selection_shares(mu)
            mu = self.step(mu)
        return out

    def stationary(self, max_states: int = 6000) -> np.ndarray:
        """Stationary distribution by an exact linear solve of pi P = pi, sum(pi) = 1.

        Power iteration is unreliable here: a provider nobody picks keeps its level, so mass in
        states with one provider far below the rest decays over millions of rounds.
        """
        s = self.n_states
        if s > max_states:
            raise ValueError(f"{s} states: exact solve is limited to {max_states}; "
                             "use transient() for finite horizons")
        a = np.zeros((s, s))
        np.add.at(a, (self._dst, self._src), self._prob)        # a = P^T
        a -= np.eye(s)
        a[-1] = 1.0                                             # normalisation replaces a row
        b = np.zeros(s)
        b[-1] = 1.0
        return np.linalg.solve(a, b)

    def long_run_shares(self) -> np.ndarray:
        return self.selection_shares(self.stationary())


def wsls_shares(p) -> np.ndarray:
    """Long-run shares of plain win-stay/lose-shift with a uniform switch: each visit to j lasts
    a geometric 1 / (1 - p_j) rounds and visits are uniform, so share_j is proportional to it."""
    stay = 1.0 / (1.0 - np.asarray(p, dtype=float))
    return stay / stay.sum()
