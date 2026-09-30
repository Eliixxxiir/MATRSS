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


class BetaEvidence(TrustModel):
    """Beta posterior Beta(r + a0, s + b0) per ledger entry, from success evidence r and
    failure evidence s. Subclasses decide how evidence is kept (discounted, windowed, reset).
    Thompson sampling and UCB need this interface."""

    def __init__(self, n_clients: int, n_providers: int, prior_a: float = 1.0,
                 prior_b: float = 1.0) -> None:
        super().__init__(n_clients, n_providers)
        if prior_a <= 0 or prior_b <= 0:
            raise ValueError("prior_a and prior_b must be positive")
        self.a0, self.b0 = prior_a, prior_b
        self.r = np.zeros((n_clients, n_providers))
        self.s = np.zeros((n_clients, n_providers))

    def alpha(self) -> np.ndarray:
        return self.r + self.a0

    def beta(self) -> np.ndarray:
        return self.s + self.b0

    def counts(self) -> np.ndarray:
        """Amount of (possibly discounted) evidence per ledger entry."""
        return self.r + self.s

    def scores(self) -> np.ndarray:
        a = self.alpha()
        return a / (a + self.beta())


class BetaReputation(BetaEvidence):
    """Beta reputation (Josang & Ismail, 2002) with exponential forgetting.

    Evidence r (successes) and s (failures) is discounted by lambda; the score is the posterior
    mean (r + a0) / (r + s + a0 + b0). lambda = 1 is the stationary Bayesian estimate; lambda < 1
    gives an effective memory of ~1 / (1 - lambda). `discount` sets what the memory counts:

    - "interaction": a ledger entry is discounted only when that provider is used again, as in
      reputation systems. A provider nobody uses keeps its (stale) evidence.
    - "time": every entry decays every round, as in discounted bandits (D-UCB, discounted
      Thompson sampling), so unused providers drift back to the prior and get re-explored.

    `forgetting_fail` discounts failures separately (asymmetric forgetting; defaults to
    `forgetting`), e.g. to remember failures longer than successes.
    """

    def __init__(self, n_clients: int, n_providers: int, prior_a: float = 1.0,
                 prior_b: float = 1.0, forgetting: float = 1.0,
                 forgetting_fail: float | None = None, discount: str = "interaction") -> None:
        super().__init__(n_clients, n_providers, prior_a, prior_b)
        lam_fail = forgetting if forgetting_fail is None else forgetting_fail
        if not (0.0 < forgetting <= 1.0 and 0.0 < lam_fail <= 1.0):
            raise ValueError("forgetting must be in (0, 1]")
        if discount not in ("interaction", "time"):
            raise ValueError("discount must be 'interaction' or 'time'")
        self.lam, self.lam_fail, self.discount = forgetting, lam_fail, discount

    def update(self, providers: np.ndarray, successes: np.ndarray) -> None:
        idx = (self._rows, providers)
        if self.discount == "time":
            self.r *= self.lam
            self.s *= self.lam_fail
            self.r[idx] += successes
            self.s[idx] += ~successes
        else:
            self.r[idx] = self.lam * self.r[idx] + successes
            self.s[idx] = self.lam_fail * self.s[idx] + ~successes


class SlidingWindowBeta(BetaEvidence):
    """Beta evidence from recent observations only.

    - mode="interaction": the last `window` outcomes of each (client, provider) entry.
    - mode="time": the outcomes of the client's last `window` rounds (SW-UCB / SW-TS style),
      so a provider unused for `window` rounds has no evidence left and is re-explored.
    """

    def __init__(self, n_clients: int, n_providers: int, window: int = 50,
                 mode: str = "interaction", prior_a: float = 1.0, prior_b: float = 1.0) -> None:
        super().__init__(n_clients, n_providers, prior_a, prior_b)
        if window < 1:
            raise ValueError("window must be >= 1")
        if mode not in ("interaction", "time"):
            raise ValueError("mode must be 'interaction' or 'time'")
        self.window, self.mode = int(window), mode
        if mode == "interaction":      # ring buffer per ledger entry; -1 = empty slot
            self._buf = np.full((n_clients, n_providers, self.window), -1, dtype=np.int8)
            self._n = np.zeros((n_clients, n_providers), dtype=np.int64)
        else:                          # ring buffer of (provider, outcome) per client
            self._prov = np.full((n_clients, self.window), -1, dtype=np.int64)
            self._out = np.zeros((n_clients, self.window), dtype=bool)
            self._t = 0

    def update(self, providers: np.ndarray, successes: np.ndarray) -> None:
        rows, x = self._rows, successes.astype(float)
        if self.mode == "interaction":
            pos = self._n[rows, providers] % self.window
            old = self._buf[rows, providers, pos]
            self.r[rows, providers] += x - (old == 1)
            self.s[rows, providers] += (1.0 - x) - (old == 0)
            self._buf[rows, providers, pos] = successes
            self._n[rows, providers] += 1
            return
        slot = self._t % self.window
        old_p, old_o = self._prov[:, slot], self._out[:, slot]
        full = old_p >= 0                              # the slot leaving the window is filled
        self.r[rows[full], old_p[full]] -= old_o[full]
        self.s[rows[full], old_p[full]] -= ~old_o[full]
        self.r[rows, providers] += x
        self.s[rows, providers] += 1.0 - x
        self._prov[:, slot], self._out[:, slot] = providers, successes
        self._t += 1


class CusumBeta(BetaEvidence):
    """Beta evidence with per-entry CUSUM change detection and local restarts, as in CUSUM-UCB
    (Liu, Lee & Shroff, 2018).

    The first `warmup` outcomes after a (re)start fix a reference mean u0. Afterwards
    g_down += u0 - x - drift and g_up += x - u0 - drift (both floored at 0) accumulate evidence
    of a drop or a rise; when one exceeds `threshold` the entry's evidence is discarded and
    learning restarts from the prior. With sides="down" only drops trigger a restart, so a
    provider cannot erase a bad record by starting to behave well.
    """

    def __init__(self, n_clients: int, n_providers: int, warmup: int = 20, drift: float = 0.15,
                 threshold: float = 4.0, sides: str = "both", forgetting: float = 1.0,
                 prior_a: float = 1.0, prior_b: float = 1.0) -> None:
        super().__init__(n_clients, n_providers, prior_a, prior_b)
        if warmup < 1 or drift < 0 or threshold <= 0:
            raise ValueError("need warmup >= 1, drift >= 0 and threshold > 0")
        if sides not in ("both", "down"):
            raise ValueError("sides must be 'both' or 'down'")
        if not 0.0 < forgetting <= 1.0:
            raise ValueError("forgetting must be in (0, 1]")
        self.warmup, self.drift, self.threshold = int(warmup), drift, threshold
        self.sides, self.lam = sides, forgetting
        shape = (n_clients, n_providers)
        self._m = np.zeros(shape, dtype=np.int64)       # outcomes since the last restart
        self._ref = np.zeros(shape)                      # sum of the warm-up outcomes
        self._g_up, self._g_down = np.zeros(shape), np.zeros(shape)
        self.restarts = np.zeros(shape, dtype=np.int64)

    def update(self, providers: np.ndarray, successes: np.ndarray) -> None:
        idx, x = (self._rows, providers), successes.astype(float)
        m = self._m[idx] + 1
        warm = m <= self.warmup
        ref = self._ref[idx] + np.where(warm, x, 0.0)
        u0 = ref / np.minimum(m, self.warmup)
        g_up = np.where(warm, 0.0, np.maximum(0.0, self._g_up[idx] + x - u0 - self.drift))
        g_down = np.where(warm, 0.0, np.maximum(0.0, self._g_down[idx] + u0 - x - self.drift))
        alarm = g_down > self.threshold
        if self.sides == "both":
            alarm |= g_up > self.threshold
        keep = ~alarm
        self.r[idx] = np.where(keep, self.lam * self.r[idx] + x, 0.0)
        self.s[idx] = np.where(keep, self.lam * self.s[idx] + 1.0 - x, 0.0)
        self._m[idx] = np.where(keep, m, 0)
        self._ref[idx] = np.where(keep, ref, 0.0)
        self._g_up[idx] = np.where(keep, g_up, 0.0)
        self._g_down[idx] = np.where(keep, g_down, 0.0)
        self.restarts[idx] += alarm


def build_trust(n_clients: int, n_providers: int, type: str, **params) -> TrustModel:
    models = {"asymmetric": AsymmetricAdditive, "beta": BetaReputation,
              "window": SlidingWindowBeta, "cusum": CusumBeta}
    if type not in models:
        raise ValueError(f"unknown trust model '{type}', expected one of {sorted(models)}")
    return models[type](n_clients, n_providers, **params)
