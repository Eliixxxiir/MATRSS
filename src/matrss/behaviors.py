"""Provider behavior profiles: each maps a round index t to a success probability p(t).

Service agents have no agency here: they are Bernoulli processes whose parameter may drift.
Clients never see these objects; they only observe binary outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class Behavior(Protocol):
    @property
    def label(self) -> str: ...        # read-only, so frozen dataclasses satisfy it

    def p_success(self, t: int) -> float: ...


@dataclass(frozen=True)
class Stationary:
    """Constant success probability (honest, malicious, noisy)."""

    label: str
    p: float

    def p_success(self, t: int) -> float:
        return self.p


@dataclass(frozen=True)
class Drifting:
    """p(t) = p_end + (p_start - p_end) * exp(-rate * max(0, t - onset)).

    p_start > p_end gives a degrading ("sleeper") provider; p_start < p_end a recovering one.
    """

    label: str
    p_start: float
    p_end: float
    onset: int
    rate: float

    def p_success(self, t: int) -> float:
        if t < self.onset:
            return self.p_start
        return self.p_end + (self.p_start - self.p_end) * math.exp(-self.rate * (t - self.onset))


_DEFAULTS: dict[str, dict[str, float]] = {
    "honest": {"p": 0.95},
    "malicious": {"p": 0.05},
    "noisy": {"p": 0.50},
    "degrading": {"p_start": 0.95, "p_end": 0.05, "onset": 50, "rate": 0.1},
    "recovering": {"p_start": 0.05, "p_end": 0.95, "onset": 50, "rate": 0.1},
}


def build_behavior(kind: str, **params) -> Behavior:
    if kind not in _DEFAULTS:
        raise ValueError(f"unknown behavior '{kind}', expected one of {sorted(_DEFAULTS)}")
    cfg = {**_DEFAULTS[kind], **params}
    if "p" in cfg:
        return Stationary(label=kind, p=float(cfg["p"]))
    return Drifting(label=kind, p_start=float(cfg["p_start"]), p_end=float(cfg["p_end"]),
                    onset=int(cfg["onset"]), rate=float(cfg["rate"]))
