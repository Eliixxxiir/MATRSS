"""YAML experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Condition:
    name: str
    trust: dict
    selection: dict


@dataclass
class ExperimentConfig:
    name: str
    n_rounds: int
    n_clients: int
    seeds: list[int]
    providers: list[dict]
    conditions: list[Condition]
    reference: str | None = None
    isolation_threshold: float = 0.3      # isolated = routed to at <= this x random-routing rate
    isolation_window: int = 5             # ... averaged over this many upcoming rounds
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_rounds < 1 or self.n_clients < 1:
            raise ValueError("n_rounds and n_clients must be >= 1")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not self.conditions:
            raise ValueError("at least one condition is required")
        names = [c.name for c in self.conditions]
        if len(set(names)) != len(names):
            raise ValueError(f"condition names must be unique, got {names}")
        if self.reference is not None and self.reference not in names:
            raise ValueError(f"reference '{self.reference}' is not a condition name {names}")
        if not 0.0 < self.isolation_threshold < 1.0:
            raise ValueError("isolation_threshold must be in (0, 1)")
        if self.isolation_window < 1:
            raise ValueError("isolation_window must be >= 1")

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text())
        s = raw["seeds"]
        seeds = list(range(s["start"], s["stop"])) if isinstance(s, dict) else list(s)
        return cls(
            name=raw["name"], n_rounds=raw["n_rounds"], n_clients=raw["n_clients"], seeds=seeds,
            providers=raw["providers"], conditions=[Condition(**c) for c in raw["conditions"]],
            reference=raw.get("reference"), isolation_threshold=raw.get("isolation_threshold", 0.3),
            isolation_window=raw.get("isolation_window", 5),
        )
