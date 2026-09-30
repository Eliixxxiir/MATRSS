"""YAML experiment configuration.

An experiment crosses scenarios (provider populations) with conditions (trust model + selection
policy) over seeds. Any scenario or condition may carry a `grid` of dotted parameter paths,
e.g. `{trust.forgetting: [0.9, 0.99]}` or `{providers.1.good: [25, 50]}`, which expands it into
one entry per combination, named `name[forgetting=0.9]` and so on.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Condition:
    name: str
    trust: dict
    selection: dict


@dataclass
class Scenario:
    name: str
    providers: list[dict]


@dataclass
class ExperimentConfig:
    name: str
    n_rounds: int
    n_clients: int
    seeds: list[int]
    conditions: list[Condition]
    providers: list[dict] = field(default_factory=list)      # shorthand for a single scenario
    scenarios: list[Scenario] = field(default_factory=list)
    reference: str | None = None
    isolation_threshold: float = 0.3      # isolated = routed to at <= this x random-routing rate
    isolation_window: int = 5             # ... averaged over this many upcoming rounds
    readmission_level: float = 0.5       # recovered providers readmitted at this share of tasks
    trajectories: bool = True             # write first-seed trust trajectories
    n_boot: int = 10_000                  # bootstrap resamples per summary CI
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_rounds < 1 or self.n_clients < 1:
            raise ValueError("n_rounds and n_clients must be >= 1")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not self.conditions:
            raise ValueError("at least one condition is required")
        if self.providers and self.scenarios:
            raise ValueError("give either providers or scenarios, not both")
        if not self.scenarios:
            if not self.providers:
                raise ValueError("providers or scenarios are required")
            self.scenarios = [Scenario(self.name, self.providers)]
        for kind, names in (("condition", [c.name for c in self.conditions]),
                            ("scenario", [s.name for s in self.scenarios])):
            if len(set(names)) != len(names):
                raise ValueError(f"{kind} names must be unique, got {names}")
        names = [c.name for c in self.conditions]
        if self.reference is not None and self.reference not in names:
            raise ValueError(f"reference '{self.reference}' is not a condition name {names}")
        if not 0.0 < self.isolation_threshold < 1.0:
            raise ValueError("isolation_threshold must be in (0, 1)")
        if self.isolation_window < 1:
            raise ValueError("isolation_window must be >= 1")
        if not 0.0 < self.readmission_level <= 1.0:
            raise ValueError("readmission_level must be in (0, 1]")

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    @classmethod
    def from_dict(cls, raw: dict) -> ExperimentConfig:
        s = raw["seeds"]
        seeds = list(range(s["start"], s["stop"])) if isinstance(s, dict) else list(s)
        optional = ("reference", "isolation_threshold", "isolation_window", "readmission_level",
                    "trajectories", "n_boot")
        return cls(
            name=raw["name"], n_rounds=raw["n_rounds"], n_clients=raw["n_clients"], seeds=seeds,
            providers=raw.get("providers", []),
            scenarios=[Scenario(**x) for sc in raw.get("scenarios", []) for x in expand_grid(sc)],
            conditions=[Condition(**x) for c in raw["conditions"] for x in expand_grid(c)],
            **{k: raw[k] for k in optional if k in raw},
        )


def expand_grid(item: dict) -> list[dict]:
    """Expand `item["grid"]` ({dotted.path: [values]}) into one item per combination."""
    grid = item.get("grid")
    base = {k: v for k, v in item.items() if k != "grid"}
    if not grid:
        return [base]
    keys = list(grid)
    out = []
    for values in itertools.product(*(grid[k] for k in keys)):
        new = copy.deepcopy(base)
        for key, value in zip(keys, values):
            _set_path(new, key, value)
        tag = ",".join(f"{k.rsplit('.', 1)[-1]}={v}" for k, v in zip(keys, values))
        new["name"] = f"{base['name']}[{tag}]"
        out.append(new)
    return out


def _set_path(obj: Any, path: str, value: Any) -> None:
    *parents, last = path.split(".")
    for part in parents:
        obj = obj[int(part)] if isinstance(obj, list) else obj.setdefault(part, {})
    if isinstance(obj, list):
        obj[int(last)] = value
    else:
        obj[last] = value
