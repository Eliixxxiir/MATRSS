"""Independent, reproducible RNG streams spawned from one master seed.

The environment stream and the clients' decision stream are separate children of a
SeedSequence, so strategies that consume different amounts of randomness cannot shift
the environment's draws (a flaw of sharing a single java.util.Random across everything).
"""

from __future__ import annotations

import numpy as np


def make_streams(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """Return (environment stream, decision stream) for a master seed."""
    env_ss, decision_ss = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(env_ss), np.random.default_rng(decision_ss)
