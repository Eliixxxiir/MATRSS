"""Client agents as views onto the population trust model.

A client's belief is its row of the trust model; its intention is the selection policy
applied to that row (see selection.py). The engine updates all clients in one vectorized
step, so this view exists for inspection, e.g. printing what client 3 believes after a run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trust import TrustModel


@dataclass(frozen=True)
class ClientAgent:
    cid: int
    trust: TrustModel

    def ledger(self) -> np.ndarray:
        """Copy of this client's private trust scores, shape (n_providers,)."""
        return self.trust.scores()[self.cid].copy()
