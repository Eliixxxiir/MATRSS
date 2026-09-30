"""Run-level metrics. All are computed from recorded data, never hard-coded."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .behaviors import Behavior, Drifting, OnOff
from .engine import RunResult


def rounds_until_level(share: np.ndarray, start: int, level: float, window: int,
                       above: bool = False) -> int:
    """Rounds after `start` until the forward `window`-round mean of `share` is <= `level`
    (>= `level` if `above`).

    A forward window (rounds t .. t+window-1) avoids the lag a trailing average would add.
    Censored: returns the number of remaining rounds if the level is never reached.
    """
    s = share[start:]
    n = s.size
    csum = np.concatenate(([0.0], np.cumsum(s)))
    begin = np.arange(n)
    end = np.minimum(begin + window, n)
    fwd = (csum[end] - csum[begin]) / (end - begin)
    hit = np.flatnonzero(fwd >= level if above else fwd <= level)
    return int(hit[0]) if hit.size else n


def rounds_until_isolated(share: np.ndarray, start: int, limit: float, window: int) -> int:
    """Rounds after `start` until the forward `window`-round mean of `share` is <= `limit`."""
    return rounds_until_level(share, start, limit, window)


def summarize(res: RunResult, behaviors: Sequence[Behavior], isolation_threshold: float = 0.3,
              window: int = 5, readmission_level: float = 0.5) -> dict[str, float]:
    n_providers = res.p.shape[1]
    sel_labels = res.labels[res.selections]
    p_chosen = np.take_along_axis(res.p, res.selections, axis=1)       # (T, C)
    p_best = res.p.max(axis=1, keepdims=True)                           # oracle per round
    out: dict[str, float] = {
        "success_rate": float(res.successes.mean()),
        "expected_success": float(p_chosen.mean()),                     # removes outcome noise
        "mean_regret_per_task": float((p_best - p_chosen).mean()),
        "malicious_selection_rate": float((sel_labels == "malicious").mean()),
    }

    # Isolation is behavioural, not trust-based: a provider that stops being selected keeps a
    # stale score (absence of evidence), so we measure when clients stop *routing* to it.
    # "Isolated" = routed to at <= isolation_threshold x the rate of uniform random routing.
    n_mal = int((res.labels == "malicious").sum())
    if n_mal:
        share = (sel_labels == "malicious").mean(axis=1)                # (T,) per-round share
        limit = isolation_threshold * n_mal / n_providers
        out["malicious_isolation_rounds"] = float(
            rounds_until_isolated(share, 0, limit, window))

    # Degrading providers: exposure = share of all tasks delegated to one while its current
    # p(t) < 0.5; adaptation = rounds from that drop until clients have isolated it.
    deg = [j for j, b in enumerate(behaviors) if isinstance(b, Drifting) and b.p_end < b.p_start]
    if deg:
        out["degraded_exposure_rate"] = _exposure(res, deg)
        adapt = []
        for j in deg:
            below = np.flatnonzero(res.p[:, j] < 0.5)
            if below.size:
                share_j = (res.selections == j).mean(axis=1)
                adapt.append(rounds_until_isolated(share_j, int(below[0]),
                                                   isolation_threshold / n_providers, window))
        if adapt:
            out["degraded_adaptation_rounds"] = float(np.mean(adapt))

    # On-off attackers: exposure = share of all tasks delegated to one during a bad phase.
    onoff = [j for j, b in enumerate(behaviors) if isinstance(b, OnOff)]
    if onoff:
        out["onoff_selection_rate"] = float(np.isin(res.selections, onoff).mean())
        out["onoff_exposure_rate"] = _exposure(res, onoff)

    # Recovering providers (assumed to recover together): from the first round one of them is
    # good again (p >= 0.5), their share of all tasks, and the rounds until that share first
    # reaches `readmission_level` (forward-window mean, censored like isolation).
    rec = [j for j, b in enumerate(behaviors) if isinstance(b, Drifting) and b.p_end > b.p_start]
    if rec:
        good = np.flatnonzero((res.p[:, rec] >= 0.5).any(axis=1))
        if good.size:
            start = int(good[0])
            share = np.isin(res.selections, rec).mean(axis=1)
            out["recovered_share"] = float(share[start:].mean())
            out["readmission_rounds"] = float(
                rounds_until_level(share, start, readmission_level, window, above=True))
    return out


def _exposure(res: RunResult, providers: list[int]) -> float:
    """Share of all tasks delegated to one of `providers` while its current p(t) < 0.5."""
    bad_now = np.zeros_like(res.p, dtype=bool)
    bad_now[:, providers] = res.p[:, providers] < 0.5
    return float(np.take_along_axis(bad_now, res.selections, axis=1).mean())


def bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
