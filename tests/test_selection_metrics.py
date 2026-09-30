import math

import numpy as np
import pytest

from matrss.behaviors import Drifting, OnOff, Stationary, build_behavior
from matrss.engine import RunResult, simulate
from matrss.environment import Environment
from matrss.metrics import rounds_until_isolated, rounds_until_level, summarize
from matrss.rng import make_streams
from matrss.selection import (
    UCB,
    EpsilonGreedy,
    Greedy,
    ThompsonSampling,
    argmax_random_tiebreak,
    build_selection,
)
from matrss.trust import AsymmetricAdditive, BetaReputation, CusumBeta, SlidingWindowBeta


def test_tiebreak_is_uniform_over_ties():
    rng = np.random.default_rng(0)
    x = np.tile([0.2, 0.9, 0.9, 0.9], (30_000, 1))
    counts = np.bincount(argmax_random_tiebreak(x, rng), minlength=4)
    assert counts[0] == 0
    assert np.allclose(counts[1:] / counts[1:].sum(), 1 / 3, atol=0.01)


def test_epsilon_zero_equals_greedy():
    trust = AsymmetricAdditive(50, 5)
    trust._t[:] = np.random.default_rng(1).random((50, 5))
    a = Greedy().select(trust, np.random.default_rng(2))
    b = EpsilonGreedy(0.0).select(trust, np.random.default_rng(3))
    assert np.array_equal(a, b)                    # no ties, so choice is deterministic


def test_epsilon_explores_at_its_rate():
    trust = AsymmetricAdditive(20_000, 10)
    trust._t[:, 0] = 1.0                           # provider 0 is everyone's clear favourite
    picks = EpsilonGreedy(0.2).select(trust, np.random.default_rng(0))
    # non-favourite picks happen with probability epsilon * (P - 1) / P = 0.18
    assert np.mean(picks != 0) == pytest.approx(0.18, abs=0.01)


@pytest.mark.parametrize("policy", [ThompsonSampling(), UCB()])
def test_beta_policies_require_beta_evidence(policy):
    with pytest.raises(TypeError):
        policy.select(AsymmetricAdditive(1, 2), np.random.default_rng(0))


def test_unknown_selection():
    with pytest.raises(ValueError, match="unknown selection"):
        build_selection(type="nope")


def test_ucb_tries_unobserved_providers_first():
    trust = BetaReputation(5, 3)
    trust.r[:, 0] = 100.0                          # provider 0 looks perfect, 1 and 2 untried
    assert np.all(UCB().select(trust, np.random.default_rng(0)) != 0)


def test_ucb_bonus_favours_the_less_observed_provider():
    trust = BetaReputation(1, 2)
    trust.r[0], trust.s[0] = [80.0, 8.0], [20.0, 2.0]          # both 0.8, 100 vs 10 outcomes
    assert UCB(c=1.0).select(trust, np.random.default_rng(0))[0] == 1


@pytest.mark.parametrize("make_trust,policy", [
    (lambda c, p: AsymmetricAdditive(c, p), Greedy()),
    (lambda c, p: AsymmetricAdditive(c, p), EpsilonGreedy(0.1)),
    (lambda c, p: BetaReputation(c, p), ThompsonSampling()),
    (lambda c, p: BetaReputation(c, p), UCB()),
    (lambda c, p: BetaReputation(c, p, forgetting=0.995, discount="time"), UCB()),
    (lambda c, p: SlidingWindowBeta(c, p, window=50, mode="time"), ThompsonSampling()),
    (lambda c, p: CusumBeta(c, p), UCB()),
])
def test_clients_learn_to_avoid_malicious(make_trust, policy):
    beh = [build_behavior("honest")] + [build_behavior("malicious") for _ in range(4)]
    env_rng, rng = make_streams(0)
    env = Environment(beh, 20, 300, env_rng)
    res = simulate("x", 0, env, make_trust(20, 5), policy, rng)
    late = res.successes[-100:].mean()
    assert late > 0.8                              # vs 0.23 for uniform random routing


def test_rounds_until_isolated():
    share = np.array([1.0] * 10 + [0.0] * 90)
    assert rounds_until_isolated(share, 0, 0.1, window=5) == 10
    assert rounds_until_isolated(share, 4, 0.1, window=5) == 6
    assert rounds_until_isolated(np.ones(50), 0, 0.1, window=5) == 50   # censored


def test_rounds_until_level_above():
    share = np.array([0.0] * 10 + [1.0] * 10)
    assert rounds_until_level(share, 2, 0.5, window=1, above=True) == 8
    assert rounds_until_level(np.zeros(20), 2, 0.5, window=1, above=True) == 18   # censored


def _run_result(behaviors, selections):
    """RunResult for hand-written per-round selections (successes are irrelevant here)."""
    sel = np.asarray(selections)
    p = np.array([[b.p_success(t) for b in behaviors] for t in range(sel.shape[0])])
    labels = np.array([b.label for b in behaviors])
    return RunResult("x", 0, labels, p, sel, np.ones(sel.shape, dtype=bool), np.zeros_like(p))


def test_onoff_exposure_counts_bad_phase_only():
    beh = [Stationary("honest", 0.9), OnOff("onoff", 0.9, 0.05, good=2, bad=2)]
    res = _run_result(beh, [[1, 0], [0, 0], [1, 0], [0, 0]])   # attacker is bad at t = 2, 3
    m = summarize(res, beh)
    assert m["onoff_selection_rate"] == pytest.approx(2 / 8)
    assert m["onoff_exposure_rate"] == pytest.approx(1 / 8)


def test_recovery_share_and_readmission():
    beh = [Stationary("honest", 0.9), Drifting("recovering", 0.05, 0.95, onset=2, rate=math.inf)]
    res = _run_result(beh, [[0], [0], [0], [0], [1], [1]])     # readmitted two rounds late
    m = summarize(res, beh, window=1, readmission_level=0.5)
    assert m["recovered_share"] == pytest.approx(2 / 4)
    assert m["readmission_rounds"] == 2
