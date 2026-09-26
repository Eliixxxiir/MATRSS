import numpy as np
import pytest

from matrss.behaviors import build_behavior
from matrss.engine import simulate
from matrss.environment import Environment
from matrss.metrics import rounds_until_isolated
from matrss.rng import make_streams
from matrss.selection import (
    EpsilonGreedy,
    Greedy,
    ThompsonSampling,
    argmax_random_tiebreak,
    build_selection,
)
from matrss.trust import AsymmetricAdditive, BetaReputation


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


def test_thompson_requires_beta():
    with pytest.raises(TypeError):
        ThompsonSampling().select(AsymmetricAdditive(1, 2), np.random.default_rng(0))


def test_unknown_selection():
    with pytest.raises(ValueError, match="unknown selection"):
        build_selection(type="nope")


@pytest.mark.parametrize("trust_cls,policy", [
    (AsymmetricAdditive, Greedy()), (AsymmetricAdditive, EpsilonGreedy(0.1)),
    (BetaReputation, ThompsonSampling()),
])
def test_clients_learn_to_avoid_malicious(trust_cls, policy):
    beh = [build_behavior("honest")] + [build_behavior("malicious") for _ in range(4)]
    env_rng, rng = make_streams(0)
    env = Environment(beh, 20, 300, env_rng)
    res = simulate("x", 0, env, trust_cls(20, 5), policy, rng)
    late = res.successes[-100:].mean()
    assert late > 0.8                              # vs 0.23 for uniform random routing


def test_rounds_until_isolated():
    share = np.array([1.0] * 10 + [0.0] * 90)
    assert rounds_until_isolated(share, 0, 0.1, window=5) == 10
    assert rounds_until_isolated(share, 4, 0.1, window=5) == 6
    assert rounds_until_isolated(np.ones(50), 0, 0.1, window=5) == 50   # censored
