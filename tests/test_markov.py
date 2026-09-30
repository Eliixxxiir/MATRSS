import numpy as np
import pytest

from matrss.behaviors import Stationary
from matrss.engine import simulate
from matrss.environment import Environment
from matrss.markov import AdditiveChain, wsls_shares
from matrss.rng import make_streams
from matrss.selection import Greedy
from matrss.trust import AsymmetricAdditive


def test_grid_is_derived_from_the_rule():
    chain = AdditiveChain.from_rule([0.9, 0.5], reward=0.1, penalty=0.2, init=0.5)
    assert (chain.up, chain.down, chain.top, chain.init) == (1, 2, 10, 5)
    assert chain.n_states == 11 ** 2


def test_transitions_conserve_probability():
    chain = AdditiveChain.from_rule([0.95, 0.8, 0.3])
    mu = chain.start()
    for _ in range(50):
        mu = chain.step(mu)
    assert mu.sum() == pytest.approx(1.0) and np.all(mu >= 0)


def test_identical_providers_share_evenly():
    shares = AdditiveChain.from_rule([0.9, 0.9]).long_run_shares()
    assert shares == pytest.approx([0.5, 0.5])


def test_exact_transient_matches_simulation():
    p = [0.95, 0.8, 0.5]
    exact = AdditiveChain.from_rule(p).transient(200)                       # (T, K)
    beh = [Stationary(f"p{x}", x) for x in p]
    env_rng, rng = make_streams(3)
    env = Environment(beh, 4000, 200, env_rng)
    res = simulate("x", 0, env, AsymmetricAdditive(4000, 3), Greedy(), rng)
    simulated = np.stack([(res.selections == j).mean(axis=1) for j in range(3)], axis=1)
    assert np.abs(simulated - exact).max() < 0.04          # 4000 clients: sampling noise only


def test_wsls_shares_follow_expected_run_length():
    assert wsls_shares([0.95, 0.8]) == pytest.approx([20 / 25, 5 / 25])
