import numpy as np
import pytest

from matrss.trust import AsymmetricAdditive, BetaReputation, build_trust


def _observe(model, provider: int, success: bool, client: int = 0) -> None:
    """Single observation for one client; every other client repeats nothing new."""
    providers = np.zeros(model.n_clients, dtype=int)
    successes = np.zeros(model.n_clients, dtype=bool)
    providers[client], successes[client] = provider, success
    model.update(providers, successes)


def test_asymmetric_stays_in_unit_interval():
    rng = np.random.default_rng(0)
    m = AsymmetricAdditive(4, 3)
    for _ in range(10_000):
        m.update(rng.integers(3, size=4), rng.random(4) < 0.5)
        assert np.all((m.scores() >= 0) & (m.scores() <= 1))


def test_asymmetric_malicious_floor_in_three_failures():
    m = AsymmetricAdditive(1, 1)
    for _ in range(3):
        _observe(m, 0, False)
    assert m.scores()[0, 0] == 0.0


@pytest.mark.parametrize("p,direction", [(0.5, -1), (0.9, +1)])
def test_asymmetric_drift_sign_matches_break_even(p, direction):
    m = AsymmetricAdditive(1, 1)
    drift = p * m.reward - (1 - p) * m.penalty
    assert np.sign(drift) == direction
    assert m.break_even == pytest.approx(2 / 3)


def test_beta_is_posterior_mean_without_forgetting():
    m = BetaReputation(1, 1)
    for s in [1, 1, 0, 1]:
        _observe(m, 0, bool(s))
    assert m.scores()[0, 0] == pytest.approx((3 + 1) / (4 + 2))


def test_beta_forgetting_bounds_evidence():
    m = BetaReputation(1, 1, forgetting=0.9)
    for _ in range(1000):
        _observe(m, 0, True)
    assert m.r[0, 0] == pytest.approx(1 / (1 - 0.9), rel=1e-3)


@pytest.mark.parametrize("kind", ["asymmetric", "beta"])
def test_ledgers_are_private(kind):
    """Decentralization: an update touches only the observing client's own row and provider."""
    m = build_trust(3, 4, type=kind)
    before = m.scores().copy()
    m.update(np.array([2, 0, 1]), np.array([False, True, False]))
    changed = ~np.isclose(m.scores(), before)
    expected = np.zeros_like(changed)
    expected[[0, 1, 2], [2, 0, 1]] = True
    assert np.array_equal(changed, expected)


@pytest.mark.parametrize("kwargs", [{"reward": 0}, {"penalty": -1}, {"init": 1.5}])
def test_asymmetric_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        AsymmetricAdditive(1, 1, **kwargs)


def test_unknown_trust_model():
    with pytest.raises(ValueError, match="unknown trust model"):
        build_trust(1, 1, type="nope")
