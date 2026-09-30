import numpy as np
import pytest

from matrss.trust import (
    AsymmetricAdditive,
    BetaReputation,
    CusumBeta,
    SlidingWindowBeta,
    build_trust,
)


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


def _feed(model, outcomes, provider: int = 0) -> None:
    for x in outcomes:
        _observe(model, provider, bool(x))


def test_time_discount_decays_unused_entries():
    m = BetaReputation(1, 2, forgetting=0.5, discount="time")
    _feed(m, [1], provider=0)
    _feed(m, [1], provider=1)                      # provider 0's evidence decays meanwhile
    assert (m.r[0, 0], m.r[0, 1]) == (pytest.approx(0.5), pytest.approx(1.0))


def test_interaction_discount_keeps_unused_entries():
    m = BetaReputation(1, 2, forgetting=0.5)
    _feed(m, [1], provider=0)
    _feed(m, [1], provider=1)
    assert m.r[0, 0] == pytest.approx(1.0)


def test_asymmetric_forgetting_remembers_failures_longer():
    m = BetaReputation(1, 1, forgetting=0.5, forgetting_fail=1.0)
    _feed(m, [0] + [1] * 10)
    assert m.s[0, 0] == 1.0
    assert m.r[0, 0] == pytest.approx(2.0, abs=0.01)     # geometric memory 1 / (1 - 0.5)


@pytest.mark.parametrize("mode", ["interaction", "time"])
def test_window_keeps_only_recent_evidence(mode):
    m = SlidingWindowBeta(1, 1, window=4, mode=mode)
    _feed(m, [0, 0, 0, 1, 1, 1, 1])
    assert (m.r[0, 0], m.s[0, 0]) == (4, 0)


@pytest.mark.parametrize("mode,left", [("interaction", 1), ("time", 0)])
def test_window_mode_decides_whether_unused_evidence_expires(mode, left):
    m = SlidingWindowBeta(1, 2, window=3, mode=mode)
    _feed(m, [1], provider=0)
    _feed(m, [0, 0, 0], provider=1)
    assert m.counts()[0, 0] == left and m.counts()[0, 1] == 3


def test_cusum_restarts_after_a_drop():
    m = CusumBeta(1, 1, warmup=10, drift=0.1, threshold=3.0)
    _feed(m, [1] * 30 + [0] * 3)                   # g_down = 3 x 0.9 = 2.7: no alarm yet
    assert m.restarts[0, 0] == 0
    _feed(m, [0, 0])                               # 4th failure alarms, 5th starts afresh
    assert m.restarts[0, 0] == 1
    assert (m.r[0, 0], m.s[0, 0]) == (0, 1)


@pytest.mark.parametrize("sides,expected", [("both", 1), ("down", 0)])
def test_cusum_rise_restarts_only_when_two_sided(sides, expected):
    m = CusumBeta(1, 1, warmup=10, drift=0.1, threshold=3.0, sides=sides)
    _feed(m, [0] * 10 + [1] * 5)
    assert m.restarts[0, 0] == expected


ALL_MODELS = [
    ("asymmetric", {}), ("beta", {}), ("beta", {"forgetting": 0.9, "discount": "time"}),
    ("beta", {"forgetting": 0.9, "forgetting_fail": 0.99}), ("window", {"window": 3}),
    ("window", {"window": 3, "mode": "time"}), ("cusum", {"warmup": 2, "threshold": 1.0}),
]


@pytest.mark.parametrize("kind,params", ALL_MODELS)
def test_rows_depend_only_on_own_observations(kind, params):
    """Decentralization: client 0's ledger is the same whatever the other clients observe."""
    rng = np.random.default_rng(0)
    a, b = build_trust(3, 4, type=kind, **params), build_trust(3, 4, type=kind, **params)
    for _ in range(50):
        prov, ok = rng.integers(4, size=3), rng.random(3) < 0.5
        a.update(prov, ok)
        prov[1:], ok[1:] = rng.integers(4, size=2), rng.random(2) < 0.5
        b.update(prov, ok)
    assert np.array_equal(a.scores()[0], b.scores()[0])


@pytest.mark.parametrize("kind,params", [
    ("beta", {"forgetting_fail": 0}), ("beta", {"discount": "weekly"}),
    ("window", {"window": 0}), ("window", {"mode": "x"}),
    ("cusum", {"sides": "up"}), ("cusum", {"threshold": 0}),
])
def test_new_models_reject_bad_params(kind, params):
    with pytest.raises(ValueError):
        build_trust(1, 1, type=kind, **params)
