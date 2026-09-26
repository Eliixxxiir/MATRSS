import numpy as np
import pytest

from matrss.cli import build_providers, run_experiment
from matrss.config import Condition, ExperimentConfig
from matrss.environment import Environment
from matrss.rng import make_streams


def _cfg(seeds=(0,), **overrides):
    kw = {
        "name": "t", "n_rounds": 100, "n_clients": 3, "seeds": list(seeds),
        "providers": [{"type": "honest", "count": 2}, {"type": "malicious", "count": 2},
                      {"type": "degrading", "count": 1, "onset": 20}],
        "conditions": [Condition("g", {"type": "asymmetric"}, {"type": "greedy"}),
                       Condition("e", {"type": "asymmetric"}, {"type": "epsilon_greedy"})],
    }
    return ExperimentConfig(**{**kw, **overrides})


def test_environment_is_seed_determined():
    beh = build_providers([{"type": "honest", "count": 3}])
    a = Environment(beh, 4, 50, make_streams(7)[0])
    b = Environment(beh, 4, 50, make_streams(7)[0])
    assert np.array_equal(a._u, b._u)


def test_same_seed_same_results(tmp_path):
    d1 = run_experiment(_cfg([0, 1]), tmp_path / "a")
    d2 = run_experiment(_cfg([0, 1]), tmp_path / "b")
    assert d1.equals(d2)


def test_different_seeds_differ(tmp_path):
    d = run_experiment(_cfg([0, 1]), tmp_path)
    g = d[d.condition == "e"]
    assert g.success_rate.nunique() == 2


@pytest.mark.parametrize("overrides,match", [
    ({"reference": "missing"}, "reference"),
    ({"seeds": []}, "seed"),
    ({"conditions": [Condition("x", {"type": "asymmetric"}, {"type": "greedy"})] * 2}, "unique"),
])
def test_config_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        _cfg(**overrides)
