import numpy as np
import pandas as pd
import pytest
import yaml

from matrss.cli import build_providers, main, run_experiment
from matrss.config import Condition, ExperimentConfig, Scenario
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


def test_parallel_run_matches_serial(tmp_path):
    cfg = _cfg([0, 1, 2], trajectories=False)
    serial = run_experiment(cfg, tmp_path / "s", jobs=1)
    parallel = run_experiment(cfg, tmp_path / "p", jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)


@pytest.mark.parametrize("overrides,match", [
    ({"reference": "missing"}, "reference"),
    ({"seeds": []}, "seed"),
    ({"conditions": [Condition("x", {"type": "asymmetric"}, {"type": "greedy"})] * 2}, "unique"),
    ({"scenarios": [Scenario("s", [{"type": "honest"}])]}, "not both"),
])
def test_config_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        _cfg(**overrides)


def test_grid_expands_scenarios_and_conditions():
    cfg = ExperimentConfig.from_dict({
        "name": "g", "n_rounds": 10, "n_clients": 2, "seeds": [0],
        "scenarios": [{"name": "oo", "providers": [{"type": "honest"}, {"type": "onoff"}],
                       "grid": {"providers.1.good": [5, 10]}}],
        "conditions": [{"name": "b", "trust": {"type": "beta"}, "selection": {"type": "thompson"},
                        "grid": {"trust.forgetting": [0.9, 1.0],
                                 "selection.type": ["thompson", "ucb"]}}],
    })
    assert [s.name for s in cfg.scenarios] == ["oo[good=5]", "oo[good=10]"]
    assert [s.providers[1]["good"] for s in cfg.scenarios] == [5, 10]    # copies, not shared
    assert len(cfg.conditions) == 4
    assert cfg.conditions[1].name == "b[forgetting=0.9,type=ucb]"
    assert cfg.conditions[1].trust == {"type": "beta", "forgetting": 0.9}
    assert cfg.conditions[1].selection == {"type": "ucb"}


def test_scenarios_get_their_own_rows(tmp_path):
    cfg = ExperimentConfig(
        name="m", n_rounds=50, n_clients=2, seeds=[0], trajectories=False,
        scenarios=[Scenario("a", [{"type": "honest"}, {"type": "malicious"}]),
                   Scenario("b", [{"type": "honest"}, {"type": "onoff", "good": 5, "bad": 5}])],
        conditions=[Condition("ts", {"type": "beta"}, {"type": "thompson"})])
    df = run_experiment(cfg, tmp_path)
    assert list(df.scenario) == ["a", "b"]
    assert np.isnan(df.onoff_exposure_rate[0]) and df.onoff_exposure_rate[1] >= 0
    assert not list(tmp_path.glob("trust_trajectory_*"))


def test_cli_writes_summary_with_paired_differences(tmp_path):
    raw = {"name": "c", "n_rounds": 60, "n_clients": 2, "seeds": [0, 1], "n_boot": 200,
           "reference": "g", "providers": [{"type": "honest"}, {"type": "malicious", "count": 2}],
           "conditions": [{"name": "g", "trust": {"type": "asymmetric"},
                           "selection": {"type": "greedy"}},
                          {"name": "u", "trust": {"type": "beta"}, "selection": {"type": "ucb"}}]}
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(raw))
    main(["run", str(tmp_path / "c.yaml"), "--out", str(tmp_path / "out"), "--quiet"])
    summary = pd.read_csv(tmp_path / "out" / "summary.csv")
    assert {"scenario", "condition", "metric", "diff_vs_ref"} <= set(summary.columns)
    assert summary[summary.condition == "u"].diff_vs_ref.notna().all()
