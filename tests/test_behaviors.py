import math

import pytest

from matrss.behaviors import Drifting, OnOff, build_behavior
from matrss.experiment import build_providers


def test_onoff_schedule_repeats():
    b = OnOff("onoff", p_on=0.9, p_off=0.1, good=3, bad=2)
    assert [b.p_success(t) for t in range(10)] == [0.9, 0.9, 0.9, 0.1, 0.1] * 2


def test_onoff_phase_advances_schedule():
    b = OnOff("onoff", 0.9, 0.1, good=3, bad=2, phase=0.4)         # 0.4 x 5 = 2 rounds ahead
    assert [b.p_success(t) for t in range(5)] == [0.9, 0.1, 0.1, 0.9, 0.9]


@pytest.mark.parametrize("kwargs", [{"good": 0}, {"bad": -1}, {"phase": 1.0}])
def test_onoff_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        build_behavior("onoff", **kwargs)


def test_infinite_rate_switches_abruptly():
    b = Drifting("degrading", p_start=0.95, p_end=0.05, onset=10, rate=math.inf)
    assert (b.p_success(9), b.p_success(10), b.p_success(500)) == (0.95, 0.05, 0.05)


def test_stagger_spreads_onoff_phases():
    beh = build_providers([{"type": "onoff", "count": 4, "good": 6, "bad": 2, "stagger": True}])
    assert [b.phase for b in beh] == [0.0, 0.25, 0.5, 0.75]
    for t in range(16):                            # exactly one attacker is bad at any time
        assert sum(b.p_success(t) < 0.5 for b in beh) == 1


def test_stagger_requires_onoff():
    with pytest.raises(ValueError, match="stagger"):
        build_providers([{"type": "honest", "count": 2, "stagger": True}])
