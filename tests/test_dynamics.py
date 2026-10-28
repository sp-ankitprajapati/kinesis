import math
import numpy as np
import pytest
from kinesis.dynamics import State, MotionConfig, advance, wrap_angle, action_to_command, V_MAX, W_MAX


def test_straight_motion_matches_analytic_solution():
    state = advance(State(), [1, 0], MotionConfig(dt=2, tau_v=0, tau_w=0))
    assert (state.x, state.y, state.heading) == pytest.approx((2, 0, 0))


def test_constant_turn_matches_circle():
    state = State()
    for _ in range(10):
        state = advance(state, [1, 1], MotionConfig(tau_v=0, tau_w=0))
    assert (state.x, state.y, state.heading) == pytest.approx((math.sin(1), 1 - math.cos(1), 1))


def test_first_order_response_matches_analytic_distance_and_speed():
    config = MotionConfig(dt=1, tau_v=0.25)
    state = advance(State(), [1, 0], config)
    assert state.v == pytest.approx(1 - math.exp(-4))
    assert state.x == pytest.approx(1 - 0.25 * (1 - math.exp(-4)))


def test_substeps_converge_for_changing_twist():
    start = State(v=0.2, w=-0.4)
    results = [advance(start, [1, 1.5], MotionConfig(dt=0.5, substeps=n)) for n in (1, 5, 100)]
    reference = np.array([results[-1].x, results[-1].y])
    errors = [np.linalg.norm(np.array([s.x, s.y]) - reference) for s in results[:2]]
    assert errors[1] < errors[0] / 5


def test_action_and_motion_bounds():
    assert action_to_command([10, -10]) == pytest.approx([V_MAX, -W_MAX])
    state = advance(State(), [50, 50], MotionConfig(dt=10))
    assert 0 <= state.v <= V_MAX
    assert abs(state.w) <= W_MAX


@pytest.mark.parametrize("action", [[float("nan"), 0], [0], [0, float("inf")]])
def test_invalid_actions_are_rejected(action):
    with pytest.raises(ValueError):
        action_to_command(action)


def test_angle_wrapping_and_periodicity():
    assert wrap_angle(math.pi) == pytest.approx(-math.pi)
    assert wrap_angle(-math.pi - 0.1) == pytest.approx(math.pi - 0.1)
    assert wrap_angle(0.2 + 12 * math.pi) == pytest.approx(0.2)
