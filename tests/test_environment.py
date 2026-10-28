import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env
from kinesis.env import TrackingEnv, Stress, OBS_SIZE
from kinesis.dynamics import STOP_ACTION, State
from kinesis.routes import Route, Scenario, scenarios
from kinesis.controllers import PurePursuit


def straight_case(split="train"):
    return Scenario(split, 999, "straight", 10, 0.5, 1, 0, 0, 0)


def test_gymnasium_and_sb3_contract():
    check_env(TrackingEnv(), warn=True)


def test_split_manifests_are_disjoint_and_hold_out_route_family():
    fingerprints = [{s.route().fingerprint() for s in scenarios(split)}
                    for split in ("train", "validation", "test")]
    assert all(not fingerprints[i] & fingerprints[j] for i in range(3) for j in range(i))
    assert all(s.family != "double_sine" for s in scenarios("train") + scenarios("validation"))
    assert any(s.family == "double_sine" for s in scenarios("test"))


def test_cross_split_reset_rejected():
    with pytest.raises(ValueError, match="split"):
        TrackingEnv("train").reset(options={"scenario": straight_case("test")})


def test_reset_seed_is_reproducible_under_noise():
    env = TrackingEnv(stress=Stress("noise", 0.03, 0.05))
    a, _ = env.reset(seed=42)
    b, _ = env.reset(seed=42)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (OBS_SIZE,)
    assert np.isfinite(a).all()


def test_delay_queue_is_observed_and_applied_at_correct_time():
    env = TrackingEnv(stress=Stress("delay", delay_steps=2))
    obs, _ = env.reset(options={"scenario": straight_case()})
    assert obs[21] == 1
    for i in range(2):
        obs, _, _, _, _ = env.step([1, 0])
        assert env.state.x == pytest.approx(0)
    np.testing.assert_array_equal(obs[15:19], [1, 0, 1, 0])
    env.step(STOP_ACTION)
    assert env.state.x > 0
    assert env.state.v > 0


def test_stopping_is_not_successful_or_profitable():
    env = TrackingEnv()
    env.reset(options={"scenario": straight_case()})
    total = 0
    for _ in range(env.max_steps):
        _, reward, terminated, truncated, info = env.step(STOP_ACTION)
        total += reward
        if terminated or truncated:
            break
    assert total < 0
    assert not terminated and truncated
    assert not info["success"]
    assert info["completion"] == pytest.approx(0)
    with pytest.raises(RuntimeError):
        env.step(STOP_ACTION)


def test_endpoint_teleport_cannot_complete_route():
    env = TrackingEnv()
    env.reset(options={"scenario": straight_case()})
    env.state = State(x=10)
    _, _, terminated, _, info = env.step(STOP_ACTION)
    assert terminated
    assert not info["success"]
    assert info["completion"] < 0.05


def test_local_projection_cannot_jump_at_crossing():
    route = Route([[-2, -2], [2, 2], [-2, 2], [2, -2]])
    anchor = 2.8
    progress, _, _ = route.project([0, 0], anchor, radius=0.2)
    assert abs(progress - anchor) <= 0.2 + 1e-8
    assert progress < 4


def test_pure_pursuit_finishes_straight_route():
    env = TrackingEnv()
    obs, _ = env.reset(options={"scenario": straight_case()})
    controller = PurePursuit()
    while True:
        obs, _, terminated, truncated, info = env.step(controller.predict(obs))
        if terminated or truncated:
            break
    assert info["success"]
    assert max(abs(s["cross_track_error"]) for s in env.history) < 1e-6


def test_controllers_receive_equal_time_and_action_limits():
    a, b = TrackingEnv(), TrackingEnv(stress=Stress("delay", delay_steps=2))
    a.reset(options={"scenario": straight_case()})
    b.reset(options={"scenario": straight_case()})
    assert a.max_steps == b.max_steps
    np.testing.assert_array_equal(a.action_space.low, b.action_space.low)
    np.testing.assert_array_equal(a.action_space.high, b.action_space.high)
