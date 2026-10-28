import numpy as np
from kinesis.controllers import PurePursuit
from kinesis.env import TrackingEnv
from kinesis.routes import Scenario
from kinesis.evaluation import rollout


def test_metrics_include_failure_instead_of_dropping_it():
    class Stop:
        def predict(self, observation):
            return np.array([-1, 0], dtype=np.float32)
    case = Scenario("validation", 999, "straight", 10, 0.5, 1, 0, 0, 0)
    result = rollout(Stop(), case)
    assert not result["success"]
    assert result["completion"] == 0
    assert result["reason"] == "time_limit"
    assert result["elapsed_s"] == 25
    assert result["steps"] == 250


def test_route_cutting_misses_ordered_gate():
    case = Scenario("train", 999, "sine", 10, 1.2, 1, 0, 0, 0)
    env = TrackingEnv()
    env.reset(options={"scenario": case})
    env.state.heading = 0
    for _ in range(env.max_steps):
        _, _, terminated, truncated, info = env.step([0.5, 0])
        if terminated or truncated:
            break
    assert not info["success"]
    assert info["completion"] < 1
    assert info["reason"] == "missed_gate"


def test_controllers_only_need_public_observation():
    env = TrackingEnv()
    obs, _ = env.reset(seed=1)
    action = PurePursuit().predict(obs.copy())
    assert action.shape == (2,)
    assert np.isfinite(action).all()
    assert np.all(np.abs(action) <= 1)
