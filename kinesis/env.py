"""Gymnasium environment shared by conventional and learned controllers."""
from dataclasses import dataclass
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .dynamics import State, MotionConfig, advance, wrap_angle, action_to_command, STOP_ACTION, V_MAX, W_MAX
from .routes import scenarios

PREVIEW_DISTANCES = np.array([0.4, 0.8, 1.4, 2.2])
PREVIEW_SCALE = 3.0
# Layout: preview 0:8, cte 8, sin/cos heading 9:11, speeds 11:13,
# previous action 13:15, oldest-to-newest pending commands 15:19,
# remaining distance 19, time remaining 20, delay length 21,
# next required gate in body coordinates 22:24, gates remaining 24.
OBS_SIZE = 25


@dataclass(frozen=True)
class Stress:
    name: str = "nominal"
    position_noise: float = 0.0
    heading_noise: float = 0.0
    delay_steps: int = 0
    response_scale: float = 1.0

    def __post_init__(self):
        if self.delay_steps not in (0, 1, 2):
            raise ValueError("Supported delays are 0, 1, or 2 steps.")
        if self.position_noise < 0 or self.heading_noise < 0 or self.response_scale <= 0:
            raise ValueError("Invalid stress parameters.")


def stresses():
    return [
        Stress(),
        Stress("noise", 0.03, math.radians(3)),
        Stress("delay_200ms", delay_steps=2),
        Stress("slow_response", response_scale=2.5),
    ]


class TrackingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, split="train", stress=None):
        self.split = split
        self.cases = scenarios(split)
        self.stress = stress or Stress()
        self.motion = MotionConfig(tau_v=0.25 * self.stress.response_scale,
                                   tau_w=0.18 * self.stress.response_scale)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_SIZE,), dtype=np.float32)
        self._finished = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        case = options.get("scenario")
        if case is None:
            case = self.cases[int(self.np_random.integers(len(self.cases)))]
        if case.split != self.split:
            raise ValueError("A scenario cannot cross the environment's data split.")
        self.case, self.route = case, case.route()
        heading = self.route.heading_at(0)
        self.state = State(-case.offset * math.sin(heading), case.offset * math.cos(heading),
                           wrap_angle(heading + case.heading_error))
        self.progress = 0.0
        self.gate_distances = np.r_[np.arange(0.5, self.route.length, 0.5), self.route.length]
        self.gates = self.route.at(self.gate_distances)
        self.gate_index = 0
        self.steps = 0
        # Equal physical time allowance for every controller on this route.
        self.max_steps = int(math.ceil((self.route.length / 0.5 + 5) / self.motion.dt))
        self.previous_action = STOP_ACTION.copy()
        self.pending = [STOP_ACTION.copy() for _ in range(self.stress.delay_steps)]
        self._finished = False
        self.history = []
        return self._observation(), self._info(False, "running")

    def _observation(self):
        position = np.array([self.state.x, self.state.y], dtype=float)
        position += self.np_random.normal(0, self.stress.position_noise, 2)
        heading = self.state.heading + self.np_random.normal(0, self.stress.heading_noise)
        preview = self.route.at(self.progress + PREVIEW_DISTANCES) - position
        c, s = math.cos(heading), math.sin(heading)
        body_preview = np.c_[c * preview[:, 0] + s * preview[:, 1],
                             -s * preview[:, 0] + c * preview[:, 1]]
        _, cte, _ = self.route.project(position, self.progress)
        error = wrap_angle(heading - self.route.heading_at(self.progress))
        pending = [STOP_ACTION] * (2 - len(self.pending)) + self.pending
        gate = self.gates[min(self.gate_index, len(self.gates) - 1)] - position
        body_gate = np.array([c * gate[0] + s * gate[1], -s * gate[0] + c * gate[1]])
        obs = np.r_[body_preview.ravel() / PREVIEW_SCALE, cte / 1.5,
                    math.sin(error), math.cos(error), self.state.v / V_MAX, self.state.w / W_MAX,
                    self.previous_action, np.asarray(pending).ravel(),
                    (self.route.length - self.progress) / 15,
                    1 - self.steps / self.max_steps, self.stress.delay_steps / 2, body_gate / PREVIEW_SCALE,
                    1 - self.gate_index / len(self.gates)]
        return obs.astype(np.float32)

    def _info(self, success, reason):
        _, cte, distance = self.route.project([self.state.x, self.state.y], self.progress)
        return {"scenario": self.case.name, "success": bool(success), "reason": reason,
                "progress": self.progress, "route_length": self.route.length,
                "completion": self.gate_index / len(self.gates),
                "progress_ratio": self.progress / self.route.length,
                "gates_passed": self.gate_index, "gates_total": len(self.gates),
                "cross_track_error": cte, "path_distance": distance,
                "elapsed_seconds": self.steps * self.motion.dt}

    def step(self, action):
        if self._finished:
            raise RuntimeError("Call reset before stepping a finished episode.")
        # Validation happens before clipping so NaN and incorrect shapes cannot pass.
        action_to_command(action)
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        last_action = self.previous_action.copy()
        self.pending.append(action.copy())
        applied = self.pending.pop(0)
        self.state = advance(self.state, action_to_command(applied), self.motion)
        old_progress = self.progress
        self.progress, cte, distance = self.route.project(
            [self.state.x, self.state.y], old_progress, radius=2 * V_MAX * self.motion.dt + 0.02)
        self.steps += 1
        end_distance = float(np.linalg.norm(self.route.points[-1] - [self.state.x, self.state.y]))
        old_gate_index = self.gate_index
        while self.gate_index < len(self.gates) and np.linalg.norm(
                self.gates[self.gate_index] - [self.state.x, self.state.y]) <= 0.35:
            self.gate_index += 1
        missed_gate = (self.gate_index < len(self.gates)
                       and self.progress > self.gate_distances[self.gate_index] + 0.5)
        success = (self.gate_index == len(self.gates)
                   and self.progress >= self.route.length - 0.15 and end_distance <= 0.25)
        failure = distance > 1.5 or missed_gate
        terminated = bool(success or failure)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        heading_error = wrap_angle(self.state.heading - self.route.heading_at(self.progress))
        # Stopping has a negative per-step reward. Progress is signed, so reversing
        # and repeating a route interval does not create net progress reward.
        reward = (5 * (self.progress - old_progress) - 0.8 * cte**2
                  - 0.1 * (1 - math.cos(heading_error))
                  - 0.02 * float(np.sum((action - last_action)**2)) - 0.01)
        reward += 2 * (self.gate_index - old_gate_index)
        reward += 10 if success else (-10 if failure else 0)
        self.previous_action = action.copy()
        reason = "success" if success else ("missed_gate" if missed_gate else ("off_route" if failure else ("time_limit" if truncated else "running")))
        info = self._info(success, reason)
        self.history.append({"x": self.state.x, "y": self.state.y, "heading": self.state.heading,
                             "v": self.state.v, "w": self.state.w, "v_command": float(action_to_command(action)[0]),
                             "w_command": float(action_to_command(action)[1]), **info})
        self._finished = terminated or truncated
        return self._observation(), float(reward), terminated, truncated, info
