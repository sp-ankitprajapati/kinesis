"""Planar unicycle kinematics with bounded commands and actuator lag.

This is a differential-drive-style abstraction, not a wheel/motor or vehicle
dynamics model. No slip, collisions, terrain, mass, or tire forces are modelled.
"""
from dataclasses import dataclass
import math
import numpy as np

V_MAX = 1.2  # metres/second
W_MAX = 2.0  # radians/second
STOP_ACTION = np.array([-1.0, 0.0], dtype=np.float32)


def wrap_angle(angle: float) -> float:
    """Map an angle in radians into [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class MotionConfig:
    dt: float = 0.1
    tau_v: float = 0.25
    tau_w: float = 0.18
    substeps: int = 5

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.dt, self.tau_v, self.tau_w)) or self.dt <= 0 or self.tau_v < 0 or self.tau_w < 0:
            raise ValueError("dt must be positive and time constants nonnegative.")
        if self.substeps < 1:
            raise ValueError("At least one integration substep is required.")


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    v: float = 0.0
    w: float = 0.0


def action_to_command(action) -> np.ndarray:
    a = np.asarray(action, dtype=np.float64)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError("Action must be a finite vector of length two.")
    a = np.clip(a, -1, 1)
    return np.array([(a[0] + 1) * V_MAX / 2, a[1] * W_MAX])


def command_to_action(v: float, w: float) -> np.ndarray:
    return np.clip([2 * v / V_MAX - 1, w / W_MAX], -1, 1).astype(np.float32)


def _response(old: float, command: float, tau: float, h: float):
    if tau == 0:
        return command, command
    decay = math.exp(-h / tau)
    # Exact endpoint and interval mean of a first-order response.
    return command + (old - command) * decay, command + (old - command) * tau / h * (1 - decay)


def advance(state: State, command, config: MotionConfig) -> State:
    command = np.asarray(command, dtype=float)
    if command.shape != (2,) or not np.isfinite(command).all():
        raise ValueError("Command must be a finite vector of length two.")
    vc = float(np.clip(command[0], 0, V_MAX))
    wc = float(np.clip(command[1], -W_MAX, W_MAX))
    x, y, theta, v, w = state.x, state.y, state.heading, state.v, state.w
    h = config.dt / config.substeps
    for _ in range(config.substeps):
        v, vm = _response(v, vc, config.tau_v, h)
        w, wm = _response(w, wc, config.tau_w, h)
        delta = wm * h
        # Exact constant-twist arc for each substep, using interval means.
        # With lag, changing twist is approximated; substeps control that error.
        distance = vm * h
        factor = 1.0 if abs(delta) < 1e-10 else math.sin(delta / 2) / (delta / 2)
        x += distance * factor * math.cos(theta + delta / 2)
        y += distance * factor * math.sin(theta + delta / 2)
        theta = wrap_angle(theta + delta)
    return State(x, y, theta, v, w)
