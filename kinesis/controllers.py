"""Controllers consume only the same public observation vector."""
from dataclasses import dataclass
import math
import numpy as np
from .dynamics import command_to_action, W_MAX
from .env import PREVIEW_DISTANCES, PREVIEW_SCALE


@dataclass
class PurePursuit:
    lookahead: float = 0.8
    cruise_speed: float = 0.8

    def predict(self, observation):
        preview = np.asarray(observation[:8]).reshape(4, 2) * PREVIEW_SCALE
        target = np.array([np.interp(self.lookahead, PREVIEW_DISTANCES, preview[:, k]) for k in range(2)])
        squared_distance = max(float(np.dot(target, target)), 1e-8)
        curvature = 2 * target[1] / squared_distance
        alignment = max(0.15, math.cos(math.atan2(target[1], target[0])))
        remaining = max(0.0, float(observation[19]) * 15)
        speed = min(self.cruise_speed * alignment,
                    0.9 * W_MAX / max(abs(curvature), 1e-6),
                    max(0.15, math.sqrt(2 * 0.8 * remaining)))
        return command_to_action(speed, speed * curvature)


class SACController:
    def __init__(self, model):
        self.model = model

    def predict(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action
