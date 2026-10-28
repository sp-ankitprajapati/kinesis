"""Reference geometry and fixed, disjoint scenario manifests."""
from dataclasses import dataclass
import hashlib
import math
import numpy as np


class Route:
    def __init__(self, points):
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
            raise ValueError("A route needs at least two finite 2D points.")
        delta = np.diff(p, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        if np.any(lengths < 1e-9):
            raise ValueError("Consecutive route points must be distinct.")
        self.points = p
        self.delta = delta
        self.segment_lengths = lengths
        self.s = np.r_[0.0, np.cumsum(lengths)]
        self.length = float(self.s[-1])

    def at(self, distances):
        s = np.clip(np.asarray(distances, dtype=float), 0, self.length)
        return np.stack([np.interp(s, self.s, self.points[:, k]) for k in range(2)], axis=-1)

    def heading_at(self, distance):
        i = int(np.clip(np.searchsorted(self.s, distance, side="right") - 1, 0, len(self.delta) - 1))
        return math.atan2(self.delta[i, 1], self.delta[i, 0])

    def project(self, position, anchor=None, radius=0.6):
        """Project onto a local route interval to prevent jumps at crossings."""
        if anchor is None:
            lo, hi = 0, len(self.delta)
        else:
            lo = max(0, int(np.searchsorted(self.s, anchor - radius) - 1))
            hi = min(len(self.delta), int(np.searchsorted(self.s, anchor + radius) + 1))
        p, d = self.points[lo:hi], self.delta[lo:hi]
        offsets = np.asarray(position) - p
        fraction = np.clip(np.sum(offsets * d, axis=1) / np.sum(d * d, axis=1), 0, 1)
        if anchor is not None:
            fraction = np.clip(
                fraction,
                np.clip((anchor - radius - self.s[lo:hi]) / self.segment_lengths[lo:hi], 0, 1),
                np.clip((anchor + radius - self.s[lo:hi]) / self.segment_lengths[lo:hi], 0, 1),
            )
        projection = p + fraction[:, None] * d
        distances = np.linalg.norm(np.asarray(position) - projection, axis=1)
        j = int(np.argmin(distances))
        tangent = d[j] / self.segment_lengths[lo + j]
        signed_error = tangent[0] * (position[1] - projection[j, 1]) - tangent[1] * (position[0] - projection[j, 0])
        return float(self.s[lo + j] + fraction[j] * self.segment_lengths[lo + j]), float(signed_error), float(distances[j])

    def fingerprint(self):
        return hashlib.sha256(self.points.round(7).tobytes()).hexdigest()


@dataclass(frozen=True)
class Scenario:
    split: str
    index: int
    family: str
    length_parameter: float
    amplitude: float
    frequency: float
    rotation: float
    offset: float
    heading_error: float

    @property
    def name(self):
        return f"{self.split}-{self.index:03d}-{self.family}"

    def route(self):
        t = np.linspace(0, 1, 401)
        length = self.length_parameter
        if self.family == "arc":
            angle = self.amplitude
            radius = length / angle
            x = radius * np.sin(angle * t)
            y = radius * (1 - np.cos(angle * t))
        else:
            x = length * t
            if self.family == "straight":
                y = np.zeros_like(t)
            elif self.family == "sine":
                y = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
            elif self.family == "double_sine":
                # Held-out family: two spatial frequencies.
                y = self.amplitude * (np.sin(2 * np.pi * t) + 0.3 * np.sin(4 * np.pi * t))
            else:
                raise ValueError(self.family)
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        return Route(np.c_[c * x - s * y, s * x + c * y])


def scenarios(split: str):
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    seed, count = {"train": (1101, 24), "validation": (2202, 12), "test": (3303, 24)}[split]
    rng = np.random.default_rng(seed)
    families = ("straight", "sine", "arc", "double_sine") if split == "test" else ("straight", "sine", "arc")
    result = []
    for i in range(count):
        family = families[i % len(families)]
        amplitude = float(rng.uniform(0.5, 1.2)) if family != "arc" else float(rng.uniform(-1.8, 1.8))
        if family == "arc" and abs(amplitude) < 0.4:
            amplitude = math.copysign(0.4, amplitude)
        result.append(Scenario(split, i, family, float(rng.uniform(10, 14)), amplitude,
                               float(rng.uniform(0.6, 1.1)), float(rng.uniform(-math.pi, math.pi)),
                               float(rng.uniform(-0.35, 0.35)), float(rng.uniform(-0.3, 0.3))))
    return result
