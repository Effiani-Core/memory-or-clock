"""Bounded growth controls. Plateau detection is a heuristic, not a novelty claim."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import math


@dataclass
class GrowthPolicy:
    kind: str = "none"
    node: str = "memory"
    add_count: int = 1
    interval: int = 10
    patience: int = 5
    min_improvement: float = 0.01
    cooldown: int = 5
    max_events: int = 2
    last_growth: int = 0
    events: int = 0
    history: list[float] = field(default_factory=list)
    last_observation: int = 0

    def __post_init__(self):
        if self.kind not in ("none", "fixed_schedule", "plateau"):
            raise ValueError("Growth kind must be none, fixed_schedule or plateau")
        if not isinstance(self.node, str) or not self.node:
            raise ValueError("Growth node must be nonempty")
        for name in ("add_count", "interval", "patience", "cooldown"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.patience < 2:
            raise ValueError("plateau patience must be at least 2 observations")
        if type(self.max_events) is not int or self.max_events < 0:
            raise ValueError("max_events must be a nonnegative integer")
        if not math.isfinite(self.min_improvement) or self.min_improvement < 0:
            raise ValueError("min_improvement must be finite and nonnegative")

    def observe(self, update: int, score: float) -> bool:
        """Call once per training update with mean training return (larger is better)."""
        if type(update) is not int or update <= self.last_observation:
            raise ValueError("Growth observations require strictly increasing positive update numbers")
        if not math.isfinite(score):
            raise ValueError("Growth score must be finite")
        self.last_observation = update
        self.history.append(float(score))
        self.history = self.history[-self.patience:]
        if self.kind == "none" or self.events >= self.max_events or update - self.last_growth < self.cooldown:
            return False
        if self.kind == "fixed_schedule":
            return update % self.interval == 0
        if len(self.history) < self.patience:
            return False
        # Compare averages of first and second halves, not a single unlucky episode.
        split = len(self.history) // 2
        early, late = self.history[:split], self.history[split:]
        improvement = sum(late) / len(late) - sum(early) / len(early)
        return improvement < self.min_improvement

    def record_growth(self, update: int):
        if update != self.last_observation or update <= self.last_growth or self.events >= self.max_events:
            raise ValueError("Growth must correspond to the latest observation within event budget")
        self.events += 1
        self.last_growth = update
        self.history.clear()

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)


def make_growth_policy(config=None):
    return GrowthPolicy(**(config or {}))
