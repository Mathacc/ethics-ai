"""Configuration loading — see tasks/epic-1-project-setup/S1.3-config-schema.md."""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class Config:
    seed: int = 42
    group_size: int | None = 6
    n_groups: int | None = None
    weights: dict[str, float] = field(
        default_factory=lambda: {"skill_balance": 1.0, "diversity": 1.0, "preference": 0.5}
    )
    local_search: dict = field(default_factory=dict)
    simulated_annealing: dict = field(default_factory=dict)
    ilp: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    def __post_init__(self) -> None:
        if self.group_size is None and self.n_groups is None:
            raise ValueError("Set either group_size or n_groups in the config")
