"""Core domain models — see tasks/epic-1-project-setup/S1.2-data-models.md."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Participant:
    """A single participant to be placed in a group."""

    id: str
    skills: dict[str, float]              # e.g., {"coding": 0.8, "design": 0.3}
    experience: int                       # ordinal level, e.g., 0..4
    diversity_attrs: dict[str, str] = field(default_factory=dict)  # e.g., {"nationality": "IT"}
    preferences: dict[str, list[str]] = field(default_factory=dict)  # {"likes": [...], "dislikes": [...]}

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Participant.id must be non-empty")


@dataclass
class Group:
    """A group with a fixed capacity."""

    id: str
    capacity: int
    members: list[Participant] = field(default_factory=list)


@dataclass
class Assignment:
    """A mapping of participant -> group, with convenience views."""

    participant_to_group: dict[str, str]   # participant_id -> group_id

    def group_of(self, participant_id: str) -> str:
        return self.participant_to_group[participant_id]

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for pid, gid in self.participant_to_group.items():
            out.setdefault(gid, []).append(pid)
        return out

    # TODO(S1.2): to/from dict round-trip helpers.
