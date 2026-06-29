"""Synthetic participant generator — see tasks/epic-2-dataset/S2.1-participant-generator.md.

Reproducible: the same (n, seed, scenario) always yields identical participants.
Scenarios shift the underlying distributions to stress different fairness situations.
"""

from __future__ import annotations

import numpy as np

from ..models import Participant

# Fixed attribute vocabularies (kept here so metrics/explanations share one source).
SKILL_NAMES: tuple[str, ...] = ("coding", "design", "communication", "leadership")
EXPERIENCE_LEVELS: int = 5  # ordinal levels 0..4
NATIONALITIES: tuple[str, ...] = ("IT", "DE", "FR", "ES", "PL", "UA")
GENDERS: tuple[str, ...] = ("F", "M", "X")

SCENARIOS: tuple[str, ...] = ("balanced", "skewed-skill", "minority-underrepresented")

# Probability a participant expresses any preferences at all.
_PREF_PROB = 0.5


def generate(n: int, *, seed: int = 42, scenario: str = "balanced") -> list[Participant]:
    """Generate ``n`` reproducible synthetic participants for the given scenario."""
    if n <= 0:
        raise ValueError("n must be positive")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")

    rng = np.random.default_rng(seed)
    ids = [f"p{idx:03d}" for idx in range(n)]

    skills = _gen_skills(rng, n, scenario)
    experience = _gen_experience(rng, n)
    nationalities = _gen_nationalities(rng, n, scenario)
    genders = rng.choice(GENDERS, size=n)

    participants: list[Participant] = []
    for i, pid in enumerate(ids):
        participants.append(
            Participant(
                id=pid,
                skills={name: round(float(skills[i, j]), 3) for j, name in enumerate(SKILL_NAMES)},
                experience=int(experience[i]),
                diversity_attrs={"nationality": str(nationalities[i]), "gender": str(genders[i])},
                preferences=_gen_preferences(rng, pid, ids),
            )
        )
    return participants


def _gen_skills(rng: np.random.Generator, n: int, scenario: str) -> np.ndarray:
    """Return an (n, len(SKILL_NAMES)) array of skill values in [0, 1]."""
    k = len(SKILL_NAMES)
    if scenario == "skewed-skill":
        # ~15% high performers (Beta(5,2)), the rest lower (Beta(2,5)).
        is_high = rng.random(n) < 0.15
        low = rng.beta(2.0, 5.0, size=(n, k))
        high = rng.beta(5.0, 2.0, size=(n, k))
        return np.where(is_high[:, None], high, low)
    # balanced / minority-underrepresented: centred, moderate spread.
    return np.clip(rng.normal(0.5, 0.18, size=(n, k)), 0.0, 1.0)


def _gen_experience(rng: np.random.Generator, n: int) -> np.ndarray:
    # Slightly more juniors than seniors.
    probs = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    return rng.choice(EXPERIENCE_LEVELS, size=n, p=probs)


def _gen_nationalities(rng: np.random.Generator, n: int, scenario: str) -> np.ndarray:
    if scenario == "minority-underrepresented":
        # One nationality is rare (~3%); the rest split the remainder.
        rare_p = 0.03
        rest = (1.0 - rare_p) / (len(NATIONALITIES) - 1)
        probs = np.array([rare_p] + [rest] * (len(NATIONALITIES) - 1))
    else:
        probs = np.full(len(NATIONALITIES), 1.0 / len(NATIONALITIES))
    return rng.choice(NATIONALITIES, size=n, p=probs)


def _gen_preferences(rng: np.random.Generator, pid: str, ids: list[str]) -> dict[str, list[str]]:
    """Optional likes/dislikes referencing other participants (never self)."""
    if rng.random() >= _PREF_PROB or len(ids) < 3:
        return {}
    others = [other for other in ids if other != pid]
    likes = rng.choice(others, size=min(2, len(others)), replace=False).tolist()
    remaining = [other for other in others if other not in likes]
    dislikes = (
        rng.choice(remaining, size=1, replace=False).tolist()
        if remaining and rng.random() < 0.4
        else []
    )
    prefs: dict[str, list[str]] = {}
    if likes:
        prefs["likes"] = likes
    if dislikes:
        prefs["dislikes"] = dislikes
    return prefs
