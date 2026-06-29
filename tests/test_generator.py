"""Tests for S2.1 generator + S2.2 persistence."""

from collections import Counter

import pytest

from groupformation.data import scenarios
from groupformation.data.generator import NATIONALITIES, SKILL_NAMES, generate
from groupformation.models import Participant


def test_count_and_unique_ids():
    ps = generate(50, seed=1, scenario="balanced")
    assert len(ps) == 50
    assert len({p.id for p in ps}) == 50


def test_reproducible_same_seed():
    a = generate(30, seed=7, scenario="balanced")
    b = generate(30, seed=7, scenario="balanced")
    assert a == b


def test_different_seed_differs():
    a = generate(30, seed=7, scenario="balanced")
    b = generate(30, seed=8, scenario="balanced")
    assert a != b


def test_skill_keys_and_ranges():
    for p in generate(40, seed=2, scenario="balanced"):
        assert set(p.skills) == set(SKILL_NAMES)
        assert all(0.0 <= v <= 1.0 for v in p.skills.values())


def test_preferences_never_self_referential():
    for p in generate(40, seed=3, scenario="balanced"):
        flat = [x for v in p.preferences.values() for x in v]
        assert p.id not in flat


def test_minority_scenario_underrepresents_one_nationality():
    ps = generate(400, seed=4, scenario="minority-underrepresented")
    counts = Counter(p.diversity_attrs["nationality"] for p in ps)
    rarest = min(counts, key=counts.get)
    # The rare nationality should be well below an even split.
    assert counts[rarest] < 400 / len(NATIONALITIES) / 2


def test_skewed_skill_has_high_performers():
    balanced = generate(300, seed=5, scenario="balanced")
    skewed = generate(300, seed=5, scenario="skewed-skill")

    def top_skill(ps):
        return [max(p.skills.values()) for p in ps]

    # Skewed scenario should produce more very-high-skill participants.
    assert sum(v > 0.85 for v in top_skill(skewed)) > sum(v > 0.85 for v in top_skill(balanced))


def test_invalid_args():
    with pytest.raises(ValueError):
        generate(0, seed=1)
    with pytest.raises(ValueError):
        generate(10, seed=1, scenario="nope")


@pytest.mark.parametrize("ext", [".json", ".csv"])
def test_persistence_round_trip(tmp_path, ext):
    ps = generate(25, seed=9, scenario="skewed-skill")
    path = str(tmp_path / f"participants{ext}")
    scenarios.save(ps, path)
    loaded = scenarios.load(path)
    assert loaded == ps
    assert all(isinstance(p, Participant) for p in loaded)
