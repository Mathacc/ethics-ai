"""Preset scenarios + persistence — see tasks/epic-2-dataset/S2.2-scenarios-persistence.md.

Scenario distributions live in ``generator.py``; this module re-exports the names and
provides save/load for CSV and JSON. Complex fields (skills, diversity_attrs,
preferences) are stored as JSON strings in CSV columns to round-trip losslessly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..models import Participant
from .generator import SCENARIOS  # noqa: F401  (re-exported for convenience)

_COMPLEX_FIELDS = ("skills", "diversity_attrs", "preferences")


def save(participants: list[Participant], path: str) -> None:
    """Save participants to ``path``; format chosen by extension (.json or .csv)."""
    suffix = Path(path).suffix.lower()
    records = [_to_record(p) for p in participants]
    if suffix == ".json":
        Path(path).write_text(json.dumps(records, indent=2))
    elif suffix == ".csv":
        df = pd.DataFrame(records)
        for field in _COMPLEX_FIELDS:
            df[field] = df[field].map(json.dumps)
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported extension {suffix!r}; use .json or .csv")


def load(path: str) -> list[Participant]:
    """Load participants from ``path``; format chosen by extension (.json or .csv)."""
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        records = json.loads(Path(path).read_text())
    elif suffix == ".csv":
        df = pd.read_csv(path)
        for field in _COMPLEX_FIELDS:
            df[field] = df[field].map(json.loads)
        records = df.to_dict(orient="records")
    else:
        raise ValueError(f"unsupported extension {suffix!r}; use .json or .csv")
    return [_from_record(r) for r in records]


def _to_record(p: Participant) -> dict:
    return {
        "id": p.id,
        "skills": p.skills,
        "experience": p.experience,
        "diversity_attrs": p.diversity_attrs,
        "preferences": p.preferences,
    }


def _from_record(r: dict) -> Participant:
    return Participant(
        id=str(r["id"]),
        skills={k: float(v) for k, v in r["skills"].items()},
        experience=int(r["experience"]),
        diversity_attrs=dict(r.get("diversity_attrs") or {}),
        preferences={k: list(v) for k, v in (r.get("preferences") or {}).items()},
    )
