"""Smoke tests — confirm the package imports and the CLI parser builds."""

from groupformation import __version__
from groupformation.cli import build_parser
from groupformation.models import Assignment, Participant


def test_version():
    assert __version__


def test_cli_parser_builds():
    parser = build_parser()
    args = parser.parse_args(["generate", "--n", "10", "-o", "data/p.csv"])
    assert args.command == "generate"
    assert args.n == 10


def test_assignment_views():
    a = Assignment({"p1": "g1", "p2": "g1", "p3": "g2"})
    assert a.group_of("p1") == "g1"
    assert sorted(a.groups()["g1"]) == ["p1", "p2"]


def test_participant_requires_id():
    try:
        Participant(id="", skills={}, experience=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty id")
