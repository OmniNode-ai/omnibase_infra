# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The age bound on lane scratch under /tmp (OMN-22123)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_DIR = Path(__file__).resolve().parents[3] / "deploy" / "disk-gc"
_ROOT = "/" + "tmp"  # the path the rules bound, not a scratch location used here
_CONF = _DIR / "onex-lane-tmp.tmpfiles.conf"
_UNIT_SECONDS = {"h": 3600, "d": 86400}
_AGE = re.compile(r"^([1-9][0-9]*)([hd])$")


def _parse(text: str) -> list[list[str]]:
    """Rule lines as field lists; blank and comment lines (indented or not) are skipped.

    A rule line carries exactly six fields (type, path, mode, uid, gid, age), so a
    trailing comment or a missing field fails here with the line named, not as an
    unpack error further on.
    """
    rules = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        assert len(fields) == 6, (
            f"line {number} has {len(fields)} fields, not 6: {line!r}"
        )
        rules.append(fields)
    return rules


def _rules() -> list[list[str]]:
    return _parse(_CONF.read_text())


def _seconds(age: str) -> int:
    match = _AGE.match(age)
    assert match, f"{age!r} is not a whole number of hours (h) or days (d)"
    return int(match.group(1)) * _UNIT_SECONDS[match.group(2)]


def _ages() -> dict[str, int]:
    return {path: _seconds(age) for _, path, _, _, _, age in _rules()}


def test_the_parser_skips_indented_comments_and_names_a_malformed_line() -> None:
    text = "# header\n   # indented comment\n\ne /scratch/a - - - 1h\n"
    assert _parse(text) == [["e", "/scratch/a", "-", "-", "-", "1h"]]
    with pytest.raises(AssertionError, match="line 1 has 8 fields"):
        _parse("e /scratch/a - - - 1h # trailing\n")
    with pytest.raises(AssertionError, match="line 1 has 5 fields"):
        _parse("e /scratch/a - - 1h\n")


def test_every_rule_is_an_age_bound_under_tmp() -> None:
    rules = _rules()
    assert rules
    for kind, path, mode, uid, gid, age in rules:
        assert kind == "e"
        assert path.startswith(_ROOT + "/")
        assert (mode, uid, gid) == ("-", "-", "-")
        assert _seconds(age) > 0


def test_a_malformed_age_is_refused() -> None:
    for bad in ("xh", "12", "0h", "h", "1.5d", "12H"):
        with pytest.raises(AssertionError):
            _seconds(bad)


def test_the_three_rules_are_bound_tightest_first() -> None:
    ages = _ages()
    assert set(ages) == {_ROOT + "/pytest-of-*", _ROOT + "/claude-*", _ROOT + "/*"}
    assert ages[_ROOT + "/pytest-of-*"] < ages[_ROOT + "/claude-*"] < ages[_ROOT + "/*"]
    assert ages[_ROOT + "/pytest-of-*"] == 12 * 3600
    assert ages[_ROOT + "/claude-*"] == 2 * 86400
    assert ages[_ROOT + "/*"] == 3 * 86400
