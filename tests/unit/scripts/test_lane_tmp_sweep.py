# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The age bound on lane scratch under /tmp (OMN-22123)."""

from __future__ import annotations

import os
import re
import subprocess
import time
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
    return {path: _seconds(age) for kind, path, _, _, _, age in _rules() if kind == "e"}


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
        assert kind in ("e", "x")
        assert path.startswith(_ROOT + "/")
        assert (mode, uid, gid) == ("-", "-", "-")
        if kind == "e":
            assert _seconds(age) > 0
        else:
            assert age == "-"


def test_a_malformed_age_is_refused() -> None:
    for bad in ("xh", "12", "0h", "h", "1.5d", "12H"):
        with pytest.raises(AssertionError):
            _seconds(bad)


def test_only_retired_pytest_trees_have_an_age_bound() -> None:
    # Numbered runs are excluded; generic lane/service roots are not matched.
    assert _ages() == {_ROOT + "/pytest-of-*": 12 * 3600}
    assert [r[1] for r in _rules() if r[0] == "x"] == [_ROOT + "/pytest-of-*/pytest-*"]


def test_cleanup_preserves_live_and_unrelated_trees(tmp_path: Path) -> None:
    """Run the shipped rules against a private filesystem root, never the host temp."""
    root = tmp_path / "root"
    scratch = root / "tmp"
    fixtures = {
        "pytest-of-lane/pytest-0/input.db": "live fixture",
        "pytest-of-lane/pytest-0/.lock": str(os.getpid()),
        "pytest-of-lane/garbage-retired/nested/input.db": "retired fixture",
        "pytest-of-lane/garbage-empty/.keep": "retired marker",
        "claude-lane/input.db": "live lane",
        "service-worker/input.db": "service state",
    }
    for relative, contents in fixtures.items():
        path = scratch / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    config = tmp_path / "cleanup.conf"
    # Scale only the age; retain the shipped rule types and path globs.
    config.write_text(
        "\n".join(
            " ".join([*rule[:5], "1s" if rule[0] == "e" else "-"]) for rule in _rules()
        )
        + "\n"
    )
    # ctime cannot be set with utime: let every fixture age beyond the test bound.
    time.sleep(2)
    cleaned = subprocess.run(
        ["systemd-tmpfiles", "--clean", f"--root={root}", str(config)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert cleaned.returncode == 0, cleaned.stdout + cleaned.stderr
    for relative, contents in fixtures.items():
        if "garbage-" in relative:
            assert not (scratch / relative).exists(), "retired scratch must be cleaned"
        else:
            assert (scratch / relative).read_text() == contents
    for retired in ("garbage-retired", "garbage-empty"):
        assert not (scratch / "pytest-of-lane" / retired).exists(), (
            "retired directory must be removed, not left empty"
        )
