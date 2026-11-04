# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The age bound on lane scratch under /tmp (OMN-22123)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_DIR = Path(__file__).resolve().parents[3] / "deploy" / "disk-gc"
_ROOT = "/" + "tmp"  # the path the rules bound, not a scratch location used here
_CONF = _DIR / "onex-lane-tmp.tmpfiles.conf"


def _rules() -> list[list[str]]:
    return [
        line.split()
        for line in _CONF.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def test_every_rule_is_an_age_bound_under_tmp() -> None:
    rules = _rules()
    assert rules
    for kind, path, mode, uid, gid, age in rules:
        assert kind == "e"
        assert path.startswith(_ROOT + "/")
        assert (mode, uid, gid) == ("-", "-", "-")
        assert age.endswith(("h", "d"))


def test_pytest_trees_are_bound_tighter_than_the_catch_all() -> None:
    ages = {path: age for _, path, _, _, _, age in _rules()}
    assert ages[_ROOT + "/pytest-of-*"] == "12h"
    assert ages[_ROOT + "/*"] == "3d"
