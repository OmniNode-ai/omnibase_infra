# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression for the dev runtime incident of OMN-22233 (OMN-21590 and OMN-21597).

omnimarket#4773 (OMN-21597) and omnibase_infra#5547 (OMN-21590) both declared the
``onex.nodes`` entry point ``node_package_announce_effect``. The dev image installs
both packages, discovery rejected the second owner by name and every dev runtime
container reported ``runtime_degraded``.

The fixtures are the recorded duplicate log line and the entry point names omnimarket
declared on ``origin/dev`` at the time. The omnibase_infra side is this repository's
real ``pyproject.toml``, so the test fails while the pair both declare the name and
passes once one package owns it. It does not skip when omnimarket is not installed.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURES = Path(__file__).parent / "fixtures" / "omn21590_omn21597"
_LOG_LINE = re.compile(
    r"DUPLICATE_REGISTRATION prevented: contract='(?P<contract>[^']+)' "
    r"first_owner='(?P<first_owner>[^']+)' "
    r"duplicate_package='(?P<duplicate_package>[^']+)' "
    r"entry_point='(?P<entry_point>[^']+)'"
)


def _recorded_incident() -> dict[str, str]:
    match = _LOG_LINE.search(
        (_FIXTURES / "duplicate_registration_log_line.txt").read_text()
    )
    assert match is not None, "the recorded log line no longer parses"
    return match.groupdict()


def _declared_by_package() -> dict[str, frozenset[str]]:
    infra = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    infra_names = frozenset(infra["project"]["entry-points"]["onex.nodes"])
    market_names = frozenset(
        line
        for line in (_FIXTURES / "omnimarket_onex_nodes_entry_point_names.txt")
        .read_text()
        .splitlines()
        if line and not line.startswith("#")
    )
    return {"omnibase_infra": infra_names, "omnimarket": market_names}


@pytest.mark.unit
def test_recorded_log_line_names_the_two_packages_of_the_pair() -> None:
    incident = _recorded_incident()
    declared = _declared_by_package()
    assert incident["first_owner"] in declared
    assert incident["duplicate_package"] in declared
    assert incident["contract"] == incident["entry_point"]


@pytest.mark.unit
def test_duplicate_entry_point_names_incident_name_has_one_owner() -> None:
    incident = _recorded_incident()
    owners = sorted(
        pkg
        for pkg, names in _declared_by_package().items()
        if incident["entry_point"] in names
    )
    assert len(owners) == 1, (
        f"entry point {incident['entry_point']!r} is declared by {owners}"
    )


@pytest.mark.unit
def test_duplicate_entry_point_names_pair_shares_no_name() -> None:
    declared = _declared_by_package()
    shared = declared["omnibase_infra"] & declared["omnimarket"]
    assert shared == frozenset()
