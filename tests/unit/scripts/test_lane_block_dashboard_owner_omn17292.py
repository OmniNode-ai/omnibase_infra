# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17292: the lane table generator declares the ``dashboard-owner`` lane.

The private lane census declared this lane on 2026-10-22. A census lane with no
row in the generator's port and boundary maps is reported by
``undeclared_lanes`` and reds the lab-overlay parity job on every pull request,
so both rows exist, and they say what the lane may and may not be used for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPTS = str(Path(__file__).resolve().parents[3] / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from generate_claude_lane_block import (
    _LANE_BOUNDARY,
    _LANE_PORT_MAP,
    _PORTS_UNDECLARED,
    undeclared_lanes,
)

LANE = "dashboard-owner"


def test_the_lane_has_a_port_row_that_says_its_ports_are_undeclared() -> None:
    # No port block for this lane is recorded in this repository, so the row
    # must say so rather than render an em-dash that reads as "no ports".
    assert _LANE_PORT_MAP[LANE] == _PORTS_UNDECLARED


def test_the_lane_boundary_rules_it_out_as_a_proof_or_grant_source() -> None:
    boundary = _LANE_BOUNDARY[LANE]
    assert "NOT a proof lane" in boundary
    assert "never sourced for stability/prod grants" in boundary


def test_a_manifest_declaring_the_lane_has_no_undeclared_rows() -> None:
    manifest = {"lanes": {LANE: {"hosts": ["host-b"], "services": []}}}
    assert undeclared_lanes(manifest) == []
