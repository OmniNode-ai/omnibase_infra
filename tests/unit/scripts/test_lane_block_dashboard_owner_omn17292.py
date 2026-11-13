# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17292, OMN-24214: the lane table generator carries no ``dashboard-owner`` rows.

The private lane census retired the ``dashboard-owner`` lane (OMN-24214). A
generator row for a lane the census no longer declares is reported by
``orphan_map_entries`` and reds the lab-overlay parity job on every pull
request, so both rows are gone and a manifest without the lane reports no
orphan for it.
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
    orphan_map_entries,
)

LANE = "dashboard-owner"


def test_the_port_map_has_no_row_for_the_retired_lane() -> None:
    assert LANE not in _LANE_PORT_MAP


def test_the_boundary_map_has_no_row_for_the_retired_lane() -> None:
    assert LANE not in _LANE_BOUNDARY


def test_a_manifest_without_the_lane_has_no_orphan_for_it() -> None:
    manifest = {"lanes": {"dev": {"hosts": ["host-a"], "services": []}}}
    assert LANE not in {lane for lane, _ in orphan_map_entries(manifest)}
