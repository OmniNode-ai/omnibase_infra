# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A retired lab compose lane has no row in the lane block generator's maps.

The census guard refuses a generator map row for a lane the lane census does
not declare (``orphan_map_entries``). The lab compose lane ``prod`` was retired
on 2026-09-13 and its rows were removed; the ``judge`` compose project was taken
down on 2026-10-21 and removed from the census the same day, and its two rows,
left behind, made every pull request's lab-overlay parity check fail. This
check needs no private overlay, so it runs in every split.
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
)

RETIRED_LANES = frozenset({"prod", "judge"})


@pytest.mark.parametrize("retired", sorted(RETIRED_LANES))
def test_a_retired_lane_has_no_port_or_boundary_row(retired: str) -> None:
    assert retired not in _LANE_PORT_MAP, (
        f"{retired!r} is retired from the lane census but keeps a port row"
    )
    assert retired not in _LANE_BOUNDARY, (
        f"{retired!r} is retired from the lane census but keeps a boundary row"
    )


def test_the_maps_still_cover_the_live_lanes() -> None:
    """Positive control: the maps are not empty, so the absence is real."""
    for live in ("dev", "dogfood", "principal-issuer"):
        assert live in _LANE_PORT_MAP, live
        assert live in _LANE_BOUNDARY, live
