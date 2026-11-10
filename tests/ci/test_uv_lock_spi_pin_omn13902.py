# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The committed uv.lock carries omnibase-spi 0.23.15 because the floor fill
orchestrator imports ModelLaneProcessRunResult, which 0.23.14 lacks (OMN-13902)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_FLOOR = (0, 23, 15)


@pytest.mark.unit
def test_uv_lock_pins_omnibase_spi_at_least_0_23_15() -> None:
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    match = re.search(r'name = "omnibase-spi"\nversion = "([^"]+)"', lock)
    assert match is not None
    version = tuple(int(part) for part in match.group(1).split("."))
    assert version >= EXPECTED_FLOOR


@pytest.mark.unit
def test_pyproject_floor_for_omnibase_spi_matches_lock_floor() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    expected = "omnibase-spi>=0.23.15,<0.24.0"
    assert expected in pyproject["project"]["dependencies"]
    assert expected in pyproject["tool"]["uv"]["override-dependencies"]


@pytest.mark.unit
def test_locked_omnibase_spi_provides_the_fill_orchestrators_lane_process_types() -> (
    None
):
    """Seam with omnimarket's node_floor_fill_orchestrator (orchestration/ports.py).

    The fill imports these three names from omnibase_spi.contracts.services; the
    dispatch venv is built from this lock, so the installed spi must carry them.
    """
    from omnibase_spi.contracts.services import (
        ModelLaneProcessRunResult,
        ModelLaneProcessStartResult,
        ModelLaneStartSpec,
    )

    assert ModelLaneProcessRunResult.__name__ == "ModelLaneProcessRunResult"
    assert ModelLaneProcessStartResult.__name__ == "ModelLaneProcessStartResult"
    assert ModelLaneStartSpec.__name__ == "ModelLaneStartSpec"
