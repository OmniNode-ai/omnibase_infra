# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Systemd coordination tests for the lane-census timer drop-in (OMN-13011).

The ticket requires sharing the OMN-13008 timer unit rather than adding a second
one. These tests assert the coordination contract:
  1. The lane-census ExecStart is delivered as a drop-in for the SHARED
     onex-disk-gc.service (not a new .timer / .service).
  2. The drop-in invokes scripts/lane-census-check.sh.
  3. The installer refuses to install a second timer and depends on the base unit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_LANE_CENSUS_DIR = _REPO / "deploy" / "lane-census"
_DROPIN = _LANE_CENSUS_DIR / "onex-disk-gc.service.d" / "20-lane-census.conf"
_INSTALLER = _LANE_CENSUS_DIR / "install-lane-census.sh"


def test_dropin_targets_shared_disk_gc_service() -> None:
    """The drop-in lives under onex-disk-gc.service.d (shared unit), not a new unit."""
    assert _DROPIN.exists(), f"missing drop-in: {_DROPIN}"
    assert _DROPIN.parent.name == "onex-disk-gc.service.d"
    body = _DROPIN.read_text(encoding="utf-8")
    assert "[Service]" in body
    assert "ExecStart=" in body


def test_dropin_invokes_lane_census_check() -> None:
    body = _DROPIN.read_text(encoding="utf-8")
    assert "scripts/lane-census-check.sh" in body


def test_no_second_timer_unit_shipped() -> None:
    """Coordination rule: do NOT add a second timer. No .timer file in this dir."""
    timers = list(_LANE_CENSUS_DIR.glob("*.timer"))
    assert not timers, (
        f"lane-census must share the onex-disk-gc.timer, not add its own: {timers}"
    )
    # And no standalone lane-census .service either.
    services = [
        p
        for p in _LANE_CENSUS_DIR.glob("*.service")
        if p.name != "onex-disk-gc.service"
    ]
    assert not services, f"unexpected standalone service unit(s): {services}"


def test_installer_depends_on_base_unit() -> None:
    """Installer fails fast if the base onex-disk-gc.service is not installed."""
    body = _INSTALLER.read_text(encoding="utf-8")
    assert "onex-disk-gc.service" in body
    assert "install-disk-gc.sh" in body  # points operator at OMN-13008's installer
    assert "daemon-reload" in body
