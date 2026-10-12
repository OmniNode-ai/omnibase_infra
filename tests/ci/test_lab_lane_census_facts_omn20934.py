# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Facts about the lab's own lane manifest that this repository's tools rely on.

OMN-20934: the lab's lane manifest is private deployment config, read from
``config/lab/lane-census`` under ``ONEX_WORKSPACE_CONFIG_ROOT``. These cases were
asserted against the committed copy until it left; they now run where the
overlay is fetched (the ``lab_overlay`` marker, selected by
``lane-census-parity.yml``) and fail loud wherever it is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.helpers.lane_census_fixtures import LAB_MANIFEST

pytestmark = [pytest.mark.unit, pytest.mark.lab_overlay]

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _manifest() -> dict[str, Any]:
    loaded = yaml.safe_load(LAB_MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict) and loaded.get("lanes"), LAB_MANIFEST
    return loaded


def test_the_lab_manifest_loads_through_the_planner() -> None:
    """The real manifest passes the planner's own load-time validation."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    from lane_census_plan import load_manifest, validate_hosts

    manifest = load_manifest(LAB_MANIFEST)
    validate_hosts(manifest)
    assert len(manifest["lanes"]) >= 8


def test_every_migration_gate_is_a_keepalive() -> None:
    """Manifest ratchet: a migration gate stays `keepalive` (OMN-13772).

    Flipping it back to `oneshot` reintroduces the census false-positive on
    every healthy lane; flipping it to `service` would hard-require a container
    whose restart policy is `restart: "no"` on some lanes. Read over every lane
    that declares one, so a lane added later is covered without naming it.
    """
    gates = [
        (lane, svc)
        for lane, spec in _manifest()["lanes"].items()
        for svc in spec.get("services") or []
        if svc["name"].endswith("-migration-gate")
    ]
    assert gates, "positive control: the lab manifest declares migration gates"
    for lane, svc in gates:
        assert svc["kind"] == "keepalive", (
            f"lane {lane!r}: {svc['name']} kind is {svc['kind']!r}, expected "
            "'keepalive' (long-running healthcheck sentinel, OMN-13772)"
        )


def test_the_alarm_lanes_crash_looping_writer_carries_the_tightest_bound() -> None:
    """The container that crash-looped unobserved for nine days is armed first.

    The lab alarm reads its restart bounds from the census lane its shipped
    config names, so the bound that matters is the manifest's.
    """
    import json

    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    from lane_census_plan import restart_bounds_for_lane

    config = json.loads(
        (_SCRIPTS.parent / "config" / "lab_alarm.json").read_text(encoding="utf-8")
    )
    bounds = restart_bounds_for_lane(_manifest(), config["census_lane"])
    assert (
        bounds["omnimarket-projection-savings-writer"]
        < bounds["omnibase-infra-redpanda"]
    )


def test_the_dev_entry_declares_the_protected_delegation_lane() -> None:
    """The desired-state census names the guards beside the lane they guard."""
    text = LAB_MANIFEST.read_text(encoding="utf-8")
    head, _, rest = text.partition("\n  dev:\n")
    assert rest, "the manifest has no dev lane entry"
    preceding = head[-3000:]
    assert "PROTECTED DELEGATION LANE (OMN-20594)" in preceding
    for guard in ("chain-canary.yml", "delegation_chain_canary", "staging delivery"):
        assert guard in preceding, f"the dev entry's comment does not name {guard}"
