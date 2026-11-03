# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A forward deploy across the fold is not refused by a pre-fold snapshot (OMN-21288).

Measured on dev 2026-11-03: the live family was unhealthy, so prepare fell
back to last-healthy.json, written before the fold and before the
folded_profiles field existed, and every forward deploy refused with "refusing
to pair its image with this overlay". Driven through the contract-routed
``prepare`` operation with the fake host seam of the OMN-21279 integration
test: prepare holds the snapshot, and the restore still refuses the pairing
without tagging or starting anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.handlers import handler_dev_boot_guard as guard_mod
from omnibase_infra.models.model_dev_boot_request import ModelDevBootRequest
from tests.integration.handlers.test_handler_dev_boot_guard_omn21279 import (
    PROJECT,
    REVISION,
    _container,
    _guard,
    _Host,
    _routes,
)

pytestmark = pytest.mark.integration

FOLDED = frozenset(
    {
        "tenant-projection",
        "projection-writer-live-events",
        "projection-writer-registration",
        "projection-writer-savings",
        "projection-writer-tenant-credentials",
        "projection-writer-tenant-registry",
    }
)


def _plant_legacy_snapshot(guard: guard_mod.HandlerDevBootGuard) -> None:
    """The dev snapshot's shape: written before folded_profiles existed."""
    guard.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    guard.snapshot_path.write_text(
        json.dumps(
            {
                "project": PROJECT,
                "revision": REVISION,
                "container_ids": [f"id-{s}" for s in guard_mod.DEV_BOOT_SERVICES],
                "images": {"retained:runtime": "sha256:old"},
                "compose": {"name": PROJECT, "services": {}},
            }
        )
    )


def _unhealthy_folded_host() -> _Host:
    return _Host(
        containers=[_container(s, healthy=False) for s in guard_mod.DEV_BOOT_SERVICES]
    )


def test_prepare_holds_a_pre_fold_snapshot_instead_of_refusing(tmp_path: Path) -> None:
    host = _unhealthy_folded_host()
    guard = _guard(host, tmp_path)
    guard.execution.lane_folded_profiles = lambda lane: FOLDED
    _plant_legacy_snapshot(guard)
    result = _routes()["prepare"].handle(
        guard, ModelDevBootRequest(operation="prepare")
    )
    assert result.status == "pass"
    assert guard.point.current is not None
    assert guard.point.current.revision == REVISION
    assert guard.point.current.folded_profiles == ()
    assert not any("up" in argv for argv in host.calls)
    assert host.tagged == []


def test_the_restore_still_refuses_the_cross_fold_pairing(tmp_path: Path) -> None:
    host = _unhealthy_folded_host()
    guard = _guard(host, tmp_path)
    guard.execution.lane_folded_profiles = lambda lane: FOLDED
    _plant_legacy_snapshot(guard)
    guard.handle(ModelDevBootRequest(operation="prepare"))
    with pytest.raises(guard_mod.DevBootFailedError, match="pair its image"):
        guard._restore()
    assert not any("up" in argv for argv in host.calls)
    assert host.tagged == []
