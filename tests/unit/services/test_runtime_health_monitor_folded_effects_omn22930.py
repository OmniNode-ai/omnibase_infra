# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22930: the runtime's health reads the effects role it carries.

On primary dev the effects nodes run inside main. The health monitor behind
``/health`` (and so behind the effects port main now publishes) must therefore
expect the effects contracts' consumers and report unhealthy when one is
missing; a monitor that read only main-owned contracts would stay healthy over
a lost effects handler. The fold is read from the rendered dev policy, never
spelled here.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.services.service_runtime_health_monitor import (
    ServiceRuntimeHealthMonitor,
    _expected_consumer_groups_from_manifest,
)
from scripts.render_runtime_policy_env import load_contract, render_env
from tests.unit.services.test_service_runtime_health_monitor import (
    ConsumerGroupSnapshot,
    _make_discovered_contract,
)

pytestmark = pytest.mark.unit

_MONITOR = "omnibase_infra.services.service_runtime_health_monitor"
MAIN_NODE = "node_main_owned"
EFFECTS_NODE = "node_effects_owned"


def _manifest() -> ModelAutoWiringManifest:
    return ModelAutoWiringManifest(
        contracts=(
            _make_discovered_contract(
                name=MAIN_NODE, topic="onex.cmd.omnimarket.main-owned.v1"
            ),
            _make_discovered_contract(
                name=EFFECTS_NODE,
                topic="onex.cmd.omnimarket.effects-owned.v1",
                runtime_profiles=("effects",),
            ),
        ),
        errors=(),
    )


async def _verdict(
    monkeypatch: pytest.MonkeyPatch, *, folded: str, effects_handler_present: bool
):
    monkeypatch.setenv("RUNTIME_PROFILE", "main")
    monkeypatch.setenv("ONEX_RUNTIME_FOLDED_PROFILES", folded)
    manifest = _manifest()
    groups = {
        expected.node_name: expected.group_id
        for expected in _expected_consumer_groups_from_manifest(manifest)
    }
    present = [groups[MAIN_NODE]]
    if effects_handler_present:
        present.append(groups[EFFECTS_NODE])
    snapshots = [ConsumerGroupSnapshot(group_id=g, state="STABLE") for g in present]
    monitor = ServiceRuntimeHealthMonitor(bootstrap_servers="localhost:9092")
    with (
        patch(f"{_MONITOR}._discover_contracts", return_value=manifest),
        patch(f"{_MONITOR}._list_consumer_group_snapshots", return_value=snapshots),
    ):
        return await monitor.run_once()


def _dev_folded_profiles() -> str:
    return render_env(load_contract())["DEV_RUNTIME_MAIN_FOLDED_PROFILES"]


@pytest.mark.asyncio
async def test_healthy_when_main_and_the_effects_handler_are_both_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = await _verdict(
        monkeypatch, folded=_dev_folded_profiles(), effects_handler_present=True
    )
    assert event.uncovered_topic_count == 0
    assert event.status == "HEALTHY"


@pytest.mark.asyncio
async def test_unhealthy_when_one_effects_handler_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = await _verdict(
        monkeypatch, folded=_dev_folded_profiles(), effects_handler_present=False
    )
    assert event.status != "HEALTHY"
    assert event.uncovered_topic_count == 1
    coverage = [d for d in event.dimensions if d.name == "topic_coverage"]
    assert coverage and EFFECTS_NODE in coverage[0].detail


@pytest.mark.asyncio
async def test_failing_control_without_the_fold_the_lost_handler_goes_unseen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FAILING CONTROL: the same lost handler reads healthy when the role is not folded.

    This is what makes the fold the thing under test: a runtime that does not
    carry the effects role has no business reporting on it.
    """
    event = await _verdict(monkeypatch, folded="", effects_handler_present=False)
    assert event.uncovered_topic_count == 0
    assert event.status == "HEALTHY"
