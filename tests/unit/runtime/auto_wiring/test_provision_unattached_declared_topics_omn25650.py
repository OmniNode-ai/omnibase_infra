# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot provisioning creates the topics an unwired contract declares outside event_bus (OMN-25650).

A node served by its own host process declares ``runtime_dispatch`` and no
``event_bus.subscribe_topics``, so the runtime never wires it. OMN-23951 made
boot create such a contract's ``event_bus.publish_topics``; nothing created the
topics it declares in ``runtime_dispatch.command_topic``,
``runtime_dispatch.terminal_events``, ``terminal_event``,
``externally_consumed_topics`` or ``published_events``. With the universe warm
off (``ONEX_BOOT_UNIVERSE_PROVISION=0``, the standing dev setting) those topics
were absent on the broker, so nothing the node answers could be read.

The three contracts below carry the real declarations of omnimarket's
``node_lab_fill_effect``, ``node_remote_lane_start_effect`` and
``node_stability_lane_refresh_effect``, copied verbatim, so the check does not
depend on another repository being installed. The RED control is
``test_declared_topics_of_three_real_nodes_are_created_at_boot``: it fails on the
base commit and passes with the change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
from uuid import UUID

import pytest

from omnibase_infra.handlers.boot_topic_provisioning import (
    provision_unattached_publish_topics,
)
from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest

LAB_FILL_EFFECT: Final = """\
name: "node_lab_fill_effect"
node_type: "EFFECT_GENERIC"
runtime_profiles: [effects]
runtime_dispatch:
  command_topic: "onex.cmd.omnimarket.lab-fill-effect-requested.v1"
  terminal_events:
    success: "onex.evt.omnimarket.lab-fill-effect-completed.v1"
    failure: "onex.evt.omnimarket.lab-fill-effect-failed.v1"
published_events:
  - event_type: ModelLabFillStatusWriteResult
    topic: onex.evt.omnimarket.lab-fill-effect-completed.v1
    description: "The outcome of one lab-fill effect operation, answered whatever happened."
externally_consumed_topics:
  - onex.evt.omnimarket.lab-fill-effect-completed.v1
  - onex.evt.omnimarket.lab-fill-effect-failed.v1
"""

REMOTE_LANE_START_EFFECT: Final = """\
name: "node_remote_lane_start_effect"
node_type: "EFFECT_GENERIC"
runtime_profiles: [effects]
runtime_dispatch:
  command_topic: "onex.cmd.omnimarket.remote-lane-host-call-requested.v1"
  terminal_events:
    success: "onex.evt.omnimarket.remote-lane-host-call-completed.v1"
    failure: "onex.evt.omnimarket.remote-lane-host-call-failed.v1"
published_events:
  - event_type: ModelRemoteLaneHostCallResult
    topic: onex.evt.omnimarket.remote-lane-host-call-completed.v1
    description: "The typed result of one host call."
"""

STABILITY_LANE_REFRESH_EFFECT: Final = """\
name: node_stability_lane_refresh_effect
node_type: EFFECT_GENERIC
terminal_event: onex.evt.omnimarket.stability-refresh-completed.v1
runtime_dispatch:
  external_trigger: true
  command_topic: onex.cmd.omnimarket.stability-refresh-requested.v1
event_bus:
  subscribe_topics: []
  publish_topics:
    - onex.evt.omnimarket.stability-refresh-pin-accepted.v1
    - onex.evt.omnimarket.stability-refresh-refused.v1
    - onex.evt.omnimarket.stability-refresh-completed.v1
    - onex.evt.omnimarket.stability-refresh-failed.v1
externally_consumed_topics:
  - onex.evt.omnimarket.stability-refresh-pin-accepted.v1
  - onex.evt.omnimarket.stability-refresh-refused.v1
  - onex.evt.omnimarket.stability-refresh-completed.v1
  - onex.evt.omnimarket.stability-refresh-failed.v1
"""

EXPECTED: Final[dict[str, frozenset[str]]] = {
    "node_lab_fill_effect": frozenset(
        {
            "onex.cmd.omnimarket.lab-fill-effect-requested.v1",
            "onex.evt.omnimarket.lab-fill-effect-completed.v1",
            "onex.evt.omnimarket.lab-fill-effect-failed.v1",
        }
    ),
    "node_remote_lane_start_effect": frozenset(
        {
            "onex.cmd.omnimarket.remote-lane-host-call-requested.v1",
            "onex.evt.omnimarket.remote-lane-host-call-completed.v1",
            "onex.evt.omnimarket.remote-lane-host-call-failed.v1",
        }
    ),
    "node_stability_lane_refresh_effect": frozenset(
        {
            "onex.cmd.omnimarket.stability-refresh-requested.v1",
            "onex.evt.omnimarket.stability-refresh-pin-accepted.v1",
            "onex.evt.omnimarket.stability-refresh-refused.v1",
            "onex.evt.omnimarket.stability-refresh-completed.v1",
            "onex.evt.omnimarket.stability-refresh-failed.v1",
        }
    ),
}


class RecordingProvisioner:
    """Fake provisioner recording which topics boot asked to create."""

    def __init__(self) -> None:
        self.created: list[str] = []

    async def ensure_topic_exists(
        self,
        topic_name: str,
        spec: object | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        self.created.append(topic_name)
        return True


def _manifest(tmp_path: Path) -> ModelAutoWiringManifest:
    contracts = []
    for name, text in (
        ("node_lab_fill_effect", LAB_FILL_EFFECT),
        ("node_remote_lane_start_effect", REMOTE_LANE_START_EFFECT),
        ("node_stability_lane_refresh_effect", STABILITY_LANE_REFRESH_EFFECT),
    ):
        node_dir = tmp_path / name
        node_dir.mkdir()
        path = node_dir / "contract.yaml"
        path.write_text(text, encoding="utf-8")
        contracts.append(
            _parse_contract(
                contract_path=path,
                entry_point_name=name,
                package_name="omnimarket",
                package_version="0.4.321",
            )
        )
    return ModelAutoWiringManifest(contracts=tuple(contracts))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_declared_topics_of_three_real_nodes_are_created_at_boot(
    tmp_path: Path,
) -> None:
    """RED on base: only the event_bus publish topics (stability refresh) were created."""
    provisioner = RecordingProvisioner()

    ensured = await provision_unattached_publish_topics(
        _manifest(tmp_path), (), provisioner, max_concurrent=2
    )

    assert {name: frozenset(topics) for name, topics in ensured.items()} == EXPECTED
    assert set(provisioner.created) == set().union(*EXPECTED.values())


@pytest.mark.asyncio
@pytest.mark.unit
async def test_a_topic_declared_in_two_places_is_created_once(tmp_path: Path) -> None:
    provisioner = RecordingProvisioner()

    await provision_unattached_publish_topics(
        _manifest(tmp_path), (), provisioner, max_concurrent=2
    )

    assert len(provisioner.created) == len(set(provisioner.created))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_an_interleaved_contract_is_left_to_its_own_pass(tmp_path: Path) -> None:
    """Contracts the attach interleave owns are not provisioned here."""
    provisioner = RecordingProvisioner()

    ensured = await provision_unattached_publish_topics(
        _manifest(tmp_path),
        {"node_lab_fill_effect", "node_remote_lane_start_effect"},
        provisioner,
        max_concurrent=2,
    )

    assert set(ensured) == {"node_stability_lane_refresh_effect"}


@pytest.mark.unit
def test_only_topic_names_are_read_from_the_declared_fields() -> None:
    from omnibase_infra.handlers.boot_topic_provisioning import (
        declared_topics_outside_event_bus,
    )

    declared = declared_topics_outside_event_bus(
        {
            "runtime_dispatch": {
                "external_trigger": "Direct Python invocation",
                "command_topic": "onex.cmd.omnimarket.x-requested.v1",
                "terminal_events": {"success": "not a topic"},
            },
            "terminal_event": None,
            "externally_consumed_topics": ["onex.evt.omnimarket.x-done.v1", 3, ""],
            "published_events": ["onex.evt.omnimarket.bare-string.v1"],
        }
    )

    assert declared == (
        "onex.cmd.omnimarket.x-requested.v1",
        "onex.evt.omnimarket.x-done.v1",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_no_provisioner_creates_nothing(tmp_path: Path) -> None:
    assert (
        await provision_unattached_publish_topics(
            _manifest(tmp_path), (), None, max_concurrent=2
        )
        == {}
    )
