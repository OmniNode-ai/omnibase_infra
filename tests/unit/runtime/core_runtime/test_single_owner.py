# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the single-owner boot assertions (OMN-14758 §c.3, OMN-14771 §D1=4b)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.runtime.runtime_dispatch import DispatchRoute
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.core_runtime.single_owner import (
    CORE_RUNTIME_GROUP,
    assert_single_owner_split,
)

TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"
# A genuine fan-out event topic (>1 subscriber) — the S8 §4b case.
FANOUT_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"


class _H:
    def handle(self, request: object) -> object:  # pragma: no cover
        return request


def _route() -> DispatchRoute:
    return DispatchRoute(name="H", handler=_H(), published_events={})


def _contract(
    name: str, subscribe: tuple[str, ...], *, plugin_managed: bool = False
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=0, minor=1, patch=0),
        contract_path=Path(f"/nonexistent/{name}.yaml"),
        entry_point_name=name,
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=subscribe, plugin_managed=plugin_managed
        ),
    )


def test_passes_disjoint_single_owner() -> None:
    assert_single_owner_split(
        core_runtime_topics=frozenset({TOPIC}),
        routing_map={TOPIC: _route()},
        legacy_subscribed_topics=frozenset({"onex.cmd.omnibase-infra.other.v1"}),
        contracts=[_contract("owner", (TOPIC,))],
    )


def test_raises_when_legacy_still_owns_single_owner_topic() -> None:
    with pytest.raises(ModelOnexError, match="legacy"):
        assert_single_owner_split(
            core_runtime_topics=frozenset({TOPIC}),
            routing_map={TOPIC: _route()},
            legacy_subscribed_topics=frozenset({TOPIC}),  # overlap!
            contracts=[_contract("owner", (TOPIC,))],
        )


def test_raises_when_routing_map_incomplete() -> None:
    with pytest.raises(ModelOnexError, match="do not exactly match"):
        assert_single_owner_split(
            core_runtime_topics=frozenset({TOPIC}),
            routing_map={},  # missing route
            legacy_subscribed_topics=frozenset(),
            contracts=[_contract("owner", (TOPIC,))],
        )


def test_raises_when_plugin_managed_contract_is_allowlisted() -> None:
    with pytest.raises(ModelOnexError, match="plugin-managed contract"):
        assert_single_owner_split(
            core_runtime_topics=frozenset({TOPIC}),
            routing_map={TOPIC: _route()},
            legacy_subscribed_topics=frozenset(),
            contracts=[_contract("plugin_owner", (TOPIC,), plugin_managed=True)],
        )


def test_plugin_managed_owner_of_fanout_topic_fails_closed_omn14795() -> None:
    """OMN-14795 latent trap: a plugin_managed contract that is the SOLE designated
    fan-out owner of an allowlist topic STILL fails closed.

    This reproduces the exact delegation-orchestrator shape the S8 wave-3
    ``ONEX_CORE_RUNTIME_TOPICS`` move trips over: ``node_delegation_orchestrator`` is
    ``plugin_managed: true`` AND the designated core-runtime owner of the genuine
    fan-out ``inference-response.v1`` (also subscribed by
    ``node_projection_delegation_inference_response``). The ONLY thing separating this
    case from ``test_passes_fanout_with_owner_and_distinct_legacy_group`` (which passes)
    is ``plugin_managed=True`` on the owner.

    The plugin-managed guard MUST preempt the otherwise-valid fan-out-with-owner path:
    were the gate to silently admit a plugin_managed owner, its subscription would be
    wired onto NEITHER the core-runtime route (plugin-managed) NOR the legacy path
    (skipped for allowlist topics), leaving the topic with no owner and stranding the
    orchestrator consumer group Empty/Dead after boot (the OMN-14795 symptom). Locking
    this in prevents a future refactor from regressing the fail-closed into a silent
    no-op. Full resolution is an ownership-model decision (drop plugin_managed and let
    the core-runtime group own it, or fold the plugin subscription into that group) —
    this test only guards the invariant, not the chosen model.
    """
    with pytest.raises(ModelOnexError, match="plugin-managed contract"):
        assert_single_owner_split(
            core_runtime_topics=frozenset({FANOUT_TOPIC}),
            routing_map={FANOUT_TOPIC: _route()},
            legacy_subscribed_topics=frozenset({FANOUT_TOPIC}),
            contracts=[
                _contract(
                    "node_delegation_orchestrator",
                    (FANOUT_TOPIC,),
                    plugin_managed=True,
                ),
                _contract(
                    "node_projection_delegation_inference_response", (FANOUT_TOPIC,)
                ),
            ],
            # A VALID owner is designated — proving the plugin_managed guard wins anyway.
            owners={FANOUT_TOPIC: "node_delegation_orchestrator"},
        )


# --- S8 §D1=4b: fan-out enablement (was: outright refusal) ----------------------------


def test_raises_on_fanout_topic_without_owner_designation() -> None:
    """A multi-consumer allowlist topic with NO designated owner fails closed (§4b)."""
    with pytest.raises(ModelOnexError, match="no core-runtime owner is designated"):
        assert_single_owner_split(
            core_runtime_topics=frozenset({FANOUT_TOPIC}),
            routing_map={FANOUT_TOPIC: _route()},
            legacy_subscribed_topics=frozenset(),
            contracts=[
                _contract("owner", (FANOUT_TOPIC,)),
                _contract("second_consumer", (FANOUT_TOPIC,)),
            ],
            owners=None,
        )


def test_passes_fanout_with_owner_and_distinct_legacy_group() -> None:
    """§4b: exactly one owner moves; the non-owner keeps a DISTINCT legacy group."""
    # Default resolver derives groups from (env, package, node_name, version); the two
    # distinct contract names produce two distinct groups, and the fan-out topic is
    # exempt from the legacy-overlap check because its non-owner legitimately stays legacy.
    assert_single_owner_split(
        core_runtime_topics=frozenset({FANOUT_TOPIC}),
        routing_map={FANOUT_TOPIC: _route()},
        legacy_subscribed_topics=frozenset(
            {FANOUT_TOPIC}
        ),  # non-owner still legacy: OK
        contracts=[
            _contract("node_delegation_orchestrator", (FANOUT_TOPIC,)),
            _contract("node_projection_delegation_inference_response", (FANOUT_TOPIC,)),
        ],
        owners={FANOUT_TOPIC: "node_delegation_orchestrator"},
    )


def test_raises_when_designated_owner_not_a_subscriber() -> None:
    with pytest.raises(ModelOnexError, match="not among its subscribers"):
        assert_single_owner_split(
            core_runtime_topics=frozenset({FANOUT_TOPIC}),
            routing_map={FANOUT_TOPIC: _route()},
            legacy_subscribed_topics=frozenset(),
            contracts=[
                _contract("owner", (FANOUT_TOPIC,)),
                _contract("second_consumer", (FANOUT_TOPIC,)),
            ],
            owners={FANOUT_TOPIC: "not_a_subscriber"},
        )


def test_raises_when_two_legacy_consumers_share_a_group() -> None:
    """Group collision between two NON-owner legacy consumers fails closed (§4b)."""
    with pytest.raises(ModelOnexError, match="group collision"):
        assert_single_owner_split(
            core_runtime_topics=frozenset({FANOUT_TOPIC}),
            routing_map={FANOUT_TOPIC: _route()},
            legacy_subscribed_topics=frozenset({FANOUT_TOPIC}),
            contracts=[
                _contract("owner", (FANOUT_TOPIC,)),
                _contract("legacy_a", (FANOUT_TOPIC,)),
                _contract("legacy_b", (FANOUT_TOPIC,)),
            ],
            owners={FANOUT_TOPIC: "owner"},
            # Force both legacy consumers onto ONE group id — the collision the gate bans.
            consumer_group_resolver=lambda _c: "collided.group.v1",
        )


def test_raises_when_legacy_consumer_collides_with_core_group() -> None:
    """A non-owner legacy consumer resolving to the core-runtime group fails closed."""
    with pytest.raises(ModelOnexError, match="core-runtime consumer group"):
        assert_single_owner_split(
            core_runtime_topics=frozenset({FANOUT_TOPIC}),
            routing_map={FANOUT_TOPIC: _route()},
            legacy_subscribed_topics=frozenset({FANOUT_TOPIC}),
            contracts=[
                _contract("owner", (FANOUT_TOPIC,)),
                _contract("legacy_a", (FANOUT_TOPIC,)),
            ],
            owners={FANOUT_TOPIC: "owner"},
            consumer_group_resolver=lambda _c: CORE_RUNTIME_GROUP,
        )
