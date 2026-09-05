# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17034: the forwarder's lane-mirror leg (stability source -> dev mirror).

Operator ruling 2026-09-04, paraphrased: BOTH lanes, not either. The forwarder
consumes the hook topics on the lane the hook edge actually publishes to
(``stability-test``, declared by omniclaude ``hook_edge_lane.yaml``, OMN-17204)
and mirrors them to the ``dev`` lane, from which the pre-existing trust-boundary
outbound leg carries the OD-9-approved subset onward to the cloud. Nothing is
repointed: hooks stay on stability, the delegation/inference trust-boundary legs
stay on dev, and ``mirror_topics.outbound`` (the cloud set) is unchanged --
widening that set is OMN-16979, not this ticket.

These tests are the red-first proof for:
  * the CONTRACT declaring the source lane and the mirror-lane set;
  * the MODEL refusing a self-mirroring or non-canonical declaration;
  * the RUNTIME CONFIG requiring a resolved bus for every declared mirror lane;
  * the SERVICE delivering one source record to each mirror exactly once and
    staying idempotent across redelivery;
  * the LANE OVERLAY (compose) attaching the forwarder to both lane networks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTRACT_PATH = (
    _REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_bus_forwarder_effect"
    / "contract.yaml"
)
_GATEWAY_COMPOSE_PATH = _REPO_ROOT / "docker" / "docker-compose.gateway.yml"
_RESOLVED_CONFIG_PATH = _REPO_ROOT / "docker" / "gateway" / "beta-gateway-canary.yaml"

# The four topics omniclaude/plugins/onex/hooks/contracts/hook_edge_lane.yaml
# declares as `governed_topics`. Named here as literals because this repo cannot
# import the omniclaude registry; the cross-repo binding is asserted by that
# repo's own validate_hook_edge_lane.py gate.
_GOVERNED_HOOK_TOPICS = (
    "onex.evt.omniclaude.session-started.v1",
    "onex.evt.omniclaude.session-ended.v1",
    "onex.evt.omniclaude.tool-executed.v1",
    "onex.evt.omniclaude.prompt-submitted.v1",
)


def _lane_mirror_block() -> dict[str, Any]:
    contract = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))
    return contract["config"]["gateway_forwarder"]["lane_mirror"]


# ---------------------------------------------------------------------------
# 1. Contract
# ---------------------------------------------------------------------------


def test_contract_declares_stability_as_the_lane_mirror_source() -> None:
    """The source lane is the lane the hook edge publishes to, not dev."""
    assert _lane_mirror_block()["source_lane"] == "stability-test"


def test_contract_declares_dev_in_the_mirror_lane_set() -> None:
    """The operator ruling is BOTH lanes: dev is a declared mirror target."""
    assert "dev" in _lane_mirror_block()["mirror_lanes"]


def test_contract_lane_mirror_covers_every_governed_hook_topic() -> None:
    """All four hook_edge_lane.yaml governed topics cross stability -> dev."""
    declared = set(_lane_mirror_block()["topics"])
    assert set(_GOVERNED_HOOK_TOPICS) <= declared


def test_contract_cloud_mirror_set_is_not_widened_by_this_ticket() -> None:
    """OMN-16979 owns widening the cloud set; the LANE-MIRROR leg must not
    pre-empt it.

    OMN-16979 has since landed and widened it deliberately, so the assertion is
    kept in the form that still falsifies a lane-mirror change: the cloud set
    may contain a content-bearing hook class only while `egress_redaction`
    governs it. The lane mirror crosses no trust boundary and must never be the
    reason a class reaches the cloud.
    """
    contract = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))
    forwarder = contract["config"]["gateway_forwarder"]
    outbound = set(forwarder["mirror_topics"]["outbound"])
    governed = set(forwarder.get("egress_redaction", {}).get("governed_topics", ()))
    for topic in (
        "onex.evt.omniclaude.tool-executed.v1",
        "onex.evt.omniclaude.prompt-submitted.v1",
    ):
        if topic in outbound:
            assert topic in governed


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------


def test_lane_mirror_model_rejects_a_lane_mirroring_to_itself() -> None:
    """A lane in its own mirror set is an infinite republish loop."""
    from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
        ModelGatewayLaneMirrorConfig,
    )

    with pytest.raises(ValueError, match="source_lane"):
        ModelGatewayLaneMirrorConfig(
            source_lane="stability-test",
            mirror_lanes=("dev", "stability-test"),
            topics=_GOVERNED_HOOK_TOPICS,
        )


def test_lane_mirror_model_rejects_a_tenant_prefixed_topic() -> None:
    """Lane mirroring is local-to-local: canonical topics only, never wire topics."""
    from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
        ModelGatewayLaneMirrorConfig,
    )

    with pytest.raises(ValueError):
        ModelGatewayLaneMirrorConfig(
            source_lane="stability-test",
            mirror_lanes=("dev",),
            topics=("tenant-beta.onex.evt.omniclaude.tool-executed.v1",),
        )


def test_lane_mirror_model_requires_at_least_one_mirror_lane() -> None:
    from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
        ModelGatewayLaneMirrorConfig,
    )

    with pytest.raises(ValueError):
        ModelGatewayLaneMirrorConfig(
            source_lane="stability-test",
            mirror_lanes=(),
            topics=_GOVERNED_HOOK_TOPICS,
        )


# ---------------------------------------------------------------------------
# 3. Runtime config resolution
# ---------------------------------------------------------------------------


def test_runtime_config_requires_a_resolved_bus_for_every_declared_mirror_lane(
    lane_mirror_runtime_raw: dict[str, Any],
) -> None:
    """A declared mirror lane with no resolved broker fails closed, not silently."""
    from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
        ModelGatewayForwarderRuntimeConfig,
    )

    raw = lane_mirror_runtime_raw
    raw["lane_mirror_buses"] = {}
    with pytest.raises(ValueError, match="dev"):
        ModelGatewayForwarderRuntimeConfig.model_validate(raw)


def test_runtime_config_rejects_a_mirror_bus_equal_to_the_source_bus(
    lane_mirror_runtime_raw: dict[str, Any],
) -> None:
    """Source and mirror pointing at one broker republishes onto its own source."""
    from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
        ModelGatewayForwarderRuntimeConfig,
    )

    raw = lane_mirror_runtime_raw
    source = raw["lane_mirror_source_bus"]["bootstrap_servers"]
    raw["lane_mirror_buses"]["dev"]["bootstrap_servers"] = source
    with pytest.raises(ValueError, match="distinct"):
        ModelGatewayForwarderRuntimeConfig.model_validate(raw)


def test_resolved_deployment_yaml_names_the_contract_lane_mirror_set() -> None:
    """The resolved tenant YAML may not redeclare lane-mirror topic literals."""
    resolved = yaml.safe_load(_RESOLVED_CONFIG_PATH.read_text(encoding="utf-8"))
    forwarder = resolved["forwarder"]
    assert forwarder["lane_mirror_set"] == "node_bus_forwarder_effect"
    assert "lane_mirror" not in forwarder


# ---------------------------------------------------------------------------
# 4. Service behaviour -- exactly once per mirror, idempotent on redelivery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_source_record_reaches_each_mirror_exactly_once(
    lane_mirror_harness: Any,
) -> None:
    from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_lane_mirror import (
        NodeLaneMirror,
    )

    harness = lane_mirror_harness
    service = NodeLaneMirror(**harness.kwargs)
    harness.source.offer(harness.record(envelope_id="e-1"))

    await service.drain_once()

    for lane, producer in harness.mirrors.items():
        assert [m.topic for m in producer.sent] == [
            "onex.evt.omniclaude.tool-executed.v1"
        ], lane
        assert len(producer.sent) == 1, lane


@pytest.mark.asyncio
async def test_redelivery_of_the_same_envelope_publishes_nothing_further(
    lane_mirror_harness: Any,
) -> None:
    """At-least-once source redelivery must not duplicate on any mirror."""
    from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_lane_mirror import (
        NodeLaneMirror,
    )

    harness = lane_mirror_harness
    service = NodeLaneMirror(**harness.kwargs)
    harness.source.offer(harness.record(envelope_id="e-1"))
    await service.drain_once()
    harness.source.offer(harness.record(envelope_id="e-1"))
    await service.drain_once()

    for lane, producer in harness.mirrors.items():
        assert len(producer.sent) == 1, lane


@pytest.mark.asyncio
async def test_source_offset_commits_only_after_every_mirror_acknowledges(
    lane_mirror_harness: Any,
) -> None:
    """A failing mirror leaves the source uncommitted so the record is retried."""
    from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_lane_mirror import (
        NodeLaneMirror,
    )

    harness = lane_mirror_harness
    harness.mirrors["dev"].fail_next = True
    service = NodeLaneMirror(**harness.kwargs)
    harness.source.offer(harness.record(envelope_id="e-1"))

    await service.drain_once()

    assert harness.source.committed == []
    assert harness.source.nacked != []


@pytest.mark.asyncio
async def test_a_topic_outside_the_declared_lane_mirror_set_is_never_mirrored(
    lane_mirror_harness: Any,
) -> None:
    """Contract-declared topics only -- the same rule the cloud legs obey."""
    from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_lane_mirror import (
        NodeLaneMirror,
    )

    harness = lane_mirror_harness
    service = NodeLaneMirror(**harness.kwargs)
    harness.source.offer(
        harness.record(
            envelope_id="e-2", topic="onex.evt.omnibase-infra.gateway-heartbeat.v1"
        )
    )

    await service.drain_once()

    for lane, producer in harness.mirrors.items():
        assert producer.sent == [], lane


# ---------------------------------------------------------------------------
# 5. Lane overlay -- the container must see both lane networks
# ---------------------------------------------------------------------------


def _gateway_compose() -> dict[str, Any]:
    return yaml.safe_load(_GATEWAY_COMPOSE_PATH.read_text(encoding="utf-8"))


def test_gateway_compose_declares_both_lane_networks() -> None:
    """A container attached only to dev can never see a stability-lane record."""
    networks = _gateway_compose()["networks"]
    external_names = {
        spec.get("name") for spec in networks.values() if isinstance(spec, dict)
    }
    assert "omnibase-infra-network" in external_names
    assert "omnibase-infra-stability-test-network" in external_names


def test_gateway_forwarder_service_joins_both_lane_networks() -> None:
    names = set(_gateway_compose()["services"]["gateway-forwarder"]["networks"])
    assert "gateway-runtime" in names
    assert "gateway-lane-mirror-source" in names


def test_lane_brokers_are_addressed_by_unique_container_name_not_bare_redpanda() -> (
    None
):
    """`redpanda` resolves on BOTH lane networks -- a bare alias is ambiguous."""
    resolved = yaml.safe_load(_RESOLVED_CONFIG_PATH.read_text(encoding="utf-8"))
    source = resolved["lane_mirror_source_bus"]["bootstrap_servers"]
    mirrors = resolved["lane_mirror_buses"]
    assert source.startswith("omnibase-infra-stability-test-redpanda:")
    assert mirrors["dev"]["bootstrap_servers"].startswith("omnibase-infra-redpanda:")


def test_dns_bastion_joins_every_lane_network_the_forwarder_resolves_on() -> None:
    """The forwarder sends ALL DNS to the bastion, so the bastion needs the lane.

    dnsmasq forwards non-overridden queries to ``server=127.0.0.11`` -- Docker's
    embedded resolver in the BASTION's namespace, which only answers for
    networks the bastion itself is attached to. Measured on .201 2026-09-04
    before this fix: the bastion resolved ``omnibase-infra-redpanda`` to
    172.19.0.7 and returned nothing for
    ``omnibase-infra-stability-test-redpanda``.
    """
    compose = _gateway_compose()
    forwarder = compose["services"]["gateway-forwarder"]
    # Only meaningful while the forwarder routes DNS through the sidecar.
    assert forwarder.get("dns"), "forwarder no longer overrides DNS; revisit this test"
    bastion_networks = set(compose["services"]["gateway-dns-bastion"]["networks"])
    forwarder_networks = set(forwarder["networks"])
    assert forwarder_networks <= bastion_networks, (
        "gateway-forwarder resolves names on "
        f"{sorted(forwarder_networks - bastion_networks)} that the DNS bastion "
        "cannot see"
    )


def test_a_round_tripped_null_lane_mirror_is_not_treated_as_a_redeclaration(
    tmp_path: Path,
) -> None:
    """``lane_mirror: null`` is what ``model_dump()`` emits, not a redeclaration.

    Regression: the first revision of ``_materialize_contract_lane_mirror``
    rejected on KEY PRESENCE, so any config round-tripped through
    ``ModelGatewayForwarderConfig.model_dump()`` -- which always emits an
    explicit null for this optional field -- was refused at load. It broke
    three pre-existing loader tests in tests/unit/runtime.
    """
    from omnibase_infra.runtime.gateway_forwarder import (
        _materialize_contract_lane_mirror,
    )

    raw: dict[str, Any] = {"forwarder": {"lane_mirror": None}}
    _materialize_contract_lane_mirror(raw, _CONTRACT_PATH)
    assert raw["forwarder"]["lane_mirror"] is None


def test_a_populated_inline_lane_mirror_block_is_still_refused(
    tmp_path: Path,
) -> None:
    """The contract stays the sole authority for the lane-mirror declaration."""
    from omnibase_infra.runtime.gateway_forwarder import (
        _materialize_contract_lane_mirror,
    )

    raw: dict[str, Any] = {
        "forwarder": {
            "lane_mirror": {"source_lane": "dev", "mirror_lanes": ["prod"]},
            "lane_mirror_set": "node_bus_forwarder_effect",
        }
    }
    with pytest.raises(ValueError, match="lane_mirror_set"):
        _materialize_contract_lane_mirror(raw, _CONTRACT_PATH)
