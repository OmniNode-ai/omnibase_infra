# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18691 phase 3: the forwarder can be configured as a LANE MIRROR ONLY.

Why this exists
---------------
The fleet's CI publishers used to write to the dev lane's Redpanda, so every
routine dev-lane rebuild reddened Receipt Gates on unrelated pull requests.
Phase 2 stood up a dedicated CI-bus broker (compose project ``omninode-ci-bus``)
to hold the publish leg. That only half-closes the outage class: the four
consumers of the CI topics are hosted by ``omninode-runtime-effects``, a service
of the DEV LANE project, and the runtime kernel takes one bootstrap address for
its whole lifetime. A publisher repointed before a consumer attaches would be
green and silent, which is strictly worse than today's red publish.

The option of record (OMN-18691 comment 08f132a0, option B) is to mirror the CI
topics from the dedicated broker onto the lane broker with the existing
``node_bus_forwarder_effect`` lane-mirror leg, leaving all four consumers exactly
where they are. That option was costed as "a configuration change to a primitive
that already runs". **It is not one today**, and that is what these tests pin:

``ModelGatewayForwarderConfig`` requires ``cloud_bus``, and
``ModelGatewayForwarderRuntimeConfig`` requires a resolved ``cloud_bus`` /
``local_bus`` pair and validates them unconditionally. ``cloud_bus`` is an
AWS-MSK-shaped declaration (``security_protocol`` is pinned to ``SASL_SSL``,
``sasl_mechanism`` to ``OAUTHBEARER`` or ``AWS_MSK_IAM``) whose deployment in
``docker-compose.gateway.yml`` needs a TPM device, an IAM Roles Anywhere
certificate and the AWS signing helper. Reusing the primitive as a pure
intra-host mirror would therefore mean giving the CI bus a cloud egress path
nobody asked for, or writing a cloud config that is never dialled.

So this change makes the two trust-boundary legs OPTIONAL and makes a
mirror-only deployment expressible. Nothing about a cloud deployment changes:
the legs travel together, all-present or all-absent, and every existing
validation still fires when they are present.

What is deliberately NOT here
-----------------------------
The compose service and the repoint of the publishers. Both are phase-3 steps 3
and 4 in ``runbooks/ci-bus-dedicated-broker.md``, both need the host and an
operator consent row, and neither merges on this branch.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCanaryConfig,
    ModelGatewayCloudBusConfig,
    ModelGatewayForwarderConfig,
    ModelGatewayLaneMirrorConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_forwarder_runtime_config import (
    ModelGatewayForwarderRuntimeConfig,
)

pytestmark = pytest.mark.unit

# The two endpoints the phase-3 topology is between. Spelled as the container
# names the mirror resolves on the host, never as a tailnet address: the
# committed lane->broker map in omnimarket `config/ci_bus_lanes.yaml` is the
# authority for the external addresses, and duplicating one here would be a
# second place to drift.
_CI_BUS_SOURCE = "omninode-ci-bus-redpanda:9092"
_DEV_LANE_MIRROR = "omnibase-infra-redpanda:9092"

_CI_BUS_TOPICS = (
    "onex.cmd.omnimarket.occ-autobind.v1",
    "onex.cmd.omnimarket.occ-companion-effect-requested.v1",
    "onex.evt.github.pr-merged.v1",
    "onex.cmd.omnimarket.redeploy-start.v1",
)


def _tenant_identity() -> ModelGatewayTenantIdentity:
    return ModelGatewayTenantIdentity(
        tenant_id=UUID("79afa726-3852-464f-b7a4-d4b8b9c75ee7"),
        tenant_slug="beta-gateway-canary-79afa7263852",
        principal_id="t-79afa7263852464fb7a4d4b8b9c75ee7",
    )


def _cloud_bus_declaration() -> ModelGatewayCloudBusConfig:
    return ModelGatewayCloudBusConfig(
        broker_provider_id=UUID("22222222-2222-2222-2222-222222222222"),
        cloud_broker_ref="gateway.cloud.kafka.broker",
        cloud_auth_ref="gateway.cloud.kafka.msk_iam",
        acl_provisioner_ref="gateway.cloud.kafka.authorization",
        msk_region_ref="gateway.cloud.kafka.msk_region",
        sasl_mechanism="AWS_MSK_IAM",
    )


def _mirror_topics() -> ModelGatewayMirrorTopics:
    return ModelGatewayMirrorTopics(
        inbound=("onex.cmd.omnibase-infra.delegation-request.v1",),
        outbound=(
            "onex.evt.omnibase-infra.delegation-completed.v1",
            "onex.evt.omnibase-infra.gateway-heartbeat.v1",
        ),
    )


def _canary() -> ModelGatewayCanaryConfig:
    return ModelGatewayCanaryConfig(
        topic="onex.evt.omnibase-infra.gateway-canary.v1",
        cadence_seconds=30,
        produce_deadline_seconds=8,
        readback_deadline_seconds=12,
    )


def _ci_bus_lane_mirror() -> ModelGatewayLaneMirrorConfig:
    return ModelGatewayLaneMirrorConfig(
        source_lane="ci-bus",
        mirror_lanes=("dev",),
        topics=_CI_BUS_TOPICS,
    )


def _mirror_only_forwarder() -> ModelGatewayForwarderConfig:
    """The phase-3 shape: a lane mirror and no trust-boundary legs at all."""
    return ModelGatewayForwarderConfig(
        tenant_identity=_tenant_identity(),
        local_transport_flavor="containerized",
        dedupe_store_path=Path("/app/data/ci-bus-mirror.sqlite3"),
        lane_mirror=_ci_bus_lane_mirror(),
    )


def _cloud_forwarder(**overrides: object) -> ModelGatewayForwarderConfig:
    """Today's shape, unchanged, as the regression control."""
    fields: dict[str, object] = {
        "tenant_identity": _tenant_identity(),
        "cloud_bus": _cloud_bus_declaration(),
        "local_transport_flavor": "containerized",
        "dedupe_store_path": Path("/app/data/gateway.sqlite3"),
        "mirror_topics": _mirror_topics(),
        "canary": _canary(),
    }
    fields.update(overrides)
    return ModelGatewayForwarderConfig(**fields)  # type: ignore[arg-type]


def _source_leg(bootstrap: str = _CI_BUS_SOURCE) -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers=bootstrap,
        environment="ci-bus-mirror-source",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )


def _mirror_leg(bootstrap: str = _DEV_LANE_MIRROR) -> ModelKafkaEventBusConfig:
    return ModelKafkaEventBusConfig(
        bootstrap_servers=bootstrap,
        environment="ci-bus-mirror-destination",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )


# ---------------------------------------------------------------------------
# 1. The declared config can express a mirror-only deployment
# ---------------------------------------------------------------------------


def test_declared_config_accepts_a_mirror_only_deployment() -> None:
    """A forwarder that only mirrors declares no cloud bus, topics or canary.

    This is the whole premise of option B. Today it raises, because `cloud_bus`,
    `mirror_topics` and `canary` are all required fields.
    """
    config = _mirror_only_forwarder()
    assert config.cloud_bus is None
    assert config.mirror_topics is None
    assert config.canary is None
    assert config.lane_mirror is not None
    assert config.lane_mirror.source_lane == "ci-bus"


def test_declared_config_refuses_a_forwarder_with_no_leg_at_all() -> None:
    """No cloud bus and no lane mirror is a process that forwards nothing.

    A forwarder with nothing to forward starts, reports healthy and moves zero
    records -- the green-and-silent class this whole ticket exists to remove.
    """
    with pytest.raises(ValueError, match="at least one"):
        ModelGatewayForwarderConfig(
            tenant_identity=_tenant_identity(),
            local_transport_flavor="containerized",
            dedupe_store_path=Path("/app/data/nothing.sqlite3"),
        )


@pytest.mark.parametrize(
    "partial",
    [
        pytest.param({"mirror_topics": _mirror_topics()}, id="topics-without-cloud"),
        pytest.param({"canary": _canary()}, id="canary-without-cloud"),
    ],
)
def test_declared_config_refuses_a_cloud_field_without_a_cloud_bus(
    partial: dict[str, object],
) -> None:
    """The trust-boundary fields travel together, all-present or all-absent.

    `mirror_topics` is the cloud egress set and `canary` is the cloud
    round-trip probe. Either one declared with no cloud bus is policy that
    reads live and is dead -- the same failure shape as a lane_mirror whose
    broker legs were never resolved.
    """
    with pytest.raises(ValueError, match="cloud_bus"):
        ModelGatewayForwarderConfig(
            tenant_identity=_tenant_identity(),
            local_transport_flavor="containerized",
            dedupe_store_path=Path("/app/data/partial.sqlite3"),
            lane_mirror=_ci_bus_lane_mirror(),
            **partial,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "missing",
    [
        pytest.param("mirror_topics", id="cloud-without-topics"),
        pytest.param("canary", id="cloud-without-canary"),
    ],
)
def test_declared_config_still_requires_the_cloud_fields_with_a_cloud_bus(
    missing: str,
) -> None:
    """Regression control: a cloud deployment is byte-unchanged in its demands."""
    fields: dict[str, object] = {
        "tenant_identity": _tenant_identity(),
        "cloud_bus": _cloud_bus_declaration(),
        "local_transport_flavor": "containerized",
        "dedupe_store_path": Path("/app/data/gateway.sqlite3"),
        "mirror_topics": _mirror_topics(),
        "canary": _canary(),
    }
    del fields[missing]
    with pytest.raises(ValueError, match=missing):
        ModelGatewayForwarderConfig(**fields)  # type: ignore[arg-type]


def test_declared_cloud_config_is_unchanged() -> None:
    """The existing shape still validates exactly as it does on dev."""
    config = _cloud_forwarder()
    assert config.cloud_bus is not None
    assert config.mirror_topics is not None
    assert config.canary is not None
    assert config.lane_mirror is None


# ---------------------------------------------------------------------------
# 2. The resolved runtime config can express a mirror-only process
# ---------------------------------------------------------------------------


def test_runtime_config_accepts_a_mirror_only_process() -> None:
    """The CI-bus topology: consume `ci-bus`, republish onto `dev`, no cloud."""
    runtime = ModelGatewayForwarderRuntimeConfig(
        forwarder=_mirror_only_forwarder(),
        lane_mirror_source_bus=_source_leg(),
        lane_mirror_buses={"dev": _mirror_leg()},
    )
    assert runtime.cloud_bus is None
    assert runtime.local_bus is None
    assert runtime.lane_mirror_source_bus is not None
    assert runtime.lane_mirror_source_bus.bootstrap_servers == _CI_BUS_SOURCE


def test_mirror_only_process_reads_the_dedicated_broker_and_writes_the_lane() -> None:
    """Direction is load-bearing and is pinned here, not left to a comment.

    The source is the DEDICATED broker and the destination is the DEV LANE.
    The reverse -- consuming the lane and republishing onto the CI bus -- would
    leave every consumer still reading a broker the deploy agent recreates, so
    it would look like a cutover and close nothing.
    """
    runtime = ModelGatewayForwarderRuntimeConfig(
        forwarder=_mirror_only_forwarder(),
        lane_mirror_source_bus=_source_leg(),
        lane_mirror_buses={"dev": _mirror_leg()},
    )
    source = runtime.lane_mirror_source_bus
    assert source is not None
    assert source.bootstrap_servers == _CI_BUS_SOURCE
    assert source.bootstrap_servers != _DEV_LANE_MIRROR
    assert runtime.lane_mirror_buses["dev"].bootstrap_servers == _DEV_LANE_MIRROR
    # The runbook calls this load-bearing: the source offsets live on the
    # CI-bus broker and are committed only after the mirror acknowledges, so a
    # dev-lane rebuild cannot lose the mirror's place.
    assert source.enable_auto_commit is False
    assert source.auto_offset_reset == "earliest"


@pytest.mark.parametrize(
    "resolved",
    [
        pytest.param(
            {"cloud_bus": _mirror_leg("b-1.example.kafka.amazonaws.com:9098")},
            id="cloud",
        ),
        pytest.param({"local_bus": _mirror_leg("redpanda:9092")}, id="local"),
    ],
)
def test_runtime_config_refuses_a_resolved_leg_the_contract_does_not_declare(
    resolved: dict[str, object],
) -> None:
    """Fail closed exactly as the lane-mirror legs already do.

    A resolved trust-boundary leg on a contract that declares no cloud bus is
    a deployment asserting a capability the contract does not carry. The
    contract is the authority, so this is a boot refusal.
    """
    with pytest.raises(ValueError, match="declares no cloud_bus"):
        ModelGatewayForwarderRuntimeConfig(
            forwarder=_mirror_only_forwarder(),
            lane_mirror_source_bus=_source_leg(),
            lane_mirror_buses={"dev": _mirror_leg()},
            **resolved,  # type: ignore[arg-type]
        )


def test_runtime_config_requires_both_trust_boundary_legs_when_cloud_is_declared() -> (
    None
):
    """Regression control: a declared cloud bus still needs both resolved legs."""
    with pytest.raises(ValueError, match="local_bus"):
        ModelGatewayForwarderRuntimeConfig(
            forwarder=_cloud_forwarder(),
            cloud_bus=ModelKafkaEventBusConfig(
                bootstrap_servers="b-1.example.kafka.amazonaws.com:9098",
                environment="gateway-cloud",
                security_protocol="SASL_SSL",
                sasl_mechanism="AWS_MSK_IAM",
                msk_region="us-east-1",
                enable_auto_commit=False,
                auto_offset_reset="earliest",
            ),
        )


def test_mirror_only_process_is_not_held_to_the_cloud_heartbeat_requirement() -> None:
    """The outbound heartbeat proves the CLOUD leg is alive.

    A mirror-only process has no cloud leg, so demanding one would be a
    requirement nothing can satisfy -- and satisfying it with a heartbeat topic
    that is never published is worse, because it reads as a live liveness
    signal. The mirror's own delivery counters are its liveness signal.
    """
    runtime = ModelGatewayForwarderRuntimeConfig(
        forwarder=_mirror_only_forwarder(),
        lane_mirror_source_bus=_source_leg(),
        lane_mirror_buses={"dev": _mirror_leg()},
    )
    assert runtime.forwarder.mirror_topics is None


def test_mirror_only_process_still_refuses_a_broker_mirrored_onto_itself() -> None:
    """Every existing lane-mirror refusal survives the mirror-only mode."""
    with pytest.raises(ValueError, match="must be distinct"):
        ModelGatewayForwarderRuntimeConfig(
            forwarder=_mirror_only_forwarder(),
            lane_mirror_source_bus=_source_leg(),
            lane_mirror_buses={"dev": _mirror_leg(_CI_BUS_SOURCE)},
        )


def test_mirror_only_process_still_refuses_an_unresolved_mirror_lane() -> None:
    """A declared mirror lane with no resolved broker is still a boot refusal."""
    with pytest.raises(ValueError, match="unresolved"):
        ModelGatewayForwarderRuntimeConfig(
            forwarder=_mirror_only_forwarder(),
            lane_mirror_source_bus=_source_leg(),
            lane_mirror_buses={},
        )
