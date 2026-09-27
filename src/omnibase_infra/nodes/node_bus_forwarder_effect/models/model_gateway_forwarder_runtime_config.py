# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolved process configuration for the gateway bus forwarder."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_forwarder_config import (
    ModelGatewayForwarderConfig,
)


class ModelGatewayForwarderRuntimeConfig(BaseModel):
    """Contract plus the two resolved broker legs used by one edge process.

    ``forwarder.cloud_bus`` remains the provider-neutral declaration of the
    required capabilities and secret references.  ``cloud_bus`` and
    ``local_bus`` are the resolved, typed materialization consumed by
    ``KafkaTransport``.  Keeping the layers separate prevents a process-wide
    ``KAFKA_*`` environment from silently making both legs point at one broker.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    forwarder: ModelGatewayForwarderConfig
    # OMN-18691: both trust-boundary legs are optional and are required exactly
    # when the contract declares a `cloud_bus`. A LANE MIRROR ONLY deployment
    # (the CI bus mirroring its topics onto the dev lane) has no trust boundary
    # to resolve legs for, and a resolved leg the contract does not declare is a
    # deployment asserting a capability the contract does not carry -- refused
    # below on the same terms the lane-mirror legs already are.
    local_bus: ModelKafkaEventBusConfig | None = None
    cloud_bus: ModelKafkaEventBusConfig | None = None
    # OMN-17034: resolved broker legs for the contract-declared lane mirror.
    # The contract NAMES lanes; this is where this deployment says which
    # broker each named lane is. Both stay None when the contract declares no
    # lane_mirror.
    lane_mirror_source_bus: ModelKafkaEventBusConfig | None = None
    lane_mirror_buses: dict[str, ModelKafkaEventBusConfig] = {}

    @model_validator(mode="after")
    def _validate_resolved_legs(self) -> ModelGatewayForwarderRuntimeConfig:
        if self.forwarder.local_transport_flavor != "containerized":
            raise ValueError(
                "the production gateway process currently requires the "
                "containerized local transport flavor"
            )
        self._validate_trust_boundary_presence()
        self._validate_lane_mirror_legs()
        if self.forwarder.cloud_bus is None:
            # Mirror-only: every check below is about the trust boundary, and
            # this process has none. The lane-mirror leg validated above is the
            # whole of its configuration.
            return self

        # Narrowed by _validate_trust_boundary_presence above; restated for the
        # type checker rather than assumed.
        assert self.local_bus is not None
        assert self.cloud_bus is not None

        if self.local_bus.bootstrap_servers == self.cloud_bus.bootstrap_servers:
            raise ValueError("gateway local_bus and cloud_bus must be distinct")
        if self.local_bus.enable_auto_commit or self.cloud_bus.enable_auto_commit:
            raise ValueError(
                "gateway transport legs require enable_auto_commit=false; source "
                "offsets are committed only after durable destination delivery"
            )
        if (
            self.local_bus.auto_offset_reset != "earliest"
            or self.cloud_bus.auto_offset_reset != "earliest"
        ):
            raise ValueError(
                "gateway transport legs require auto_offset_reset=earliest on "
                "both legs; enable_auto_commit=false only preserves offsets "
                "already inside the consumer's read window -- a 'latest' leg "
                "silently drops any backlog produced while the consumer group "
                "was unjoined (crash, LeaveGroup, cold restart before rejoin), "
                "and auto_offset_reset also fires mid-session on "
                "OffsetOutOfRangeError, not only on first boot (OMN-15781)"
            )
        mirror_topics = self.forwarder.mirror_topics
        assert mirror_topics is not None
        if not any(
            topic.endswith(".gateway-heartbeat.v1") for topic in mirror_topics.outbound
        ):
            raise ValueError(
                "production gateway forwarder requires an outbound heartbeat topic"
            )

        https_ingest = self.forwarder.https_ingest
        if https_ingest is not None:
            # OMN-16459: the HTTPS door is not the broker. If the operator wired
            # the ingest ref at the broker ref, the leg would "work" by dialing
            # the very endpoint this ticket exists to stop dialing.
            cloud_host = self.cloud_bus.bootstrap_servers.split(",")[0].split(":")[0]
            if https_ingest.ingest_host == cloud_host:
                raise ValueError(
                    "gateway https_ingest.ingest_url must not resolve to the cloud "
                    "broker host; the HTTPS ingest door is a gateway route, not a "
                    "broker endpoint (OMN-16459 / ruling 39 OMN-15692)"
                )

        declared_cloud = self.forwarder.cloud_bus
        if self.cloud_bus.security_protocol != declared_cloud.security_protocol:
            raise ValueError(
                "resolved cloud security_protocol does not match the gateway contract"
            )
        if self.cloud_bus.sasl_mechanism != declared_cloud.sasl_mechanism:
            raise ValueError(
                "resolved cloud sasl_mechanism does not match the gateway contract"
            )
        if (
            self.cloud_bus.sasl_mechanism == "AWS_MSK_IAM"
            and not self.cloud_bus.msk_region
        ):
            raise ValueError("resolved AWS_MSK_IAM cloud bus requires msk_region")
        return self

    def _validate_trust_boundary_presence(self) -> None:
        """OMN-18691: the resolved legs and the contract must agree, both ways.

        A resolved ``local_bus`` or ``cloud_bus`` on a contract that declares no
        ``cloud_bus`` is a deployment claiming a trust boundary the node does not
        have -- the same direction of error ``_validate_lane_mirror_legs``
        already refuses for a resolved mirror lane the contract never named. A
        declared ``cloud_bus`` with a leg unresolved is the reverse: a forwarder
        that would boot believing it can reach a broker it has no address for.

        Both are boot refusals, because both produce a process that reports
        healthy and moves nothing across the boundary it claims to serve.
        """
        resolved = {
            "local_bus": self.local_bus,
            "cloud_bus": self.cloud_bus,
        }
        present = sorted(name for name, bus in resolved.items() if bus is not None)
        if self.forwarder.cloud_bus is None:
            if present:
                raise ValueError(
                    f"resolved {present} but the node contract declares no "
                    "cloud_bus; the contract is the authority for whether this "
                    "forwarder crosses a trust boundary at all"
                )
            return
        missing = sorted(name for name, bus in resolved.items() if bus is None)
        if missing:
            raise ValueError(
                "the node contract declares a cloud_bus, so both trust-boundary "
                f"legs must be resolved; unresolved: {missing}"
            )

    def _validate_lane_mirror_legs(self) -> None:
        """Fail closed on a declared lane mirror whose brokers were not resolved.

        The failure this prevents is the one OMN-17034 exists to close: a
        forwarder that *declares* it mirrors a lane it cannot actually reach
        looks healthy and moves nothing. Silence is the defect, so an
        unresolved leg is a boot refusal, never a skipped mirror.
        """
        lane_mirror = self.forwarder.lane_mirror
        if lane_mirror is None:
            if self.lane_mirror_source_bus is not None or self.lane_mirror_buses:
                raise ValueError(
                    "lane_mirror_source_bus/lane_mirror_buses are resolved but the "
                    "node contract declares no lane_mirror; the contract is the "
                    "authority for which lanes are mirrored"
                )
            return

        if self.lane_mirror_source_bus is None:
            raise ValueError(
                "the node contract declares a lane_mirror with source_lane "
                f"{lane_mirror.source_lane!r} but no lane_mirror_source_bus was "
                "resolved for it"
            )

        missing = [
            lane
            for lane in lane_mirror.mirror_lanes
            if lane not in self.lane_mirror_buses
        ]
        if missing:
            raise ValueError(
                "every contract-declared mirror lane requires a resolved broker "
                f"leg; unresolved: {missing}"
            )
        undeclared = [
            lane
            for lane in self.lane_mirror_buses
            if lane not in lane_mirror.mirror_lanes
        ]
        if undeclared:
            raise ValueError(
                "resolved lane_mirror_buses carry lanes the node contract does not "
                f"declare: {undeclared}"
            )

        source_endpoint = self.lane_mirror_source_bus.bootstrap_servers
        for lane, bus in self.lane_mirror_buses.items():
            if bus.bootstrap_servers == source_endpoint:
                raise ValueError(
                    f"lane_mirror source and mirror lane {lane!r} must be distinct "
                    "brokers; mirroring a broker onto itself republishes every "
                    "record onto the topic it was just consumed from"
                )
        if self.lane_mirror_source_bus.enable_auto_commit:
            raise ValueError(
                "the lane_mirror source leg requires enable_auto_commit=false; "
                "the source offset commits only after every mirror lane has "
                "acknowledged"
            )
        if self.lane_mirror_source_bus.auto_offset_reset != "earliest":
            raise ValueError(
                "the lane_mirror source leg requires auto_offset_reset=earliest "
                "for the same reason both trust-boundary legs do (OMN-15781)"
            )


__all__ = ["ModelGatewayForwarderRuntimeConfig"]
