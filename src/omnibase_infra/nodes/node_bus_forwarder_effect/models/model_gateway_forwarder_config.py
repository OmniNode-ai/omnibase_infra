# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed config for the tenant gateway bus forwarder."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_canary_config import (
    ModelGatewayCanaryConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_cloud_bus_config import (
    ModelGatewayCloudBusConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_egress_metadata_scrub import (
    ModelGatewayEgressMetadataScrub,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_egress_redaction import (
    ModelGatewayEgressRedaction,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_https_ingest_config import (
    ModelGatewayHttpsIngestConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_lane_mirror_config import (
    ModelGatewayLaneMirrorConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_mirror_topics import (
    ModelGatewayMirrorTopics,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_tenant_identity import (
    ModelGatewayTenantIdentity,
)

# OMN-16979. Which capture topics require redaction proof is a RULE, not a
# topic registry -- deliberately.
#
# A list would have to be edited whenever a new hook class appears, and the
# edit that forgets it is exactly the one that leaks. The rule instead presumes
# EVERY omniclaude event class carries enough session or user context to need
# the upstream redaction provenance at this trust boundary. The metadata-only
# tool-output capture has the omnimarket producer segment, so its one event
# grammar segment is included too. Full topic literals remain in the contract.
#
# So a new omniclaude class added to mirror_topics.outbound without governance
# fails config validation, with no action required by whoever adds it. That is
# the fail-closed direction.
#
# Parsed by exact segment over the canonical topic grammar
# (``onex.<kind>.<producer>.<event-name>.<version>``) rather than by matching
# a topic-shaped prefix string. That keeps this a predicate over the grammar:
# no literal topic lives here, and malformed extra segments do not silently
# collapse into a different event name.
_ONEX_NAMESPACE = "onex"
_EVENT_KIND = "evt"
_OMNICLAUDE_PRODUCER = "omniclaude"
_HOOK_CAPTURE_PRODUCER = "omnimarket"
_TOOL_OUTPUT_CAPTURE_EVENT = "tool-output-captured"


def requires_egress_redaction(canonical_topic: str) -> bool:
    """Whether ``canonical_topic`` must carry upstream redaction provenance."""
    segments = canonical_topic.split(".")
    if len(segments) != 5:
        return False
    if (segments[0], segments[1]) != (_ONEX_NAMESPACE, _EVENT_KIND):
        return False
    version = segments[4]
    if not version.startswith("v") or not version[1:].isdigit():
        return False
    event_name = segments[3]
    producer = segments[2]
    return producer == _OMNICLAUDE_PRODUCER or (
        producer == _HOOK_CAPTURE_PRODUCER and event_name == _TOOL_OUTPUT_CAPTURE_EVENT
    )


# OMN-19439. Same fail-closed shape as the rule above, for the delegate-skill
# terminals: every ``onex.evt.omnimarket.delegate-skill-*`` class carries the
# delegated prompt and the model's answer, so any of them mirrored outbound
# must be reduced to metadata by ``egress_metadata_scrub``. A new terminal class
# added to the outbound set without the scrub fails config validation.
_DELEGATE_SKILL_PRODUCER = "omnimarket"
_DELEGATE_SKILL_EVENT_PREFIX = "delegate-skill-"


def requires_metadata_scrub(canonical_topic: str) -> bool:
    """Whether ``canonical_topic`` may cross only as scrubbed metadata."""
    segments = canonical_topic.split(".")
    if len(segments) != 5:
        return False
    if (segments[0], segments[1]) != (_ONEX_NAMESPACE, _EVENT_KIND):
        return False
    version = segments[4]
    if not version.startswith("v") or not version[1:].isdigit():
        return False
    return segments[2] == _DELEGATE_SKILL_PRODUCER and segments[3].startswith(
        _DELEGATE_SKILL_EVENT_PREFIX
    )


class ModelGatewayForwarderConfig(BaseModel):
    """Complete forwarder config for one attached tenant edge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_identity: ModelGatewayTenantIdentity
    # OMN-18691. The three CLOUD fields are optional and travel TOGETHER --
    # all present (a tenant-edge deployment, byte-unchanged from before this
    # ticket) or all absent (a LANE MIRROR ONLY deployment). `cloud_bus` is the
    # trust-boundary declaration, `mirror_topics` is the egress set that crosses
    # it, and `canary` is the round-trip probe over it; none of the three means
    # anything without the other two, and the validator below refuses every
    # partial combination rather than letting one read as live policy.
    #
    # WHY THEY BECAME OPTIONAL. The fleet's CI bus moved to a dedicated broker
    # (compose project `omninode-ci-bus`) so a dev-lane rebuild stops reddening
    # unrelated Receipt Gates. The four consumers of the CI topics are hosted by
    # the DEV LANE's effects runtime and the runtime kernel takes one bootstrap
    # address for its whole lifetime, so closing the outage class needs those
    # topics mirrored from the dedicated broker onto the lane broker. This node
    # already does exactly that, in its lane_mirror leg. What it could not do
    # was run WITHOUT a cloud leg: `cloud_bus` is AWS-MSK-shaped
    # (`security_protocol` is pinned to SASL_SSL) and its deployment needs a TPM
    # device, an IAM Roles Anywhere certificate and the AWS signing helper. A
    # mirror that had to carry all of that would be giving an intra-host CI bus
    # a cloud egress path nobody asked for; a mirror that declared a cloud bus
    # it never dialled would be a config that lies.
    cloud_bus: ModelGatewayCloudBusConfig | None = None
    local_transport_flavor: Literal["containerized", "lightweight"]
    mirror_topics: ModelGatewayMirrorTopics | None = None
    canary: ModelGatewayCanaryConfig | None = None
    # OMN-17034. Optional so a deployment that predates the lane-mirror leg
    # (or one where there is only one lane to begin with) keeps booting with
    # the two trust-boundary legs alone; the runtime config below is what
    # refuses a declaration whose broker legs were not resolved.
    lane_mirror: ModelGatewayLaneMirrorConfig | None = None
    # OMN-16459: opt-in HTTPS ingest leg for the OUTBOUND publish boundary.
    # ``None`` (the default) keeps the direct-MSK Kafka outbound leg, so every
    # deployment that has not opted in is byte-unchanged. The INBOUND leg is
    # a Kafka pull from the cloud broker either way -- see
    # ModelGatewayHttpsIngestConfig's module docstring for why that means this
    # block alone does not retire the OMN-16449 bastion.
    https_ingest: ModelGatewayHttpsIngestConfig | None = None
    # OMN-16979: fail-closed admission gate for the capture topics this ticket
    # adds to ``mirror_topics.outbound``. Optional so every
    # deployment predating the widening keeps its exact behaviour; the
    # cross-field validator below is what refuses an inconsistent pairing.
    egress_redaction: ModelGatewayEgressRedaction | None = None
    # OMN-19439: metadata-only scrub for the delegate-skill terminals. Optional
    # so a deployment that mirrors none of them is unchanged; the cross-field
    # validator refuses a delegate-skill topic mirrored without it.
    egress_metadata_scrub: ModelGatewayEgressMetadataScrub | None = None
    heartbeat_interval_seconds: int = Field(default=15, ge=1)
    max_silence_window_seconds: int = Field(default=60, ge=1)
    lag_threshold_messages: int = Field(default=500, ge=1)
    lag_threshold_seconds: int = Field(default=120, ge=1)
    drain_deadline_seconds: int = Field(default=30, ge=1)
    dedupe_store_path: Path
    dedupe_retention_hours: int = Field(default=24, ge=24)
    forward_retry_initial_seconds: float = Field(default=1.0, gt=0)
    forward_retry_max_seconds: float = Field(default=30.0, gt=0)
    reconnect_backoff_initial_seconds: float = Field(default=1.0, gt=0)
    reconnect_backoff_max_seconds: float = Field(default=30.0, gt=0)
    reconnect_backoff_jitter_seconds: float = Field(default=0.5, ge=0)
    degraded_after_seconds: int = Field(default=60, ge=1)

    @property
    def declared_inbound_topics(self) -> tuple[str, ...]:
        """Canonical topics this forwarder accepts FROM the cloud.

        OMN-18691: empty when no cloud bus is declared. A mirror-only forwarder
        declares no trust boundary, so every topic offered at that boundary is
        undeclared and the existing refusals fire unchanged -- which is the
        fail-closed direction. Returning an empty set rather than raising keeps
        the refusal at the call site that already owns it, so the error a
        caller sees is still "not declared for inbound mirroring" rather than a
        shape error about a field it never set.
        """
        return () if self.mirror_topics is None else self.mirror_topics.inbound

    @property
    def declared_outbound_topics(self) -> tuple[str, ...]:
        """Canonical topics this forwarder sends TO the cloud.

        Empty for a mirror-only forwarder, for the reason on
        ``declared_inbound_topics``.
        """
        return () if self.mirror_topics is None else self.mirror_topics.outbound

    @field_validator("dedupe_store_path")
    @classmethod
    def _validate_dedupe_store_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError(
                "dedupe_store_path must be absolute so deployment persistence "
                "cannot depend on the container working directory"
            )
        return value

    @model_validator(mode="after")
    def _validate_liveness_windows(self) -> ModelGatewayForwarderConfig:
        if self.max_silence_window_seconds <= self.heartbeat_interval_seconds:
            raise ValueError(
                "max_silence_window_seconds must exceed heartbeat_interval_seconds"
            )
        if self.forward_retry_max_seconds < self.forward_retry_initial_seconds:
            raise ValueError(
                "forward_retry_max_seconds must be greater than or equal to "
                "forward_retry_initial_seconds"
            )
        self._validate_leg_completeness()
        if self.canary is not None and self.mirror_topics is not None:
            if self.canary.topic in self.mirror_topics.inbound or (
                self.canary.topic in self.mirror_topics.outbound
            ):
                raise ValueError(
                    "canary.topic must be dedicated and must not appear in "
                    "mirror_topics.inbound or mirror_topics.outbound"
                )
        if self.reconnect_backoff_max_seconds < self.reconnect_backoff_initial_seconds:
            raise ValueError(
                "reconnect_backoff_max_seconds must be greater than or equal to "
                "reconnect_backoff_initial_seconds"
            )
        self._validate_egress_redaction_pairing()
        self._validate_egress_metadata_scrub_pairing()
        return self

    def _validate_egress_metadata_scrub_pairing(self) -> None:
        """OMN-19439: the scrub and the outbound set must agree both ways."""
        scrub = self.egress_metadata_scrub
        outbound = set(self.declared_outbound_topics)
        if scrub is not None:
            unmirrored = sorted(set(scrub.scrubbed_topics) - outbound)
            if unmirrored:
                raise ValueError(
                    "egress_metadata_scrub.scrubbed_topics must all appear in "
                    f"mirror_topics.outbound; missing: {unmirrored}"
                )
        scrubbed = set(scrub.scrubbed_topics) if scrub is not None else set()
        unscrubbed = sorted(
            topic
            for topic in outbound
            if requires_metadata_scrub(topic) and topic not in scrubbed
        )
        if unscrubbed:
            raise ValueError(
                "delegate-skill terminal topics may not be mirrored outbound "
                "unless egress_metadata_scrub reduces them to metadata; "
                f"unscrubbed: {unscrubbed}"
            )

    def _validate_leg_completeness(self) -> None:
        """OMN-18691: a forwarder forwards somewhere, and says which somewhere.

        Two refusals, and they close different holes.

        A process with NO leg at all -- no cloud bus and no lane mirror -- boots,
        reports healthy and moves zero records. That is the green-and-silent
        class the dedicated CI bus exists to avoid, reached from the other
        direction, so it is a construction error rather than an empty run.

        A PARTIAL cloud declaration is the second. `mirror_topics` names the set
        that crosses the trust boundary and `canary` probes the round trip over
        it; either one present with no `cloud_bus` is a policy that reads live
        and can never fire, which is the same failure shape as a lane_mirror
        whose broker legs were never resolved. Naming the missing field in the
        message keeps the refusal actionable.
        """
        cloud_fields = {
            "cloud_bus": self.cloud_bus,
            "mirror_topics": self.mirror_topics,
            "canary": self.canary,
        }
        declared = {name for name, value in cloud_fields.items() if value is not None}
        if declared and declared != set(cloud_fields):
            missing = sorted(set(cloud_fields) - declared)
            raise ValueError(
                "the cloud legs travel together: a forwarder declaring any of "
                f"{sorted(declared)} must declare all of cloud_bus, "
                f"mirror_topics and canary; missing: {missing}"
            )
        if not declared and self.lane_mirror is None:
            raise ValueError(
                "a forwarder must carry at least one leg: either the cloud "
                "trust-boundary legs (cloud_bus + mirror_topics + canary) or a "
                "lane_mirror. A process with neither forwards nothing while "
                "reporting healthy"
            )

    def _validate_egress_redaction_pairing(self) -> None:
        """OMN-16979: the widening and the gate must agree, in both directions.

        A gate that names a topic nobody mirrors is dead policy that reads like
        live policy. A capture topic in the outbound set that the gate does NOT
        name is the credential pipeline OMN-17209 exists to prevent -- so it is
        refused here rather than merely discouraged.
        """
        policy = self.egress_redaction
        if self.mirror_topics is None:
            # OMN-18691: a mirror-only forwarder crosses no trust boundary, so
            # there is no egress set for a redaction policy to govern. A policy
            # declared anyway names topics nobody mirrors, which is the dead
            # policy this method already refuses in the cloud case.
            if policy is not None:
                raise ValueError(
                    "egress_redaction governs the cloud egress set, but this "
                    "forwarder declares no mirror_topics (and so no cloud_bus); "
                    "a policy with nothing to govern reads live and is dead"
                )
            return
        outbound = set(self.mirror_topics.outbound)
        if policy is not None:
            ungoverned_declarations = sorted(set(policy.governed_topics) - outbound)
            if ungoverned_declarations:
                raise ValueError(
                    "egress_redaction.governed_topics must all appear in "
                    f"mirror_topics.outbound; missing: {ungoverned_declarations}"
                )
        governed = set(policy.governed_topics) if policy is not None else set()
        unguarded = sorted(
            topic
            for topic in outbound
            if requires_egress_redaction(topic) and topic not in governed
        )
        if unguarded:
            raise ValueError(
                "redaction-required capture topics may not be mirrored outbound "
                "unless egress_redaction declares them governed; unguarded: "
                f"{unguarded}"
            )
