# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed egress-redaction admission policy for the outbound cloud leg.

OMN-16979 widens ``mirror_topics.outbound`` to the content-bearing omniclaude
hook classes. OMN-17209's framing is why that widening cannot be bare: "widening
the payload without landing this contract first ships a credential pipeline."

The redaction itself is produced UPSTREAM, at omnimarket's emit seam (OMN-16019's
per-topic transform, governed by ``capture_redaction.yaml``). That is the only
place that knows tool semantics -- which fields of a ``Bash`` record are a
command name and which are its output. This node cannot re-derive that judgement
and deliberately does not try.

What this node owns is the trust boundary, and what a trust boundary can do
without duplicating upstream logic is REFUSE TO CROSS ANYTHING NOT PROVABLY
REDACTED. So the contract declares, per governed topic, the redaction state a
record must carry to be admitted; a record on a governed topic that does not
carry one is dropped at the boundary.

Two properties follow, and both are the point:

1. The widening is safe to merge before the upstream contract deploys. Until the
   emit seam stamps the field, the widened topics admit nothing, so an early
   merge opens no hole.
2. If the upstream transform is ever removed, bypassed, or silently regresses,
   the widened topics STOP crossing rather than start leaking. The failure mode
   is loss of telemetry, not disclosure.

``raw`` is refused by construction rather than by configuration. It is
``ArtifactStore``'s default state (OMN-13152) -- 27 of 27 artifact-captured
records in the local spool carry it -- so a policy that admitted it would read
exactly like a working one while gating nothing.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    validate_canonical_topic,
)

# Values of omnibase_core's ``EnumArtifactRedactionState`` (OMN-13152). Mirrored
# rather than imported for the same reason the upstream contract mirrors them:
# this is a wire-level admission check on an untrusted string that arrived from
# the bus, not a use of the enum's behaviour. Parity is asserted by test.
_KNOWN_REDACTION_STATES = frozenset(
    {"raw", "redacted", "restricted", "secret_detected"}
)

# Never admissible, whatever a contract says. See the module docstring.
_NEVER_ADMITTED_STATES = frozenset({"raw"})


class ModelGatewayEgressRedaction(BaseModel):
    """Contract-declared admission policy for redaction-governed outbound topics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Payload key carrying the upstream-stamped state. Declared rather than
    #: hardcoded so a producer that names it differently is a contract edit.
    state_field: str = Field(..., min_length=1)

    #: States that may cross. ``secret_detected`` is admissible on purpose: it
    #: means the upstream scrub FIRED and replaced the value, which is a
    #: stronger guarantee than ``redacted``, not a weaker one.
    admitted_states: tuple[str, ...] = Field(..., min_length=1)

    #: Topics this policy governs. An ungoverned topic is untouched by the gate,
    #: so the pre-existing delegation legs keep crossing unchanged.
    governed_topics: tuple[str, ...] = Field(..., min_length=1)

    @field_validator("admitted_states")
    @classmethod
    def _validate_admitted_states(cls, states: tuple[str, ...]) -> tuple[str, ...]:
        unknown = sorted(set(states) - _KNOWN_REDACTION_STATES)
        if unknown:
            raise ValueError(
                "admitted_states must name EnumArtifactRedactionState values "
                f"(OMN-13152); unknown: {unknown}"
            )
        forbidden = sorted(set(states) & _NEVER_ADMITTED_STATES)
        if forbidden:
            raise ValueError(
                f"admitted_states may never contain {forbidden} -- it is the "
                "ArtifactStore default, so admitting it makes the gate a no-op"
            )
        return states

    @field_validator("governed_topics")
    @classmethod
    def _validate_governed_topics(cls, topics: tuple[str, ...]) -> tuple[str, ...]:
        for topic in topics:
            validate_canonical_topic(topic)
        if len(set(topics)) != len(topics):
            raise ValueError("governed_topics must not repeat a topic")
        return topics

    def governs(self, canonical_topic: str) -> bool:
        """Whether this policy applies to ``canonical_topic``."""
        return canonical_topic in self.governed_topics

    def admits(self, payload: dict[str, object]) -> bool:
        """Whether ``payload`` carries an admitted redaction state.

        A missing field, a non-string value, and an unadmitted state are all
        the same answer: no. There is no "unknown" branch, because an
        indeterminate redaction posture at a trust boundary is a refusal.
        """
        state = payload.get(self.state_field)
        if not isinstance(state, str):
            return False
        return state in self.admitted_states
