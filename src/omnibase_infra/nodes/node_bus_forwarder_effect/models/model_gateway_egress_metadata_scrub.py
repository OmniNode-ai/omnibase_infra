# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Metadata-only egress scrub for the delegate-skill terminals (OMN-19439).

The cloud needs delegation OUTCOMES (MD-42): which task class ran, on which
provider and model, whether it passed its gate, what it cost and how long it
took. It must never receive what was asked or what was answered. A delegate-skill
terminal carries both in the same payload -- ``prompt_text`` and ``response`` at
the top level, and model text again inside every ``attempts[*]`` record
(``reasoning_preamble``, ``acceptance_detail``, ``error_message``).

So the scrub is an ALLOWLIST of top-level payload fields, applied at the trust
boundary before the content-addressed ``event_id`` is computed. Anything not
named is dropped, including every field a producer adds later: a new field is
retained only by a deliberate contract edit, never as a side effect of a
producer change. That is the fail-closed direction.

Retaining a known text-bearing field is refused by construction, whatever the
contract says, for the same reason ``raw`` is never an admissible redaction
state: a policy that retained ``prompt_text`` would read exactly like a working
one while crossing everything it exists to stop.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    validate_canonical_topic,
)

# Top-level delegate-skill terminal fields that carry prompt, response or model
# text, or a secret reference. Never retainable. ``attempts`` is here because
# each attempt record embeds model text; its count crosses as
# ``attempts_count``. The evidence blocks, the per-gate diagnostics and the
# failed acceptance criteria are here because each can quote the response.
_NEVER_RETAINED_FIELDS = frozenset(
    {
        "prompt_text",
        "prompt",
        "response",
        "response_text",
        "content",
        "error_message",
        "attempts",
        "reasoning_preamble",
        "acceptance_detail",
        "response_contract_evidence",
        "budget_evidence",
        "quality_gates_failed",
        "failed_acceptance_criteria",
        "secret_ref",
    }
)


class ModelGatewayEgressMetadataScrub(BaseModel):
    """Contract-declared metadata allowlist for scrubbed outbound topics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Topics whose payload is reduced to ``retained_payload_fields``.
    scrubbed_topics: tuple[str, ...] = Field(..., min_length=1)

    #: The only top-level payload keys that may cross for a scrubbed topic.
    retained_payload_fields: tuple[str, ...] = Field(..., min_length=1)

    @field_validator("scrubbed_topics")
    @classmethod
    def _validate_scrubbed_topics(cls, topics: tuple[str, ...]) -> tuple[str, ...]:
        for topic in topics:
            validate_canonical_topic(topic)
        if len(set(topics)) != len(topics):
            raise ValueError("scrubbed_topics must not repeat a topic")
        return topics

    @field_validator("retained_payload_fields")
    @classmethod
    def _validate_retained_fields(cls, fields: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(fields)) != len(fields):
            raise ValueError("retained_payload_fields must not repeat a field")
        forbidden = sorted(set(fields) & _NEVER_RETAINED_FIELDS)
        if forbidden:
            raise ValueError(
                f"retained_payload_fields may never contain {forbidden}: they "
                "carry prompt, response or model text, or a secret reference"
            )
        return fields

    def governs(self, canonical_topic: str) -> bool:
        """Whether ``canonical_topic`` is reduced to metadata at the boundary."""
        return canonical_topic in self.scrubbed_topics

    def scrub(self, payload: dict[str, object]) -> dict[str, object]:
        """Return a copy of ``payload`` holding only the retained fields."""
        return {
            key: payload[key] for key in self.retained_payload_fields if key in payload
        }


__all__ = ["ModelGatewayEgressMetadataScrub"]
