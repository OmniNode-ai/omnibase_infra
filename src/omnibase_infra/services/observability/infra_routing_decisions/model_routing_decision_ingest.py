# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka-boundary ingest model for `onex.evt.omnibase-infra.routing-decision.v1`.

OMN-16025. The delegation routing reducer publishes a ``ModelEventEnvelope``
whose ``payload`` is the routing decision (omnimarket's wire
``ModelRoutingDecision``). This module is the single place that wire shape is
turned into the ``infra_routing_decisions`` row, so the consumer never carries
raw dicts and the writer never guesses at field names.

Two things it deliberately does NOT do:

* It does not accept the pre-envelope flat shape. That shape belonged to
  ``onex.evt.omnibase-infra.routing-decided.v1``, which is emitted only by the
  legacy ``PluginLlm``/``AdapterModelRouter`` path and does not exist as a topic
  on any lane (verified 2026-09-06 on the dev broker: ``rpk topic describe``
  returns no partition row for it, while the canonical topic returns
  HIGH-WATERMARK 860). Accepting both would be a compatibility shim for a
  producer with no traffic.
* It does not invent values for columns the decision does not carry
  (``selection_mode``, ``fallback_indicator``, ``is_fallback``,
  ``candidates_evaluated``, ``candidate_providers``, ``session_id``,
  ``latency_ms``). Those are left to the column defaults declared by migration
  080 rather than written as a plausible-looking lie -- a row claiming
  ``selection_mode='round_robin'`` for a contract-driven tier decision is worse
  than a row that says nothing about the selection mode.
"""

from __future__ import annotations

from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelInfraRoutingDecisionIngest(BaseModel):
    """One routing decision, in the exact shape ``infra_routing_decisions`` stores.

    Field names are the COLUMN names, not the wire names; :meth:`from_envelope`
    owns the translation. ``extra="forbid"`` because every instance is built by
    that one classmethod.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation id of the delegation this decision routed.",
    )
    selected_provider: str = Field(
        description="Backend the decision selected (wire: selected_backend_ref).",
    )
    selected_tier: str = Field(
        description="Routing tier that produced the decision (wire: tier_name).",
    )
    selected_model: str = Field(
        description="Model the decision selected.",
    )
    reason: str = Field(
        description="Why the decision selected that model (wire: rationale).",
    )
    task_type: str | None = Field(
        default=None,
        description="Task classification carried by the original request.",
    )

    @classmethod
    def from_envelope(cls, raw: dict[str, object]) -> Self | None:
        """Build one row from a decoded envelope, or ``None`` if it is not one.

        ``None`` is the caller's DLQ signal. It is returned for a record with no
        dict ``payload`` -- that is a record from a different producer on this
        topic, not a transient failure, so retrying it would loop forever.
        """
        payload = raw.get("payload")
        if not isinstance(payload, dict):
            return None

        correlation_raw = payload.get("correlation_id") or raw.get("correlation_id")
        correlation_id: UUID | None = None
        if correlation_raw is not None:
            try:
                correlation_id = UUID(str(correlation_raw))
            except ValueError:
                return None

        selected_model = payload.get("selected_model")
        if not isinstance(selected_model, str) or not selected_model:
            return None

        task_type = payload.get("task_type")
        return cls(
            correlation_id=correlation_id,
            selected_provider=_as_str(payload.get("selected_backend_ref")),
            selected_tier=_as_str(payload.get("tier_name")),
            selected_model=selected_model,
            reason=_as_str(payload.get("rationale")),
            task_type=task_type if isinstance(task_type, str) else None,
        )


def _as_str(value: object) -> str:
    """Render an optional wire string as the NOT NULL column it becomes.

    The three columns this feeds are ``NOT NULL DEFAULT ''`` in migration 080 and
    the three wire fields all default to ``""`` in the producer, so an absent
    value and an empty value are the same fact here.
    """
    return value if isinstance(value, str) else ""


__all__: list[str] = ["ModelInfraRoutingDecisionIngest"]
