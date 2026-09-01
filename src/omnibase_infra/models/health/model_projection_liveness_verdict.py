# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection liveness verdict for the runtime health surface (OMN-16994).

Two facts about contract-declared projections that every pre-existing liveness
signal missed, because each of them measures *connectedness* rather than
*persistence*:

* **unattached** — the contract declares a projection but no consumer for its
  topics exists on this runtime's bus registry, so nothing is consumed at all.
* **DLQ-saturated** — a consumer IS attached and IS consuming, and routes 100%
  of what it takes to a DLQ/quarantine sink. Offsets commit on the DLQ route, so
  consumer lag reads 0 and every lag-based check reads green over a total loss.
* **non-writing** (OMN-17448) — a consumer IS attached and IS consuming, and its
  in-process dispatch is a deliberate no-op because the handler has the
  standalone-runner shape (OMN-15905). Offsets commit, nothing raises, nothing
  is DLQ'd, and the rows exist only if a dedicated writer process is deployed
  elsewhere. Both fields above read green through this by construction.

The projection unit itself lives in
:mod:`omnibase_infra.models.health.model_projection_contract_ref`.

The verdict is raw counts plus names. It carries no HTTP status and no
``HEALTHY``/``DEGRADED`` word: the mapping from these facts to a health
dimension belongs to the health surface that renders them
(:mod:`omnibase_infra.runtime.health.projection_liveness`), not to the model.

The two ``*_evaluated`` flags are load-bearing and must not be collapsed into
"empty list means fine". An empty attached-topic registry means "this process
cannot tell", which is a different fact from "nothing attached" — reading
absence as evidence is the exact failure this ticket exists to close.

Related Tickets:
    - OMN-16994: this model (OMN-16843 AC6, deferred)
    - OMN-16777: the consumer-flow windows the saturation half is derived from
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelProjectionLivenessVerdict(BaseModel):
    """Per-cycle projection liveness facts for one runtime process."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    projection_count: int = Field(
        ..., ge=0, description="Contract-declared projections in scope this cycle"
    )
    attachment_evaluated: bool = Field(
        ...,
        description=(
            "True when a live subscription registry was readable. False means "
            "UNKNOWN — never that every projection attached."
        ),
    )
    unattached_projections: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Projections with at least one topic that has no live consumer",
    )
    saturation_evaluated: bool = Field(
        ...,
        description=(
            "True when at least one closed flow window was observable. False "
            "means UNKNOWN — never that nothing is DLQ-saturated."
        ),
    )
    dlq_saturated_projections: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Projections routing 100% of observed traffic to a DLQ sink",
    )
    observed_window_count: int = Field(
        default=0, ge=0, description="Closed flow windows the ratio was taken over"
    )
    nonwriting_projections: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "OMN-17448. Projections this process SUBSCRIBES and deliberately "
            "does not dispatch: the standalone-runner branch returns None "
            "before any handler runs, so the consumer takes every message and "
            "commits every offset while persisting nothing here. Not a defect "
            "on its own -- the rows depend on a dedicated writer process this "
            "one cannot see -- but invisible to both fields above, because the "
            "topic IS attached and nothing raises so nothing reaches a DLQ."
        ),
    )


__all__: list[str] = ["ModelProjectionLivenessVerdict"]
