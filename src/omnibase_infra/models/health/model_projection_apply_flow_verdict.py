# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Projection apply-flow verdict for the runtime health surface (OMN-18910).

Two facts about contract-declared projections that no existing dimension
carries, because every one of them measures whether a consumer is attached and
moving rather than whether it is moving and WRITING:

* **diverging** — the projection's consumed count advanced past the declared
  floor while its upserted count stayed at zero. That is a consumer refusing,
  returning early, or dropping; offsets commit either way, so lag reads zero
  over a total loss (OMN-18880, nine hours behind three green surfaces).
* **drop-accumulating** — the cumulative discarded-delta gauge RISES across the
  retained windows. A high flat gauge is legitimate idempotence and is the
  noise that gets a dimension muted, so it is explicitly not graded.

As with :class:`ModelProjectionLivenessVerdict`, the verdict is raw counts plus
names and carries no ``HEALTHY``/``DEGRADED`` word: the mapping from these
facts to a dimension status belongs to
:mod:`omnibase_infra.runtime.health.projection_apply_flow`.

``apply_flow_evaluated`` is load-bearing and must not be collapsed into "empty
lists mean fine". No closed window means "this process cannot tell", which is a
different fact from "nothing diverged", and reading absence as evidence is the
failure class epic OMN-18906 exists to close.

Related Tickets:
    - OMN-18910: this model (epic OMN-18906 AC-4)
    - OMN-16994: the sibling liveness verdict whose shape this follows
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelProjectionApplyFlowVerdict(BaseModel):
    """Per-cycle projection apply-flow facts for one runtime process."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    projection_count: int = Field(
        ...,
        ge=0,
        description="Projections that dispatched in this process and are in scope",
    )
    observed_window_count: int = Field(
        ...,
        ge=0,
        description="Closed apply windows the verdict was taken over",
    )
    apply_flow_evaluated: bool = Field(
        ...,
        description=(
            "Whether a real measurement backs this verdict. False means no "
            "closed window was available, which is UNKNOWN and is never "
            "rendered as healthy."
        ),
    )
    in_scope_registered: bool = Field(
        ...,
        description=(
            "Whether this process registered any projection dispatch at all. "
            "False is a MEASURED zero — nothing here to diverge — and is a "
            "different fact from an unobserved window."
        ),
    )
    diverging_projections: tuple[str, ...] = Field(
        default=(),
        description="Projections that consumed past the floor and wrote nothing",
    )
    drop_accumulating_projections: tuple[str, ...] = Field(
        default=(),
        description="Projections whose discarded-delta gauge rose across windows",
    )
    indeterminate_drop_projections: tuple[str, ...] = Field(
        default=(),
        description=(
            "Projections whose cumulative gauge went BACKWARDS, so the reading "
            "did not come from one continuous process and cannot be graded flat"
        ),
    )
    excluded_immutable_grain: tuple[str, ...] = Field(
        default=(),
        description=(
            "Projections whose declared key grain is immutable and "
            "content-addressed, for which a dropped delta is the intended "
            "idempotence rather than lost data. Excluded from the drop "
            "dimension, and still rendered, because unreported is not ungated."
        ),
    )
    total_consumed: int = Field(
        default=0, ge=0, description="Envelopes dispatched across all windows in scope"
    )
    total_upserted: int = Field(
        default=0, ge=0, description="Rows persisted across all windows in scope"
    )
    total_refused_by_guard: int = Field(
        default=0,
        ge=0,
        description="Zero-row writes attributed to an ordering guard (OMN-18992)",
    )


__all__ = ["ModelProjectionApplyFlowVerdict"]
