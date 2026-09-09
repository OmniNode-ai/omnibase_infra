# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Result model for the read-only DLQ depth probe (OMN-16769)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_depth_evaluate_result import (
    ModelDlqDepthEvaluateResult,
)


class ModelDlqDepthMonitorResult(BaseModel):
    """Probe outcome: what was swept, plus the full evaluation.

    OMN-18088 -- THE ALERT IS A VALUE, NOT AN EXCEPTION.
    ---------------------------------------------------
    The alerting run used to produce no result at all: the handler formatted
    the offender list into a ``RuntimeHostError`` and raised it, ``RuntimeLocal``
    caught that and recorded ``result=failed`` without re-raising, and receipt
    mode's typed-result branch is gated on success -- so scheduled run
    ``34399417506`` went red carrying ``handler_result: null``,
    ``terminal_payload: null`` and ``error: ""``, naming no topic and stating no
    reason. The only copy of the offender list lived inside an exception that
    two layers of runtime discarded between them.

    So the gating decision is a FIELD now. The alerting run and the
    characterization run return the identical shape, the workflow reads
    ``alert_exit_requested`` and takes its non-zero exit from that, and the run
    that goes red is the same run that wrote down why.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(...)
    evaluated_at: datetime = Field(...)
    window_seconds: int = Field(..., ge=60)
    topics_matched: int = Field(
        default=0,
        ge=0,
        description="Topics matching the DLQ prefix on this broker.",
    )
    evaluation: ModelDlqDepthEvaluateResult = Field(
        ..., description="The per-topic histogram and alert decision."
    )
    suppress_alert_exit: bool = Field(
        default=False,
        description=(
            "Echo of the request knob that decided whether a breach may gate "
            "this run. Recorded so the result file says on its face whether it "
            "came from a characterization run or a scheduled one -- a reader "
            "of a green result otherwise cannot tell 'nothing breached' from "
            "'a breach was deliberately not gated on'."
        ),
    )
    alert_exit_requested: bool = Field(
        default=False,
        description=(
            "The run-gating decision, stated once: a sink breached its bound "
            "AND this run was not a characterization run. The scheduled "
            "workflow reads exactly this field and exits non-zero on it. "
            "Derived, never asserted -- see the validator below."
        ),
    )

    @property
    def alert_triggered(self) -> bool:
        """Convenience passthrough — did any sink breach its bound at all.

        Distinct from :attr:`alert_exit_requested`, which additionally accounts
        for ``suppress_alert_exit``. A characterization run can be True here and
        False there, and conflating the two is how a suppressed run would gate.
        """
        return self.evaluation.alert_triggered

    @model_validator(mode="after")
    def _alert_exit_is_derived_from_the_evidence(self) -> ModelDlqDepthMonitorResult:
        """Refuse a result whose gating decision contradicts its own histogram.

        Same property ``ModelLabPassReceipt`` enforces on its verdict, and for
        the same reason: a field an emitter can set independently of the
        evidence beside it is a field that can be set wrongly, and both
        directions are harmful. A False here on a breaching evaluation is a
        MISSED alert; a True on a clean one is a false alarm that trains
        operators to ignore the surface.
        """
        expected = self.evaluation.alert_triggered and not self.suppress_alert_exit
        if self.alert_exit_requested != expected:
            raise ValueError(
                "alert_exit_requested disagrees with its own evidence: "
                f"alert_exit_requested={self.alert_exit_requested} but "
                f"evaluation.alert_triggered={self.evaluation.alert_triggered} "
                f"and suppress_alert_exit={self.suppress_alert_exit} derive "
                f"{expected}."
            )
        return self


__all__ = ["ModelDlqDepthMonitorResult"]
