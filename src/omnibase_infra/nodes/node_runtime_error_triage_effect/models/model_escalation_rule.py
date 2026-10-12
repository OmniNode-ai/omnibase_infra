# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Escalation rule declared in the node contract (OMN-21029).

The rule is data in ``contract.yaml`` (``escalation:``); the decision it makes
is a pure function of the incident the handler just wrote, so it lives here and
the handler only acts on the answer.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.enum_runtime_error_severity import (
    EnumRuntimeErrorSeverity,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.models.enum_escalation_reason import (
    EnumEscalationReason,
)

_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contract.yaml"

# Incident states an escalation may move out of. ``escalated`` is deliberately
# absent: an escalated incident is never escalated again.
ESCALATABLE_STATES: tuple[str, ...] = ("open",)

# Only the ``alert`` action records an incident and stops there. ``suppress``
# means "ignore" and ``ticket`` already has its own path.
ESCALATABLE_ACTIONS: tuple[str, ...] = ("alert",)

_SEVERITY_ORDER: tuple[EnumRuntimeErrorSeverity, ...] = (
    EnumRuntimeErrorSeverity.WARNING,
    EnumRuntimeErrorSeverity.ERROR,
    EnumRuntimeErrorSeverity.CRITICAL,
)


class ModelEscalationRule(BaseModel):
    """When an open runtime error incident becomes an escalation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    occurrence_threshold: int = Field(
        ...,
        ge=2,
        description="An open incident escalates when its occurrence count reaches this.",
    )
    new_fingerprint_severity: EnumRuntimeErrorSeverity = Field(
        ...,
        description=(
            "A fingerprint whose severity is at least this escalates on its "
            "first occurrence, without waiting for the occurrence threshold."
        ),
    )

    def evaluate(
        self,
        *,
        incident_state: str,
        occurrence_count: int,
        severity: EnumRuntimeErrorSeverity,
        action: str,
    ) -> EnumEscalationReason | None:
        """Decide whether the incident just written crosses the rule.

        Pure: no I/O, no clock. ``None`` means do not escalate.
        """
        if incident_state not in ESCALATABLE_STATES:
            return None
        if action not in ESCALATABLE_ACTIONS:
            return None
        if occurrence_count >= self.occurrence_threshold:
            return EnumEscalationReason.OCCURRENCE_THRESHOLD
        if occurrence_count == 1 and _SEVERITY_ORDER.index(
            severity
        ) >= _SEVERITY_ORDER.index(self.new_fingerprint_severity):
            return EnumEscalationReason.NEW_FINGERPRINT_SEVERITY
        return None


def load_escalation_rule(contract_path: Path = _CONTRACT_PATH) -> ModelEscalationRule:
    """Read the ``escalation:`` block of the node contract."""
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    return ModelEscalationRule.model_validate(contract["escalation"])


__all__ = [
    "ESCALATABLE_ACTIONS",
    "ESCALATABLE_STATES",
    "ModelEscalationRule",
    "load_escalation_rule",
]
