# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure ratchet decision over the wiring-report counts (OMN-26335).

A count above its baseline is a failure. A baseline above the one the base ref
held is a failure, so only a pull request that lowers the contract value can
land. A count below its baseline passes and names the baseline to write.
"""

from __future__ import annotations

from typing import Final, Literal

from omnibase_core.models.validation.model_validation_report import (
    ModelValidationFindingEmbed,
    ModelValidationReport,
    ModelValidationRequestRef,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.wiring_ratchet import (
    ModelWiringCounts,
    ModelWiringReportRatchetRequest,
)

VALIDATOR_ID: Final[str] = "wiring-report-ratchet"
COUNT_FIELDS: Final[tuple[str, ...]] = tuple(ModelWiringCounts.model_fields)


class HandlerWiringReportRatchet:
    """Compare observed counts to the baseline; the baseline can only be lowered."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    def handle(self, request: ModelWiringReportRatchetRequest) -> ModelValidationReport:
        """Decide from the request alone, without filesystem, git, clock or env I/O."""
        findings: list[ModelValidationFindingEmbed] = []
        for name in COUNT_FIELDS:
            observed = getattr(request.observed, name)
            baseline = getattr(request.baseline, name)
            if observed > baseline:
                findings.append(
                    self._finding(
                        "FAIL",
                        name,
                        "wiring-count-above-baseline",
                        f"{request.scope}: {name}={observed} is above its baseline "
                        f"{baseline}.",
                        "Remove the wiring failure, duplicate registration or "
                        "unproduced subscription the change added.",
                    )
                )
            elif observed < baseline:
                findings.append(
                    self._finding(
                        "ADVISORY",
                        name,
                        "wiring-count-below-baseline",
                        f"{request.scope}: {name}={observed} is below its baseline "
                        f"{baseline}; the new baseline is {observed}.",
                        f"Lower {name} to {observed} in the node contract's "
                        "metadata.wiring_ratchet.baseline.",
                    )
                )
            if request.base_baseline is not None:
                base = getattr(request.base_baseline, name)
                if baseline > base:
                    findings.append(
                        self._finding(
                            "FAIL",
                            name,
                            "wiring-baseline-raised",
                            f"{name} baseline {baseline} is above the {base} the "
                            "base ref holds; a baseline can only be lowered.",
                            f"Set the {name} baseline to {base} or lower.",
                        )
                    )
        return ModelValidationReport.from_findings(
            findings=tuple(findings),
            request=ModelValidationRequestRef(profile="default"),
            validators_run=(VALIDATOR_ID,),
        )

    @staticmethod
    def _finding(
        severity: Literal["FAIL", "ADVISORY"],
        name: str,
        rule_id: str,
        message: str,
        remediation: str,
    ) -> ModelValidationFindingEmbed:
        return ModelValidationFindingEmbed(
            validator_id=VALIDATOR_ID,
            severity=severity,
            location=f"wiring_ratchet.{name}",
            message=message,
            remediation=remediation,
            rule_id=rule_id,
        )


__all__ = ["COUNT_FIELDS", "VALIDATOR_ID", "HandlerWiringReportRatchet"]
