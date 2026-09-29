# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Doctor check for a configured delegation identity."""

from pathlib import Path
from typing import ClassVar

from omnibase_core.doctor.doctor_check_base import DoctorCheckBase
from omnibase_core.enums.enum_doctor_category import EnumDoctorCategory
from omnibase_core.models.doctor.model_doctor_check_result import (
    ModelDoctorCheckResult,
)
from omnibase_infra.doctor.delegation_doctor_support import (
    GATEWAY_BLOCK_INVALID,
    GATEWAY_BLOCK_MISSING,
    default_onex_home,
    gateway_text,
    read_gateway_block,
    render_diagnosis,
    unknown_result,
)
from omnibase_infra.doctor.enum_delegation_doctor_fault import (
    EnumDelegationDoctorFault,
)
from omnibase_infra.doctor.model_delegation_diagnosis import (
    ModelDelegationDiagnosis,
)

_NO_IDENTITY_FIX = (
    "pbpaste | onex auth login --tenant-slug <slug> --base-url <origin> --api-key-stdin"
)


class CheckDelegationIdentity(DoctorCheckBase):
    """Verify that the machine declares a tenant and gateway origin."""

    check_id: ClassVar[str] = "delegation_identity"
    check_name: ClassVar[str] = "Delegation identity"
    category: ClassVar[EnumDoctorCategory] = EnumDoctorCategory.SERVICES

    def __init__(self, *, onex_home: Path | None = None) -> None:
        self._onex_home = onex_home if onex_home is not None else default_onex_home()

    def _evaluate(self) -> tuple[ModelDelegationDiagnosis, bool]:
        state, block = read_gateway_block(self._onex_home)
        if state == GATEWAY_BLOCK_INVALID:
            return (
                ModelDelegationDiagnosis(
                    fault=None,
                    detail="Delegation identity was not judged because config.yaml is unreadable or invalid.",
                    fix="",
                ),
                False,
            )
        if state == GATEWAY_BLOCK_MISSING or block is None:
            return (
                ModelDelegationDiagnosis(
                    fault=EnumDelegationDoctorFault.NO_IDENTITY,
                    detail="No delegation tenant identity is configured on this machine.",
                    fix=_NO_IDENTITY_FIX,
                ),
                True,
            )
        tenant_slug = gateway_text(block, "tenant_slug")
        base_url = gateway_text(block, "base_url")
        if tenant_slug is None or base_url is None:
            return (
                ModelDelegationDiagnosis(
                    fault=EnumDelegationDoctorFault.NO_IDENTITY,
                    detail="The gateway block does not declare both tenant_slug and base_url.",
                    fix=_NO_IDENTITY_FIX,
                ),
                True,
            )
        return (
            ModelDelegationDiagnosis(
                fault=None,
                detail=f"Delegation identity '{tenant_slug}' is configured for {base_url}.",
                fix="",
            ),
            True,
        )

    def diagnose(self) -> ModelDelegationDiagnosis:
        """Return the identity diagnosis without raising."""
        try:
            return self._evaluate()[0]
        except Exception:  # noqa: BLE001 - doctor diagnostics are total functions
            return ModelDelegationDiagnosis(
                fault=None,
                detail="Delegation identity was not judged because the check failed safely.",
                fix="",
            )

    def run(self) -> ModelDoctorCheckResult:
        """Run the identity check without allowing an exception to escape."""
        try:
            diagnosis, judged = self._evaluate()
            return render_diagnosis(
                name=self.check_name,
                diagnosis=diagnosis,
                judged=judged,
            )
        except Exception:  # noqa: BLE001 - required total doctor boundary
            return unknown_result(name=self.check_name, dependency=self.check_id)


__all__ = ["CheckDelegationIdentity"]
