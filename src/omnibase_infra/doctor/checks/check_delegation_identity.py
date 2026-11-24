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
from omnibase_infra.cli.cli_delegate import (
    DelegateTenantRefusedError,
    read_install_identity,
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

_LOCAL_IDENTITY_FIX = "onex local init"

_NO_IDENTITY_FIX = _LOCAL_IDENTITY_FIX


class CheckDelegationIdentity(DoctorCheckBase):
    """Verify that the machine declares a tenant and gateway origin."""

    check_id: ClassVar[str] = "delegation_identity"
    check_name: ClassVar[str] = "Delegation identity"
    category: ClassVar[EnumDoctorCategory] = EnumDoctorCategory.SERVICES

    def __init__(self, *, onex_home: Path | None = None) -> None:
        self._onex_home = onex_home if onex_home is not None else default_onex_home()

    def _evaluate(self) -> tuple[ModelDelegationDiagnosis, bool]:
        try:
            local_identity = read_install_identity()
        except DelegateTenantRefusedError as exc:
            # The seam raises a refusal both for an unreadable recorded
            # identity and when omnimarket itself cannot be imported. A missing
            # module is a deployment detail, not a refusal of a recorded
            # identity, so only that case collapses to "not configured".
            if not isinstance(exc.__cause__, (ImportError, AttributeError)):
                # A recorded-but-corrupt identity stays a refusal; it is not
                # collapsed into no_identity.
                return (
                    ModelDelegationDiagnosis(
                        fault=None,
                        detail=str(exc),
                        fix="",
                    ),
                    False,
                )
            local_identity = None

        if local_identity is not None:
            return (
                ModelDelegationDiagnosis(
                    fault=None,
                    detail=f"Delegation identity '{local_identity}' is configured.",
                    fix="",
                ),
                True,
            )

        # No local identity: keep the gateway identity check available for
        # cloud setups (D3). Fall through to the legacy gateway block handling.
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
                    fix=_LOCAL_IDENTITY_FIX,
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
                    fix=_LOCAL_IDENTITY_FIX,
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
