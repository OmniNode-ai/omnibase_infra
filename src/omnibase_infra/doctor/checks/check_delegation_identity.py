# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Doctor check for a configured delegation identity."""

import importlib
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
    default_onex_home,
    read_gateway_block,
    render_diagnosis,
    unknown_result,
)
from omnibase_infra.doctor.model_delegation_diagnosis import (
    ModelDelegationDiagnosis,
)

_NO_IDENTITY_FIX = "onex local init"


def _evaluate_identity(onex_home: Path) -> tuple[ModelDelegationDiagnosis, bool]:
    module = importlib.import_module(
        "omnibase_infra.handlers.handler_check_delegation_identity"
    )
    try:
        local_identity = read_install_identity()
        refusal_cause: BaseException | None = None
        refusal_detail = ""
    except DelegateTenantRefusedError as exc:
        cause: BaseException = exc.__cause__ or exc
        local_identity = None
        if isinstance(cause, (ImportError, AttributeError)):
            refusal_cause = None
            refusal_detail = ""
        else:
            refusal_cause = cause
            refusal_detail = str(exc)
    return module.evaluate_identity(
        onex_home,
        local_identity,
        refusal_cause,
        refusal_detail,
        read_gateway_block,
    )


class CheckDelegationIdentity(DoctorCheckBase):
    """Verify that the machine declares a tenant and gateway origin."""

    check_id: ClassVar[str] = "delegation_identity"
    check_name: ClassVar[str] = "Delegation identity"
    category: ClassVar[EnumDoctorCategory] = EnumDoctorCategory.SERVICES

    def __init__(self, *, onex_home: Path | None = None) -> None:
        self._onex_home = (
            onex_home if onex_home is not None else default_onex_home()
        )

    def _evaluate(self) -> tuple[ModelDelegationDiagnosis, bool]:
        return _evaluate_identity(self._onex_home)

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
