# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Doctor check for delegation gateway quota."""

from pathlib import Path
from typing import ClassVar

from omnibase_core.doctor.doctor_check_base import DoctorCheckBase
from omnibase_core.enums.enum_doctor_category import EnumDoctorCategory
from omnibase_core.errors.model_onex_error import ModelOnexError
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
    request_whoami_status,
    run_async,
    unknown_result,
)
from omnibase_infra.doctor.enum_delegation_doctor_fault import (
    EnumDelegationDoctorFault,
)
from omnibase_infra.doctor.model_delegation_diagnosis import (
    ModelDelegationDiagnosis,
)
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.gateway.client.gateway_identity_verifier import (
    ProtocolWhoamiTransport,
)
from omnibase_infra.gateway.client.gateway_transport_httpx import (
    GatewayTransportHttpx,
)
from omnibase_infra.gateway.client.store_gateway_credential import (
    StoreGatewayCredential,
)
from omnibase_infra.gateway.models.model_gateway_api_key import (
    ModelGatewayApiKeyCredential,
)


class CheckDelegationQuota(DoctorCheckBase):
    """Distinguish an exhausted quota from authentication and reachability."""

    check_id: ClassVar[str] = "delegation_quota"
    check_name: ClassVar[str] = "Delegation quota"
    category: ClassVar[EnumDoctorCategory] = EnumDoctorCategory.SERVICES

    def __init__(
        self,
        *,
        onex_home: Path | None = None,
        transport: ProtocolWhoamiTransport | None = None,
    ) -> None:
        self._onex_home = onex_home if onex_home is not None else default_onex_home()
        self._transport = (
            transport
            if transport is not None
            else GatewayTransportHttpx(timeout_seconds=5.0)
        )

    def _evaluate(self) -> tuple[ModelDelegationDiagnosis, bool]:
        state, block = read_gateway_block(self._onex_home)
        if state == GATEWAY_BLOCK_INVALID:
            return self._unknown("config.yaml is unreadable or invalid")
        if state == GATEWAY_BLOCK_MISSING or block is None:
            return self._unknown("delegation_identity is not configured")
        if gateway_text(block, "api_key_ref") is None:
            return self._unknown("delegation_key is not configured")

        try:
            credential = StoreGatewayCredential(
                onex_home=self._onex_home
            ).load_read_credential()
        except ModelOnexError:
            return self._unknown("delegation_key is not readable")
        if not isinstance(credential, ModelGatewayApiKeyCredential):
            return self._unknown("delegation_key is not an API key")

        try:
            status = run_async(
                lambda: request_whoami_status(
                    transport=self._transport,
                    credential=credential,
                )
            )
        except InfraUnavailableError:
            return self._unknown("delegation_gateway could not be reached")

        if status == 429:
            return (
                ModelDelegationDiagnosis(
                    fault=EnumDelegationDoctorFault.QUOTA_EXHAUSTED,
                    detail="The gateway reports that the tenant quota is exhausted.",
                    fix="Wait for the tenant quota window to reset, then re-run this check.",
                ),
                True,
            )
        if status in (401, 403):
            return self._unknown("delegation_key was refused")
        if status != 200:
            return self._unknown(f"delegation_gateway answered HTTP {status}")
        return (
            ModelDelegationDiagnosis(
                fault=None,
                detail="The delegation quota probe was accepted.",
                fix="",
            ),
            True,
        )

    @staticmethod
    def _unknown(reason: str) -> tuple[ModelDelegationDiagnosis, bool]:
        return (
            ModelDelegationDiagnosis(
                fault=None,
                detail=f"Delegation quota was not judged because {reason}.",
                fix="",
            ),
            False,
        )

    def diagnose(self) -> ModelDelegationDiagnosis:
        """Return the quota diagnosis without raising."""
        try:
            return self._evaluate()[0]
        except Exception:  # noqa: BLE001 - doctor diagnostics are total functions
            return self._unknown("the check failed safely")[0]

    def run(self) -> ModelDoctorCheckResult:
        """Run the quota check without allowing an exception to escape."""
        try:
            diagnosis, judged = self._evaluate()
            return render_diagnosis(
                name=self.check_name,
                diagnosis=diagnosis,
                judged=judged,
            )
        except Exception:  # noqa: BLE001 - required total doctor boundary
            return unknown_result(name=self.check_name, dependency=self.check_id)


__all__ = ["CheckDelegationQuota"]
