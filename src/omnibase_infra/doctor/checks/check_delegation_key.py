# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Doctor check for the stored delegation API key."""

from pathlib import Path
from typing import ClassVar

from omnibase_core.doctor.doctor_check_base import DoctorCheckBase
from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.enums.enum_doctor_category import EnumDoctorCategory
from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_core.models.doctor.model_doctor_check_result import (
    ModelDoctorCheckResult,
)
from omnibase_core.protocols.http.protocol_http_client import ProtocolHttpResponse
from omnibase_infra.doctor.delegation_doctor_support import (
    GATEWAY_BLOCK_INVALID,
    GATEWAY_BLOCK_MISSING,
    default_onex_home,
    gateway_text,
    read_gateway_block,
    render_diagnosis,
    run_async,
    unknown_result,
)
from omnibase_infra.doctor.enum_delegation_doctor_fault import (
    EnumDelegationDoctorFault,
)
from omnibase_infra.doctor.model_delegation_diagnosis import (
    ModelDelegationDiagnosis,
)
from omnibase_infra.gateway.client.gateway_identity_verifier import (
    GatewayIdentityVerifier,
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


class CheckDelegationKey(DoctorCheckBase):
    """Verify that the stored key authenticates through gateway whoami."""

    check_id: ClassVar[str] = "delegation_key"
    check_name: ClassVar[str] = "Delegation API key"
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
        self._last_status: int | None = None

    async def get(
        self,
        url: str,
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> ProtocolHttpResponse:
        """Delegate one verifier request while retaining only its raw status."""
        response = await self._transport.get(url, timeout=timeout, headers=headers)
        self._last_status = response.status
        return response

    def _evaluate(self) -> tuple[ModelDelegationDiagnosis, bool]:
        state, block = read_gateway_block(self._onex_home)
        if state == GATEWAY_BLOCK_INVALID:
            return self._unknown("config.yaml is unreadable or invalid")
        if state == GATEWAY_BLOCK_MISSING or block is None:
            return self._unknown("delegation_identity is not configured")

        tenant_slug = gateway_text(block, "tenant_slug")
        base_url = gateway_text(block, "base_url")
        if tenant_slug is None or base_url is None:
            return self._unknown("delegation_identity is incomplete")
        if gateway_text(block, "api_key_ref") is None:
            return (
                ModelDelegationDiagnosis(
                    fault=EnumDelegationDoctorFault.NO_KEY,
                    detail="The delegation identity has no stored API-key reference.",
                    fix=(
                        "pbpaste | onex auth login "
                        f"--tenant-slug {tenant_slug} --base-url {base_url} "
                        "--api-key-stdin"
                    ),
                ),
                True,
            )

        try:
            credential = StoreGatewayCredential(
                onex_home=self._onex_home
            ).load_read_credential()
        except ModelOnexError:
            return (
                ModelDelegationDiagnosis(
                    fault=EnumDelegationDoctorFault.NO_KEY,
                    detail="The referenced delegation API key is not readable.",
                    fix=(
                        "pbpaste | onex auth login "
                        f"--tenant-slug {tenant_slug} --base-url {base_url} "
                        "--api-key-stdin"
                    ),
                ),
                True,
            )
        if not isinstance(credential, ModelGatewayApiKeyCredential):
            return (
                ModelDelegationDiagnosis(
                    fault=EnumDelegationDoctorFault.NO_KEY,
                    detail="The stored gateway credential is not a delegation API key.",
                    fix=(
                        "pbpaste | onex auth login "
                        f"--tenant-slug {tenant_slug} --base-url {base_url} "
                        "--api-key-stdin"
                    ),
                ),
                True,
            )

        self._last_status = None
        verifier = GatewayIdentityVerifier(transport=self, credential=credential)
        try:
            identity = run_async(verifier.verify)
        except ModelOnexError as exc:
            if exc.error_code is EnumCoreErrorCode.AUTHENTICATION_ERROR:
                return (
                    ModelDelegationDiagnosis(
                        fault=EnumDelegationDoctorFault.WRONG_KEY,
                        detail="The gateway refused the stored delegation API key.",
                        fix=(
                            "Mint a new API key in the dashboard, then run "
                            "`pbpaste | onex auth login "
                            f"--tenant-slug {tenant_slug} --base-url {base_url} "
                            "--api-key-stdin`."
                        ),
                    ),
                    True,
                )
            if self._last_status == 429:
                return self._unknown("delegation_quota is exhausted")
            return self._unknown("delegation_gateway could not verify the key")

        return (
            ModelDelegationDiagnosis(
                fault=None,
                detail=f"The stored API key authenticates tenant '{identity.tenant_slug}'.",
                fix="",
            ),
            True,
        )

    @staticmethod
    def _unknown(reason: str) -> tuple[ModelDelegationDiagnosis, bool]:
        return (
            ModelDelegationDiagnosis(
                fault=None,
                detail=f"Delegation API key was not judged because {reason}.",
                fix="",
            ),
            False,
        )

    def diagnose(self) -> ModelDelegationDiagnosis:
        """Return the key diagnosis without raising or revealing the key."""
        try:
            return self._evaluate()[0]
        except Exception:  # noqa: BLE001 - doctor diagnostics are total functions
            return self._unknown("the check failed safely")[0]

    def run(self) -> ModelDoctorCheckResult:
        """Run the key check without allowing an exception to escape."""
        try:
            diagnosis, judged = self._evaluate()
            return render_diagnosis(
                name=self.check_name,
                diagnosis=diagnosis,
                judged=judged,
            )
        except Exception:  # noqa: BLE001 - required total doctor boundary
            return unknown_result(name=self.check_name, dependency=self.check_id)


__all__ = ["CheckDelegationKey"]
