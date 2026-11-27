# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-bound diagnosis for the delegation identity doctor check."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from omnibase_infra.cli.cli_delegate import DelegateTenantRefusedError
from omnibase_infra.doctor.delegation_doctor_support import (
    GATEWAY_BLOCK_INVALID,
    GATEWAY_BLOCK_MISSING,
    gateway_text,
)
from omnibase_infra.doctor.enum_delegation_doctor_fault import (
    EnumDelegationDoctorFault,
)
from omnibase_infra.doctor.model_delegation_diagnosis import (
    ModelDelegationDiagnosis,
)

_LOCAL_IDENTITY_FIX = "onex local init"
_GATEWAY_READER = Callable[[Path], tuple[str, dict[str, object] | None]]

__all__ = [
    "DelegateTenantRefusedError",
    "HandlerCheckDelegationIdentity",
    "evaluate_identity",
]


class HandlerCheckDelegationIdentity:
    """Own the local-identity and gateway-block diagnosis logic."""

    @staticmethod
    def handle(request: object) -> object:
        """Runtime adapter entry point; doctor check uses the static seam."""
        return request

    @staticmethod
    def evaluate_identity(
        onex_home: Path,
        local_identity: str | None,
        refusal_cause: BaseException | None,
        refusal_detail: str,
        gateway_reader: _GATEWAY_READER,
    ) -> tuple[ModelDelegationDiagnosis, bool]:
        """Judge the local identity first, then keep the cloud gateway path."""
        if refusal_cause is not None:
            return (
                ModelDelegationDiagnosis(
                    fault=None,
                    detail=refusal_detail,
                    fix="",
                ),
                False,
            )

        if local_identity is not None:
            return (
                ModelDelegationDiagnosis(
                    fault=None,
                    detail="Delegation identity is configured.",
                    fix="",
                ),
                True,
            )

        state, block = gateway_reader(onex_home)
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
                detail="Delegation identity is configured.",
                fix="",
            ),
            True,
        )


def evaluate_identity(
    onex_home: Path,
    local_identity: str | None,
    refusal_cause: BaseException | None,
    refusal_detail: str,
    gateway_reader: _GATEWAY_READER,
) -> tuple[ModelDelegationDiagnosis, bool]:
    """Keep the baselined doctor check thin."""
    return HandlerCheckDelegationIdentity.evaluate_identity(
        onex_home,
        local_identity,
        refusal_cause,
        refusal_detail,
        gateway_reader,
    )
