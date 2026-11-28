# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-bound diagnosis for the delegation identity doctor check."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import DelegateTenantRefusedError
from omnibase_infra.doctor.delegation_doctor_support import (
    GATEWAY_BLOCK_INVALID,
    GATEWAY_BLOCK_MISSING,
)
from omnibase_infra.doctor.enum_delegation_doctor_fault import (
    EnumDelegationDoctorFault,
)
from omnibase_infra.doctor.model_delegation_diagnosis import (
    ModelDelegationDiagnosis,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory

_LOCAL_IDENTITY_FIX = "onex local init"
_IDENTITY_UNREADABLE = (
    "The saved delegation tenant identity on this machine could not be read."
)
_GATEWAY_READER = Callable[[Path], tuple[str, dict[str, object] | None]]

__all__ = [
    "DelegateTenantRefusedError",
    "HandlerCheckDelegationIdentity",
    "evaluate_identity",
    "missing_gateway_reason",
]


class HandlerCheckDelegationIdentity:
    """Own the local-identity and gateway-block diagnosis logic."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    @staticmethod
    def handle(request: object) -> object:
        """Runtime adapter entry point; doctor check uses the static seam."""
        return request

    @staticmethod
    def evaluate_identity(
        onex_home: Path,
        local_identity: str | None,
        refusal_cause: BaseException | None,
        gateway_reader: _GATEWAY_READER,
    ) -> tuple[ModelDelegationDiagnosis, bool]:
        """Judge identity from the local tenant identity alone, as onex local identity does."""
        if refusal_cause is not None:
            return (
                ModelDelegationDiagnosis(
                    fault=None,
                    detail=f"{_IDENTITY_UNREADABLE} Fix: {_LOCAL_IDENTITY_FIX}",
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
        return (
            ModelDelegationDiagnosis(
                fault=EnumDelegationDoctorFault.NO_IDENTITY,
                detail="No delegation tenant identity is configured on this machine.",
                fix=_LOCAL_IDENTITY_FIX,
            ),
            True,
        )


def evaluate_identity(
    onex_home: Path,
    local_identity: str | None,
    refusal_cause: BaseException | None,
    gateway_reader: _GATEWAY_READER,
) -> tuple[ModelDelegationDiagnosis, bool]:
    """Keep the baselined doctor check thin."""
    return HandlerCheckDelegationIdentity.evaluate_identity(
        onex_home,
        local_identity,
        refusal_cause,
        gateway_reader,
    )


def missing_gateway_reason() -> str:
    """Say why no gateway block exists, asking the one local-identity reader.

    The identity check and every check that depends on it must agree on whether
    this machine has an identity, so they all ask ``read_install_identity``.
    """
    try:
        identity = cli_delegate.read_install_identity()
    except DelegateTenantRefusedError as exc:
        cause: BaseException = exc.__cause__ or exc
        if not isinstance(cause, (ImportError, AttributeError)):
            return (
                "the saved tenant identity on this machine could not be read "
                "(run: onex local init)"
            )
        identity = None
    if identity is None:
        return (
            "no delegation identity exists on this machine yet (run: onex local init)"
        )
    return "no delegation gateway address is saved in config.yaml"
