# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract content hash errors (OMN-18709).

Raised when the canonical content hash of a contract file cannot be computed.
The one caller that matters is the runtime manifest builder, which would
otherwise have to invent a value for a field consumers read as an identity.

Error Hierarchy:
    RuntimeHostError
    └── ContractContentHashError -- the contract file could not be read, or the
        manifest carries no path to read it from

Related:
    - OMN-18709: the manifest's per-contract hash becomes a content hash.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)

__all__ = ["ContractContentHashError"]


class ContractContentHashError(RuntimeHostError):
    """Raised when a contract's canonical content hash cannot be computed.

    The alternative -- hashing the empty string, or falling back to the
    name-and-version hash this replaced -- produces a value that is
    indistinguishable downstream from a correct one. An absent manifest is a
    visible gap; a wrong hash is a silent one.

    Attributes:
        contract_path: The contract path that could not be hashed. Empty when
            the manifest carried no path at all.
        contract_name: The contract's declared name, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        contract_path: str = "",
        contract_name: str = "",
        context: ModelInfraErrorContext | None = None,
        correlation_id: UUID | None = None,
        **extra_context: object,
    ) -> None:
        self.contract_path = contract_path
        self.contract_name = contract_name

        if correlation_id is None:
            correlation_id = uuid4()

        if context is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="contract_content_hash",
                correlation_id=correlation_id,
            )

        ctx = dict(extra_context)
        ctx.setdefault("contract_path", contract_path)
        ctx.setdefault("contract_name", contract_name)

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.CONTRACT_VIOLATION,
            context=context,
            **ctx,
        )
