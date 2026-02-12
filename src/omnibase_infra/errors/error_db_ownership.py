# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""DB Ownership Errors.

Error classes raised when database ownership validation fails during startup.
These errors cause the kernel to hard-fail, preventing a service from operating
on a database it does not own.

Related:
    - OMN-2085: Handshake hardening -- DB ownership marker + startup assertion
    - OMN-2056: DB provisioning (databases must exist first)

Error Hierarchy:
    RuntimeHostError
    ├── DbOwnershipMismatchError -- connected DB owned by wrong service
    └── DbOwnershipMissingError -- db_metadata table/row not found
"""

from __future__ import annotations

from uuid import UUID, uuid4

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class DbOwnershipMismatchError(RuntimeHostError):
    """Raised when connected database is owned by a different service.

    This is a P0 hard gate. The kernel must terminate immediately when this
    error is raised to prevent cross-service data corruption.

    Attributes:
        expected_owner: The service name this process expected.
        actual_owner: The service name recorded in db_metadata.
    """

    def __init__(
        self,
        message: str,
        *,
        expected_owner: str,
        actual_owner: str,
        context: ModelInfraErrorContext | None = None,
        correlation_id: UUID | None = None,
        **extra_context: object,
    ) -> None:
        self.expected_owner = expected_owner
        self.actual_owner = actual_owner

        if correlation_id is None:
            correlation_id = uuid4()

        if context is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="validate_db_ownership",
                correlation_id=correlation_id,
            )

        extra_context["expected_owner"] = expected_owner
        extra_context["actual_owner"] = actual_owner

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.CONTRACT_VIOLATION,
            context=context,
            **extra_context,
        )


class DbOwnershipMissingError(RuntimeHostError):
    """Raised when db_metadata table or ownership row is missing.

    Indicates the database has not been migrated or the service is connected
    to an uninitialized database. The kernel must terminate immediately.

    Attributes:
        expected_owner: The service name this process expected.
    """

    def __init__(
        self,
        message: str,
        *,
        expected_owner: str,
        context: ModelInfraErrorContext | None = None,
        correlation_id: UUID | None = None,
        **extra_context: object,
    ) -> None:
        self.expected_owner = expected_owner

        if correlation_id is None:
            correlation_id = uuid4()

        if context is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="validate_db_ownership",
                correlation_id=correlation_id,
            )

        extra_context["expected_owner"] = expected_owner

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.CONTRACT_VIOLATION,
            context=context,
            **extra_context,
        )


__all__ = ["DbOwnershipMismatchError", "DbOwnershipMissingError"]
