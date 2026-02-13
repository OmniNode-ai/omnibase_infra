# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Schema Fingerprint Errors.

Error classes raised when schema fingerprint validation fails during startup.
These errors cause the kernel to hard-fail, preventing a service from operating
on a database whose schema has drifted from what the code expects.

Related:
    - OMN-2087: Handshake hardening -- Schema fingerprint manifest + startup assertion
    - OMN-2085: Handshake hardening -- DB ownership marker + startup assertion

Error Hierarchy:
    RuntimeHostError
    ├── SchemaFingerprintMismatchError -- live schema fingerprint != expected
    └── SchemaFingerprintMissingError -- expected fingerprint not in db_metadata
"""

from __future__ import annotations

from uuid import UUID, uuid4

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class SchemaFingerprintMismatchError(RuntimeHostError):
    """Raised when live schema fingerprint does not match expected.

    This is a P0 hard gate. The kernel must terminate immediately when this
    error is raised to prevent operating on a database whose schema has drifted
    (missing columns, wrong types, extra tables, etc.).

    Attributes:
        expected_fingerprint: The fingerprint stored in db_metadata.
        actual_fingerprint: The fingerprint computed from the live schema.
        diff_summary: Human-readable summary of which tables differ.
    """

    def __init__(
        self,
        message: str,
        *,
        expected_fingerprint: str,
        actual_fingerprint: str,
        diff_summary: str = "",
        context: ModelInfraErrorContext | None = None,
        correlation_id: UUID | None = None,
        **extra_context: object,
    ) -> None:
        self.expected_fingerprint = expected_fingerprint
        self.actual_fingerprint = actual_fingerprint
        self.diff_summary = diff_summary

        if correlation_id is None:
            correlation_id = uuid4()

        if context is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="validate_schema_fingerprint",
                correlation_id=correlation_id,
            )

        ctx = dict(extra_context)
        ctx.setdefault("expected_fingerprint", expected_fingerprint)
        ctx.setdefault("actual_fingerprint", actual_fingerprint)
        ctx.setdefault("diff_summary", diff_summary)

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.CONTRACT_VIOLATION,
            context=context,
            **ctx,
        )


class SchemaFingerprintMissingError(RuntimeHostError):
    """Raised when expected fingerprint is not found in db_metadata.

    Indicates the database has not been migrated to include schema fingerprint
    columns, or the fingerprint has not been populated. The kernel must
    terminate immediately.

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
                operation="validate_schema_fingerprint",
                correlation_id=correlation_id,
            )

        ctx = dict(extra_context)
        ctx.setdefault("expected_owner", expected_owner)

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.CONTRACT_VIOLATION,
            context=context,
            **ctx,
        )


__all__ = ["SchemaFingerprintMismatchError", "SchemaFingerprintMissingError"]
