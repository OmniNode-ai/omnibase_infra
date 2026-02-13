# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Registry Fingerprint Errors.

Error classes raised when event registry fingerprint validation fails during
startup.  These errors cause the kernel to hard-fail, preventing a service
from emitting events whose registrations have drifted from the expected
manifest artifact.

Related:
    - OMN-2088: Handshake hardening -- Event registry fingerprint + startup assertion
    - OMN-2087: Handshake hardening -- Schema fingerprint manifest + startup assertion

Error Hierarchy:
    RuntimeHostError
    ├── EventRegistryFingerprintMismatchError -- live fingerprint != expected
    └── EventRegistryFingerprintMissingError  -- artifact file not found / unreadable
"""

from __future__ import annotations

from uuid import UUID

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_secret_path


class EventRegistryFingerprintMismatchError(RuntimeHostError):
    """Raised when live event registry fingerprint does not match expected.

    This is a P0 hard gate.  The kernel must terminate immediately when this
    error is raised to prevent emitting events whose registrations have drifted
    (added event types, removed topics, changed required fields, etc.).

    Attributes:
        expected_fingerprint: The fingerprint loaded from the artifact file.
        actual_fingerprint: The fingerprint computed from the live registry.
        diff_summary: Human-readable summary of which event types differ.
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

        if context is None:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="validate_event_registry_fingerprint",
                correlation_id=correlation_id,
            )
        elif context.correlation_id is None:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=context.transport_type,
                operation=context.operation,
                target_name=context.target_name,
                namespace=context.namespace,
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


class EventRegistryFingerprintMissingError(RuntimeHostError):
    """Raised when the expected fingerprint artifact file is not found.

    Indicates the fingerprint manifest has not been generated, or the path
    is misconfigured.  The kernel must terminate immediately.

    Attributes:
        artifact_path: The path that was expected to contain the manifest.
    """

    def __init__(
        self,
        message: str,
        *,
        artifact_path: str,
        context: ModelInfraErrorContext | None = None,
        correlation_id: UUID | None = None,
        **extra_context: object,
    ) -> None:
        self.artifact_path = artifact_path

        if context is None:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="validate_event_registry_fingerprint",
                correlation_id=correlation_id,
            )
        elif context.correlation_id is None:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=context.transport_type,
                operation=context.operation,
                target_name=context.target_name,
                namespace=context.namespace,
                correlation_id=correlation_id,
            )

        safe_artifact_path = sanitize_secret_path(artifact_path)
        ctx = dict(extra_context)
        ctx.setdefault("artifact_path", safe_artifact_path)

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.CONTRACT_VIOLATION,
            context=context,
            **ctx,
        )


__all__: list[str] = [
    "EventRegistryFingerprintMismatchError",
    "EventRegistryFingerprintMissingError",
]
