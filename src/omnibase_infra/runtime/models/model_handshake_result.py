# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handshake validation result model.

This module provides the ModelHandshakeResult for representing the outcome
of plugin handshake validation during kernel bootstrap. The handshake gate
ensures that all prerequisite checks (B1-B3) pass before consumers,
dispatchers, or handlers are wired.

Phase State Machine:
    INITIALIZING -> HANDSHAKE_VALIDATE -> HANDSHAKE_ATTEST -> WIRING -> READY

    The handshake validation phase runs between initialize() and wire_handlers()
    in the kernel bootstrap sequence. If any check fails, the kernel aborts
    before wiring handlers.

Related:
    - OMN-2089: Handshake Hardening - Bootstrap Attestation Gate
"""

from __future__ import annotations

from dataclasses import dataclass, field

from omnibase_infra.runtime.models.model_handshake_check_result import (
    ModelHandshakeCheckResult,
)


@dataclass
class ModelHandshakeResult:
    """Result of plugin handshake validation.

    Aggregates results from all validation checks run during the
    HANDSHAKE_VALIDATE phase. The kernel uses this to decide whether
    to proceed to HANDSHAKE_ATTEST and WIRING phases.

    Attributes:
        plugin_id: Identifier of the plugin that produced this result.
        passed: Whether all validation checks passed.
        checks: Individual check results for diagnostics.
        error_message: Summary error message if validation failed.

    Example:
        ```python
        result = ModelHandshakeResult(
            plugin_id="registration",
            passed=True,
            checks=[
                ModelHandshakeCheckResult(
                    check_name="db_ownership",
                    passed=True,
                    message="Database owned by omnibase_infra",
                ),
            ],
        )
        if not result:
            logger.error("Handshake failed: %s", result.error_message)
        ```
    """

    plugin_id: str
    passed: bool
    checks: list[ModelHandshakeCheckResult] = field(default_factory=list)
    error_message: str | None = None

    def __bool__(self) -> bool:
        """Return True if all validation checks passed.

        Warning:
            **Non-standard __bool__ behavior**: Returns ``True`` only when
            ``passed`` is True. Differs from typical dataclass behavior where
            ``bool(instance)`` always returns ``True``.
        """
        return self.passed

    @classmethod
    def all_passed(
        cls,
        plugin_id: str,
        checks: list[ModelHandshakeCheckResult] | None = None,
    ) -> ModelHandshakeResult:
        """Create a result indicating all checks passed.

        Args:
            plugin_id: Identifier of the plugin.
            checks: Optional list of individual check results.

        Returns:
            ModelHandshakeResult with passed=True.
        """
        return cls(
            plugin_id=plugin_id,
            passed=True,
            checks=checks or [],
        )

    @classmethod
    def failed(
        cls,
        plugin_id: str,
        error_message: str,
        checks: list[ModelHandshakeCheckResult] | None = None,
    ) -> ModelHandshakeResult:
        """Create a result indicating validation failure.

        Args:
            plugin_id: Identifier of the plugin.
            error_message: Description of the failure.
            checks: Optional list of individual check results.

        Returns:
            ModelHandshakeResult with passed=False.
        """
        return cls(
            plugin_id=plugin_id,
            passed=False,
            error_message=error_message,
            checks=checks or [],
        )

    @classmethod
    def default_pass(cls, plugin_id: str) -> ModelHandshakeResult:
        """Create a default-pass result for plugins without validation.

        Used when a plugin does not implement validate_handshake().
        The plugin passes by default since it has no checks to run.

        Args:
            plugin_id: Identifier of the plugin.

        Returns:
            ModelHandshakeResult with passed=True and no checks.
        """
        return cls(
            plugin_id=plugin_id,
            passed=True,
            checks=[],
        )


__all__ = [
    "ModelHandshakeResult",
]
