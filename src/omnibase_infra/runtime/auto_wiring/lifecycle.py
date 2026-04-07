# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lifecycle hook execution engine for contract auto-wiring.

Resolves and invokes lifecycle hooks declared in contracts during node
startup and shutdown. Hooks are resolved from dotted callable references,
invoked with a ModelAutoWiringContext, and produce structured
ModelLifecycleHookResult diagnostics.

Lifecycle Discipline:
    - Hooks are invoked sequentially: on_start -> validate_handshake -> on_shutdown
    - Each hook is wrapped with asyncio.wait_for for timeout enforcement
    - Required hooks that fail abort the lifecycle phase
    - Optional hooks that fail are logged but do not block
    - All hooks must be idempotent (enforced at schema level)

Handshake Discipline (OMN-7657):
    - Handshake runs between on_start and handler wiring
    - Handshake is a pre-subscription readiness check, NOT a second startup phase
    - Failed handshakes are retried per ModelHandshakeConfig
    - Exhausted retries quarantine the contract with structured diagnostics
    - Quarantined contracts are visible in health/readiness endpoints

.. versionadded:: 0.35.0
    Created as part of OMN-7655 (Contract lifecycle hooks).

.. versionchanged:: 0.36.0
    Added handshake validation with retry and quarantine (OMN-7657).
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import Awaitable, Callable

from omnibase_infra.runtime.auto_wiring.context import ModelAutoWiringContext
from omnibase_infra.runtime.auto_wiring.models import (
    HandshakeFailureReason,
    ModelLifecycleHookConfig,
    ModelLifecycleHookResult,
    ModelLifecycleHooks,
    ModelQuarantineRecord,
)

logger = logging.getLogger(__name__)

HookCallable = Callable[[ModelAutoWiringContext], Awaitable[ModelLifecycleHookResult]]


def resolve_hook_callable(callable_ref: str) -> HookCallable:
    """Resolve a dotted path to an async callable.

    Args:
        callable_ref: Dotted import path (e.g., 'mypackage.hooks.on_start').

    Returns:
        The resolved async callable.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the callable is not found in the module.
    """
    parts = callable_ref.rsplit(".", 1)
    module_path, attr_name = parts[0], parts[1]
    module = importlib.import_module(module_path)
    hook_fn = getattr(module, attr_name)
    if not callable(hook_fn):
        msg = f"'{callable_ref}' resolved to a non-callable: {type(hook_fn)}"
        raise AttributeError(msg)
    return hook_fn


def _classify_failure(result: ModelLifecycleHookResult) -> HandshakeFailureReason:
    """Classify a handshake failure into a structured reason."""
    msg = result.error_message.lower()
    if "timed out" in msg:
        return HandshakeFailureReason.TIMEOUT
    if "resolution failed" in msg:
        return HandshakeFailureReason.RESOLUTION_FAILED
    if "raised" in msg:
        return HandshakeFailureReason.HOOK_EXCEPTION
    return HandshakeFailureReason.HOOK_RETURNED_FAILURE


class LifecycleHookExecutor:
    """Executes contract lifecycle hooks during auto-wiring.

    The executor resolves hook callables from their dotted paths, builds
    the appropriate ModelAutoWiringContext, and invokes each hook with
    timeout enforcement and structured error handling.

    The executor also maintains a quarantine registry of contracts whose
    handshake validation failed. Quarantined contracts are excluded from
    handler wiring and visible via ``get_quarantined_contracts()``.

    Thread Safety:
        The quarantine list is append-only during startup. Read access
        via get_quarantined_contracts() is safe after startup completes.
    """

    def __init__(self) -> None:
        self._quarantined: list[ModelQuarantineRecord] = []

    def get_quarantined_contracts(self) -> list[ModelQuarantineRecord]:
        """Return all quarantined contracts for health/readiness reporting."""
        return list(self._quarantined)

    async def execute_hook(
        self,
        hook_config: ModelLifecycleHookConfig,
        context: ModelAutoWiringContext,
    ) -> ModelLifecycleHookResult:
        """Execute a single lifecycle hook with timeout and error handling.

        Args:
            hook_config: Configuration for the hook to execute.
            context: The auto-wiring context to pass to the hook.

        Returns:
            ModelLifecycleHookResult with success/failure status and diagnostics.
        """
        hook_name = context.phase

        try:
            hook_fn = resolve_hook_callable(hook_config.callable_ref)
        except (ImportError, AttributeError) as e:
            logger.exception(
                "Failed to resolve lifecycle hook",
                extra={
                    "callable_ref": hook_config.callable_ref,
                    "phase": hook_name,
                    "error": str(e),
                },
            )
            return ModelLifecycleHookResult.failed(
                hook_name=hook_name,
                error_message=f"Hook resolution failed: {e}",
            )

        try:
            result = await asyncio.wait_for(
                hook_fn(context),
                timeout=hook_config.timeout_seconds,
            )
            logger.debug(
                "Lifecycle hook completed",
                extra={
                    "phase": hook_name,
                    "callable_ref": hook_config.callable_ref,
                    "success": result.success,
                    "background_workers": result.background_workers,
                },
            )
            return result
        except TimeoutError:
            logger.warning(
                "Lifecycle hook timed out",
                extra={
                    "phase": hook_name,
                    "callable_ref": hook_config.callable_ref,
                    "timeout_seconds": hook_config.timeout_seconds,
                },
            )
            return ModelLifecycleHookResult.failed(
                hook_name=hook_name,
                error_message=(
                    f"Hook '{hook_config.callable_ref}' timed out "
                    f"after {hook_config.timeout_seconds}s"
                ),
            )
        except Exception as e:
            logger.exception(
                "Lifecycle hook failed with exception",
                extra={
                    "phase": hook_name,
                    "callable_ref": hook_config.callable_ref,
                    "error": str(e),
                },
            )
            return ModelLifecycleHookResult.failed(
                hook_name=hook_name,
                error_message=f"Hook '{hook_config.callable_ref}' raised: {e}",
            )

    async def execute_handshake(
        self,
        hooks: ModelLifecycleHooks,
        context_kwargs: dict[str, object],
    ) -> ModelLifecycleHookResult | None:
        """Execute the handshake hook with retry and quarantine semantics.

        The handshake is a pre-subscription readiness check that runs
        between on_start and handler wiring. If the handshake fails
        after all retries, the contract is quarantined.

        Args:
            hooks: The lifecycle hooks configuration from the contract.
            context_kwargs: Base kwargs for building ModelAutoWiringContext.
                Must include handler_id, node_kind.

        Returns:
            The final handshake result, or None if no handshake configured.
            On quarantine, the result will have success=False.
        """
        if hooks.validate_handshake is None:
            return None

        hook_config = hooks.validate_handshake
        hs_config = hooks.handshake_config
        handler_id = context_kwargs.get("handler_id", "unknown")
        node_kind = context_kwargs.get("node_kind", "unknown")
        max_attempts = 1 + hs_config.max_retries
        last_result: ModelLifecycleHookResult | None = None

        try:
            result = await asyncio.wait_for(
                self._execute_handshake_attempts(
                    hook_config,
                    context_kwargs,
                    max_attempts,
                    hs_config.retry_delay_seconds,
                ),
                timeout=hs_config.total_timeout_seconds,
            )
            return result
        except TimeoutError:
            last_result = ModelLifecycleHookResult.failed(
                hook_name="validate_handshake",
                error_message=(
                    f"Handshake for '{handler_id}' exceeded total timeout "
                    f"of {hs_config.total_timeout_seconds}s"
                ),
            )

        # Quarantine on total timeout
        record = ModelQuarantineRecord(
            handler_id=handler_id,
            node_kind=node_kind,
            failure_reason=HandshakeFailureReason.TIMEOUT,
            error_message=last_result.error_message,
            attempts=max_attempts,
        )
        self._quarantined.append(record)
        logger.error(
            "Contract quarantined after handshake total timeout",
            extra={
                "handler_id": handler_id,
                "failure_reason": record.failure_reason.value,
                "attempts": record.attempts,
            },
        )
        return last_result

    async def _execute_handshake_attempts(
        self,
        hook_config: ModelLifecycleHookConfig,
        context_kwargs: dict[str, object],
        max_attempts: int,
        retry_delay: float,
    ) -> ModelLifecycleHookResult:
        """Run handshake attempts with retries. Quarantine on exhaustion."""
        handler_id = context_kwargs.get("handler_id", "unknown")
        node_kind = context_kwargs.get("node_kind", "unknown")
        last_result: ModelLifecycleHookResult | None = None

        for attempt in range(1, max_attempts + 1):
            context = ModelAutoWiringContext(
                phase="validate_handshake", **context_kwargs
            )
            result = await self.execute_hook(hook_config, context)

            if result.success:
                if attempt > 1:
                    logger.info(
                        "Handshake succeeded on retry",
                        extra={"handler_id": handler_id, "attempt": attempt},
                    )
                return result

            last_result = result
            logger.warning(
                "Handshake attempt failed",
                extra={
                    "handler_id": handler_id,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "error": result.error_message,
                },
            )

            if attempt < max_attempts:
                await asyncio.sleep(retry_delay)

        # All retries exhausted — quarantine
        assert last_result is not None
        failure_reason = _classify_failure(last_result)
        record = ModelQuarantineRecord(
            handler_id=handler_id,
            node_kind=node_kind,
            failure_reason=failure_reason,
            error_message=last_result.error_message,
            attempts=max_attempts,
        )
        self._quarantined.append(record)
        logger.error(
            "Contract quarantined after handshake failure",
            extra={
                "handler_id": handler_id,
                "failure_reason": failure_reason.value,
                "attempts": max_attempts,
                "error": last_result.error_message,
            },
        )
        return last_result

    async def execute_startup(
        self,
        hooks: ModelLifecycleHooks,
        context_kwargs: dict[str, object],
    ) -> list[ModelLifecycleHookResult]:
        """Execute startup lifecycle hooks: on_start, then validate_handshake.

        on_start is executed as a standard lifecycle hook. validate_handshake
        is executed with retry and quarantine semantics via execute_handshake().

        Args:
            hooks: The lifecycle hooks configuration from the contract.
            context_kwargs: Base kwargs for building ModelAutoWiringContext.
                Must include handler_id, node_kind. Phase is set automatically.

        Returns:
            List of results from executed hooks. Empty if no hooks configured.
            If on_start is required and fails, handshake is skipped.
        """
        results: list[ModelLifecycleHookResult] = []

        # Phase 1: on_start
        if hooks.on_start is not None:
            context = ModelAutoWiringContext(phase="on_start", **context_kwargs)
            result = await self.execute_hook(hooks.on_start, context)
            results.append(result)

            if not result.success and hooks.on_start.required:
                logger.error(
                    "Required on_start hook failed, skipping handshake",
                    extra={
                        "handler_id": context_kwargs.get("handler_id"),
                        "error": result.error_message,
                    },
                )
                return results

        # Phase 2: validate_handshake (with retry + quarantine)
        handshake_result = await self.execute_handshake(hooks, context_kwargs)
        if handshake_result is not None:
            results.append(handshake_result)

        return results

    async def execute_shutdown(
        self,
        hooks: ModelLifecycleHooks,
        context_kwargs: dict[str, object],
    ) -> ModelLifecycleHookResult | None:
        """Execute the on_shutdown lifecycle hook.

        Args:
            hooks: The lifecycle hooks configuration from the contract.
            context_kwargs: Base kwargs for building ModelAutoWiringContext.

        Returns:
            Result from the shutdown hook, or None if no hook configured.
        """
        if hooks.on_shutdown is None:
            return None

        context = ModelAutoWiringContext(phase="on_shutdown", **context_kwargs)
        return await self.execute_hook(hooks.on_shutdown, context)


__all__ = [
    "HookCallable",
    "LifecycleHookExecutor",
    "resolve_hook_callable",
    "_classify_failure",
]
