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

.. versionadded:: 0.35.0
    Created as part of OMN-7655 (Contract lifecycle hooks).
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from omnibase_infra.runtime.auto_wiring.context import ModelAutoWiringContext
from omnibase_infra.runtime.auto_wiring.models import (
    ModelLifecycleHookConfig,
    ModelLifecycleHookResult,
    ModelLifecycleHooks,
)

logger = logging.getLogger(__name__)

# Type alias for hook callables
HookCallable = Callable[
    [ModelAutoWiringContext], Coroutine[Any, Any, ModelLifecycleHookResult]
]


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


class LifecycleHookExecutor:
    """Executes contract lifecycle hooks during auto-wiring.

    The executor resolves hook callables from their dotted paths, builds
    the appropriate ModelAutoWiringContext, and invokes each hook with
    timeout enforcement and structured error handling.

    Thread Safety:
        This class is stateless and thread-safe for concurrent use.
        Each method call operates on its own arguments with no shared state.
    """

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
            logger.error(
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
        except Exception as e:  # noqa: BLE001 — boundary: structured diagnostics
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

    async def execute_startup(
        self,
        hooks: ModelLifecycleHooks,
        context_kwargs: dict[str, Any],
    ) -> list[ModelLifecycleHookResult]:
        """Execute startup lifecycle hooks in order: on_start, validate_handshake.

        Args:
            hooks: The lifecycle hooks configuration from the contract.
            context_kwargs: Base kwargs for building ModelAutoWiringContext.
                Must include handler_id, node_kind. Phase is set automatically.

        Returns:
            List of results from executed hooks. Empty if no hooks configured.
            If a required hook fails, subsequent hooks are skipped.
        """
        results: list[ModelLifecycleHookResult] = []

        for phase, hook_config in [
            ("on_start", hooks.on_start),
            ("validate_handshake", hooks.validate_handshake),
        ]:
            if hook_config is None:
                continue

            context = ModelAutoWiringContext(phase=phase, **context_kwargs)
            result = await self.execute_hook(hook_config, context)
            results.append(result)

            if not result.success and hook_config.required:
                logger.error(
                    "Required startup hook failed, aborting lifecycle",
                    extra={
                        "phase": phase,
                        "handler_id": context_kwargs.get("handler_id"),
                        "error": result.error_message,
                    },
                )
                break

        return results

    async def execute_shutdown(
        self,
        hooks: ModelLifecycleHooks,
        context_kwargs: dict[str, Any],
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
]
