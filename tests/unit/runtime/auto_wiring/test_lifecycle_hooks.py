# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for contract lifecycle hooks (OMN-7655).

Covers:
    - ModelLifecycleHookConfig validation (callable_ref, idempotent, timeout)
    - ModelLifecycleHooks composition and has_hooks()
    - ModelLifecycleHookResult factory methods
    - ModelAutoWiringContext construction
    - LifecycleHookExecutor: resolution, timeout, error handling, startup/shutdown
    - ModelDiscoveredContract lifecycle_hooks extraction from YAML
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.runtime.auto_wiring.config import ModelLifecycleHookConfig
from omnibase_infra.runtime.auto_wiring.context import ModelAutoWiringContext
from omnibase_infra.runtime.auto_wiring.models import ModelLifecycleHooks
from omnibase_infra.runtime.auto_wiring.result import ModelLifecycleHookResult
from omnibase_infra.runtime.auto_wiring.wiring import (
    LifecycleHookExecutor,
    resolve_hook_callable,
)
from omnibase_infra.services.contract_publisher.sources.model_discovered import (
    ModelDiscoveredContract,
)

# ---------------------------------------------------------------------------
# ModelLifecycleHookConfig
# ---------------------------------------------------------------------------


class TestModelLifecycleHookConfig:
    """Tests for individual hook configuration validation."""

    def test_valid_config(self) -> None:
        config = ModelLifecycleHookConfig(
            callable_ref="mypackage.hooks.on_start",
        )
        assert config.callable_ref == "mypackage.hooks.on_start"
        assert config.timeout_seconds == 10.0
        assert config.required is True
        assert config.idempotent is True

    def test_callable_ref_must_be_dotted(self) -> None:
        with pytest.raises(ValueError, match="dotted path"):
            ModelLifecycleHookConfig(callable_ref="single_segment")

    def test_callable_ref_invalid_identifier(self) -> None:
        with pytest.raises(ValueError, match="not a valid Python identifier"):
            ModelLifecycleHookConfig(callable_ref="my-package.hooks.start")

    def test_idempotent_must_be_true(self) -> None:
        with pytest.raises(ValueError, match="idempotent=True"):
            ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.start",
                idempotent=False,
            )

    def test_timeout_bounds(self) -> None:
        with pytest.raises(ValueError):
            ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.start",
                timeout_seconds=0.5,
            )
        with pytest.raises(ValueError):
            ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.start",
                timeout_seconds=500.0,
            )

    def test_frozen(self) -> None:
        config = ModelLifecycleHookConfig(callable_ref="pkg.hooks.start")
        with pytest.raises(Exception):
            config.callable_ref = "other.path"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelLifecycleHooks
# ---------------------------------------------------------------------------


class TestModelLifecycleHooks:
    """Tests for the hooks container model."""

    def test_empty_hooks(self) -> None:
        hooks = ModelLifecycleHooks()
        assert not hooks.has_hooks()
        assert hooks.on_start is None
        assert hooks.validate_handshake is None
        assert hooks.on_shutdown is None

    def test_has_hooks_with_on_start(self) -> None:
        hooks = ModelLifecycleHooks(
            on_start=ModelLifecycleHookConfig(callable_ref="pkg.hooks.start"),
        )
        assert hooks.has_hooks()

    def test_has_hooks_with_on_shutdown(self) -> None:
        hooks = ModelLifecycleHooks(
            on_shutdown=ModelLifecycleHookConfig(callable_ref="pkg.hooks.stop"),
        )
        assert hooks.has_hooks()

    def test_all_hooks_set(self) -> None:
        hooks = ModelLifecycleHooks(
            on_start=ModelLifecycleHookConfig(callable_ref="pkg.hooks.start"),
            validate_handshake=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.validate"
            ),
            on_shutdown=ModelLifecycleHookConfig(callable_ref="pkg.hooks.stop"),
        )
        assert hooks.has_hooks()
        assert hooks.on_start is not None
        assert hooks.validate_handshake is not None
        assert hooks.on_shutdown is not None

    def test_frozen(self) -> None:
        hooks = ModelLifecycleHooks()
        with pytest.raises(Exception):
            hooks.on_start = ModelLifecycleHookConfig(  # type: ignore[misc]
                callable_ref="pkg.hooks.start"
            )


# ---------------------------------------------------------------------------
# ModelLifecycleHookResult
# ---------------------------------------------------------------------------


class TestModelLifecycleHookResult:
    """Tests for hook result model."""

    def test_succeeded(self) -> None:
        result = ModelLifecycleHookResult.succeeded("on_start")
        assert result.success is True
        assert result.phase == "on_start"
        assert result.error_message == ""
        assert result.background_workers == []
        assert bool(result) is True

    def test_succeeded_with_workers(self) -> None:
        result = ModelLifecycleHookResult.succeeded(
            "on_start",
            background_workers=["worker_a", "worker_b"],
        )
        assert result.background_workers == ["worker_a", "worker_b"]

    def test_failed(self) -> None:
        result = ModelLifecycleHookResult.failed("on_start", "Connection refused")
        assert result.success is False
        assert result.error_message == "Connection refused"
        assert bool(result) is False


# ---------------------------------------------------------------------------
# ModelAutoWiringContext
# ---------------------------------------------------------------------------


class TestModelAutoWiringContext:
    """Tests for auto-wiring context model."""

    def test_minimal_context(self) -> None:
        ctx = ModelAutoWiringContext(
            handler_id="my.handler",
            node_kind="COMPUTE",
            phase="on_start",
        )
        assert ctx.handler_id == "my.handler"
        assert ctx.node_kind == "COMPUTE"
        assert ctx.phase == "on_start"
        assert ctx.services == {}
        assert ctx.metadata == {}

    def test_full_context(self) -> None:
        ctx = ModelAutoWiringContext(
            handler_id="my.handler",
            node_kind="EFFECT",
            contract_version="1.2.3",
            phase="validate_handshake",
            services={"db": "mock_pool"},
            metadata={"region": "us-east-1"},
        )
        assert ctx.contract_version == "1.2.3"
        assert ctx.services["db"] == "mock_pool"

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValueError):
            ModelAutoWiringContext(
                handler_id="my.handler",
                node_kind="COMPUTE",
                phase="on_start",
                unexpected_field="value",
            )


# ---------------------------------------------------------------------------
# resolve_hook_callable
# ---------------------------------------------------------------------------


class TestResolveHookCallable:
    """Tests for hook callable resolution."""

    def test_resolve_existing_callable(self) -> None:
        # json.dumps is a real callable we can resolve
        fn = resolve_hook_callable("json.dumps")
        assert callable(fn)

    def test_resolve_nonexistent_module(self) -> None:
        with pytest.raises(ImportError):
            resolve_hook_callable("nonexistent_module_xyz.func")

    def test_resolve_nonexistent_attr(self) -> None:
        with pytest.raises(AttributeError):
            resolve_hook_callable("json.nonexistent_function_xyz")


# ---------------------------------------------------------------------------
# LifecycleHookExecutor
# ---------------------------------------------------------------------------


class TestLifecycleHookExecutor:
    """Tests for the hook execution engine."""

    @pytest.fixture
    def executor(self) -> LifecycleHookExecutor:
        return LifecycleHookExecutor()

    @pytest.fixture
    def base_context_kwargs(self) -> dict[str, str]:
        return {
            "handler_id": "test.handler",
            "node_kind": "COMPUTE",
        }

    @pytest.mark.asyncio
    async def test_execute_hook_success(self, executor: LifecycleHookExecutor) -> None:
        hook_config = ModelLifecycleHookConfig(callable_ref="pkg.hooks.start")
        context = ModelAutoWiringContext(
            handler_id="test.handler",
            node_kind="COMPUTE",
            phase="on_start",
        )

        expected_result = ModelLifecycleHookResult.succeeded("on_start")
        mock_fn = AsyncMock(return_value=expected_result)

        with patch(
            "omnibase_infra.runtime.auto_wiring.wiring.resolve_hook_callable",
            return_value=mock_fn,
        ):
            result = await executor.execute_hook(hook_config, context)

        assert result.success is True
        mock_fn.assert_awaited_once_with(context)

    @pytest.mark.asyncio
    async def test_execute_hook_resolution_failure(
        self, executor: LifecycleHookExecutor
    ) -> None:
        hook_config = ModelLifecycleHookConfig(
            callable_ref="nonexistent.module.func",
        )
        context = ModelAutoWiringContext(
            handler_id="test.handler",
            node_kind="COMPUTE",
            phase="on_start",
        )

        result = await executor.execute_hook(hook_config, context)
        assert result.success is False
        assert "resolution failed" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_hook_timeout(self, executor: LifecycleHookExecutor) -> None:
        hook_config = ModelLifecycleHookConfig(
            callable_ref="pkg.hooks.slow",
            timeout_seconds=1.0,
        )
        context = ModelAutoWiringContext(
            handler_id="test.handler",
            node_kind="COMPUTE",
            phase="on_start",
        )

        async def slow_hook(_ctx: ModelAutoWiringContext) -> ModelLifecycleHookResult:
            await asyncio.sleep(10)
            return ModelLifecycleHookResult.succeeded("on_start")

        with patch(
            "omnibase_infra.runtime.auto_wiring.wiring.resolve_hook_callable",
            return_value=slow_hook,
        ):
            result = await executor.execute_hook(hook_config, context)

        assert result.success is False
        assert "timed out" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_hook_exception(
        self, executor: LifecycleHookExecutor
    ) -> None:
        hook_config = ModelLifecycleHookConfig(callable_ref="pkg.hooks.broken")
        context = ModelAutoWiringContext(
            handler_id="test.handler",
            node_kind="COMPUTE",
            phase="on_start",
        )

        async def broken_hook(_ctx: ModelAutoWiringContext) -> ModelLifecycleHookResult:
            msg = "DB connection failed"
            raise RuntimeError(msg)

        with patch(
            "omnibase_infra.runtime.auto_wiring.wiring.resolve_hook_callable",
            return_value=broken_hook,
        ):
            result = await executor.execute_hook(hook_config, context)

        assert result.success is False
        assert "RuntimeError" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_startup_both_hooks(
        self,
        executor: LifecycleHookExecutor,
        base_context_kwargs: dict[str, str],
    ) -> None:
        hooks = ModelLifecycleHooks(
            on_start=ModelLifecycleHookConfig(callable_ref="pkg.hooks.start"),
            validate_handshake=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.validate"
            ),
        )

        mock_fn = AsyncMock(
            side_effect=[
                ModelLifecycleHookResult.succeeded("on_start"),
                ModelLifecycleHookResult.succeeded("validate_handshake"),
            ]
        )

        with patch(
            "omnibase_infra.runtime.auto_wiring.wiring.resolve_hook_callable",
            return_value=mock_fn,
        ):
            results = await executor.execute_startup(hooks, base_context_kwargs)

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_execute_startup_required_hook_failure_aborts(
        self,
        executor: LifecycleHookExecutor,
        base_context_kwargs: dict[str, str],
    ) -> None:
        hooks = ModelLifecycleHooks(
            on_start=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.start",
                required=True,
            ),
            validate_handshake=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.validate"
            ),
        )

        mock_fn = AsyncMock(
            return_value=ModelLifecycleHookResult.failed(
                "on_start", "Resource unavailable"
            )
        )

        with patch(
            "omnibase_infra.runtime.auto_wiring.wiring.resolve_hook_callable",
            return_value=mock_fn,
        ):
            results = await executor.execute_startup(hooks, base_context_kwargs)

        # on_start failed and was required, so validate_handshake was skipped
        assert len(results) == 1
        assert not results[0].success

    @pytest.mark.asyncio
    async def test_execute_startup_optional_hook_continues(
        self,
        executor: LifecycleHookExecutor,
        base_context_kwargs: dict[str, str],
    ) -> None:
        hooks = ModelLifecycleHooks(
            on_start=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.start",
                required=False,
            ),
            validate_handshake=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.validate"
            ),
        )

        mock_fn = AsyncMock(
            side_effect=[
                ModelLifecycleHookResult.failed("on_start", "Non-critical error"),
                ModelLifecycleHookResult.succeeded("validate_handshake"),
            ]
        )

        with patch(
            "omnibase_infra.runtime.auto_wiring.wiring.resolve_hook_callable",
            return_value=mock_fn,
        ):
            results = await executor.execute_startup(hooks, base_context_kwargs)

        # on_start failed but was optional, so validate_handshake still ran
        assert len(results) == 2
        assert not results[0].success
        assert results[1].success

    @pytest.mark.asyncio
    async def test_execute_startup_no_hooks(
        self,
        executor: LifecycleHookExecutor,
        base_context_kwargs: dict[str, str],
    ) -> None:
        hooks = ModelLifecycleHooks()
        results = await executor.execute_startup(hooks, base_context_kwargs)
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_shutdown(
        self,
        executor: LifecycleHookExecutor,
        base_context_kwargs: dict[str, str],
    ) -> None:
        hooks = ModelLifecycleHooks(
            on_shutdown=ModelLifecycleHookConfig(callable_ref="pkg.hooks.stop"),
        )

        mock_fn = AsyncMock(
            return_value=ModelLifecycleHookResult.succeeded("on_shutdown")
        )

        with patch(
            "omnibase_infra.runtime.auto_wiring.wiring.resolve_hook_callable",
            return_value=mock_fn,
        ):
            result = await executor.execute_shutdown(hooks, base_context_kwargs)

        assert result is not None
        assert result.success

    @pytest.mark.asyncio
    async def test_execute_shutdown_no_hook(
        self,
        executor: LifecycleHookExecutor,
        base_context_kwargs: dict[str, str],
    ) -> None:
        hooks = ModelLifecycleHooks()
        result = await executor.execute_shutdown(hooks, base_context_kwargs)
        assert result is None


# ---------------------------------------------------------------------------
# ModelDiscoveredContract lifecycle_hooks extraction
# ---------------------------------------------------------------------------


class TestDiscoveredContractLifecycleExtraction:
    """Tests for lifecycle hook extraction from contract YAML."""

    def test_contract_without_lifecycle(self) -> None:
        contract = ModelDiscoveredContract(
            origin="filesystem",
            ref="/app/contracts/foo/contract.yaml",
            text="handler_id: foo.handler\n",
        )
        extracted = contract.extract_handler_id()
        assert extracted.handler_id == "foo.handler"
        assert extracted.lifecycle_hooks is None

    def test_contract_with_lifecycle_hooks(self) -> None:
        yaml_text = (
            "handler_id: foo.handler\n"
            "lifecycle:\n"
            "  on_start:\n"
            "    callable_ref: foo.hooks.on_start\n"
            "  on_shutdown:\n"
            "    callable_ref: foo.hooks.on_shutdown\n"
        )
        contract = ModelDiscoveredContract(
            origin="filesystem",
            ref="/app/contracts/foo/contract.yaml",
            text=yaml_text,
        )
        extracted = contract.extract_handler_id()
        assert extracted.handler_id == "foo.handler"
        assert extracted.lifecycle_hooks is not None
        assert extracted.lifecycle_hooks.on_start is not None
        assert extracted.lifecycle_hooks.on_start.callable_ref == "foo.hooks.on_start"
        assert extracted.lifecycle_hooks.on_shutdown is not None
        assert extracted.lifecycle_hooks.validate_handshake is None

    def test_contract_with_invalid_lifecycle_still_extracts_handler_id(self) -> None:
        yaml_text = (
            "handler_id: foo.handler\n"
            "lifecycle:\n"
            "  on_start:\n"
            "    callable_ref: invalid\n"  # single segment, will fail validation
        )
        contract = ModelDiscoveredContract(
            origin="filesystem",
            ref="/app/contracts/foo/contract.yaml",
            text=yaml_text,
        )
        extracted = contract.extract_handler_id()
        # handler_id extracted but lifecycle_hooks parse failed gracefully
        assert extracted.handler_id == "foo.handler"
        # lifecycle_hooks is None because parsing failed
        assert extracted.lifecycle_hooks is None

    def test_lifecycle_hooks_field_default_none(self) -> None:
        contract = ModelDiscoveredContract(
            origin="filesystem",
            ref="/app/contracts/foo/contract.yaml",
            text="handler_id: foo.handler\n",
        )
        assert contract.lifecycle_hooks is None
