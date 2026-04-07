# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for handshake validation in auto-wiring (OMN-7657).

Covers:
    - ModelHandshakeConfig validation and defaults
    - ModelQuarantineRecord construction
    - HandshakeFailureReason classification
    - LifecycleHookExecutor.execute_handshake: success, retry, quarantine
    - Retry exhaustion and total timeout quarantine
    - Quarantine registry visibility (get_quarantined_contracts)
    - Integration with execute_startup (on_start -> handshake flow)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.context import ModelAutoWiringContext
from omnibase_infra.runtime.auto_wiring.lifecycle import (
    LifecycleHookExecutor,
    _classify_failure,
)
from omnibase_infra.runtime.auto_wiring.models import (
    HandshakeFailureReason,
    ModelHandshakeConfig,
    ModelLifecycleHookConfig,
    ModelLifecycleHookResult,
    ModelLifecycleHooks,
    ModelQuarantineRecord,
)

# ---------------------------------------------------------------------------
# ModelHandshakeConfig
# ---------------------------------------------------------------------------


class TestModelHandshakeConfig:
    """Tests for handshake configuration model."""

    def test_defaults(self) -> None:
        config = ModelHandshakeConfig()
        assert config.max_retries == 2
        assert config.retry_delay_seconds == 2.0
        assert config.total_timeout_seconds == 30.0

    def test_custom_values(self) -> None:
        config = ModelHandshakeConfig(
            max_retries=5,
            retry_delay_seconds=1.0,
            total_timeout_seconds=60.0,
        )
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 1.0
        assert config.total_timeout_seconds == 60.0

    def test_max_retries_bounds(self) -> None:
        with pytest.raises(ValueError):
            ModelHandshakeConfig(max_retries=-1)
        with pytest.raises(ValueError):
            ModelHandshakeConfig(max_retries=11)

    def test_frozen(self) -> None:
        config = ModelHandshakeConfig()
        with pytest.raises(Exception):
            config.max_retries = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ModelQuarantineRecord
# ---------------------------------------------------------------------------


class TestModelQuarantineRecord:
    """Tests for quarantine record model."""

    def test_construction(self) -> None:
        record = ModelQuarantineRecord(
            handler_id="test.handler",
            node_kind="COMPUTE",
            failure_reason=HandshakeFailureReason.TCP_PROBE_FAILED,
            error_message="Connection refused on port 5432",
            attempts=3,
        )
        assert record.handler_id == "test.handler"
        assert record.failure_reason == HandshakeFailureReason.TCP_PROBE_FAILED
        assert record.attempts == 3
        assert record.quarantined_at is not None

    def test_frozen(self) -> None:
        record = ModelQuarantineRecord(
            handler_id="test.handler",
            node_kind="COMPUTE",
            failure_reason=HandshakeFailureReason.TIMEOUT,
            attempts=1,
        )
        with pytest.raises(Exception):
            record.handler_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HandshakeFailureReason
# ---------------------------------------------------------------------------


class TestHandshakeFailureReason:
    """Tests for failure reason enum values."""

    def test_all_values(self) -> None:
        expected = {
            "timeout",
            "resolution_failed",
            "db_ownership",
            "schema_fingerprint",
            "tcp_probe_failed",
            "hook_exception",
            "hook_returned_failure",
        }
        assert {r.value for r in HandshakeFailureReason} == expected


# ---------------------------------------------------------------------------
# _classify_failure
# ---------------------------------------------------------------------------


class TestClassifyFailure:
    """Tests for failure classification helper."""

    def test_timeout(self) -> None:
        result = ModelLifecycleHookResult.failed(
            "validate_handshake", "Hook 'pkg.check' timed out after 10s"
        )
        assert _classify_failure(result) == HandshakeFailureReason.TIMEOUT

    def test_resolution_failed(self) -> None:
        result = ModelLifecycleHookResult.failed(
            "validate_handshake", "Hook resolution failed: No module named 'pkg'"
        )
        assert _classify_failure(result) == HandshakeFailureReason.RESOLUTION_FAILED

    def test_hook_exception(self) -> None:
        result = ModelLifecycleHookResult.failed(
            "validate_handshake", "Hook 'pkg.check' raised: ConnectionError"
        )
        assert _classify_failure(result) == HandshakeFailureReason.HOOK_EXCEPTION

    def test_hook_returned_failure(self) -> None:
        result = ModelLifecycleHookResult.failed(
            "validate_handshake", "Schema mismatch detected"
        )
        assert _classify_failure(result) == HandshakeFailureReason.HOOK_RETURNED_FAILURE


# ---------------------------------------------------------------------------
# ModelLifecycleHooks with handshake_config
# ---------------------------------------------------------------------------


class TestLifecycleHooksHandshakeConfig:
    """Tests for handshake_config on ModelLifecycleHooks."""

    def test_default_handshake_config(self) -> None:
        hooks = ModelLifecycleHooks()
        assert hooks.handshake_config.max_retries == 2

    def test_custom_handshake_config(self) -> None:
        hooks = ModelLifecycleHooks(
            validate_handshake=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.validate",
            ),
            handshake_config=ModelHandshakeConfig(max_retries=5),
        )
        assert hooks.handshake_config.max_retries == 5


# ---------------------------------------------------------------------------
# LifecycleHookExecutor.execute_handshake
# ---------------------------------------------------------------------------


class TestExecuteHandshake:
    """Tests for handshake execution with retry and quarantine."""

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
    async def test_handshake_success_first_attempt(
        self, executor: LifecycleHookExecutor, base_context_kwargs: dict[str, str]
    ) -> None:
        hooks = ModelLifecycleHooks(
            validate_handshake=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.validate",
            ),
            handshake_config=ModelHandshakeConfig(max_retries=2),
        )

        mock_fn = AsyncMock(
            return_value=ModelLifecycleHookResult.succeeded("validate_handshake")
        )

        with patch(
            "omnibase_infra.runtime.auto_wiring.lifecycle.resolve_hook_callable",
            return_value=mock_fn,
        ):
            result = await executor.execute_handshake(hooks, base_context_kwargs)

        assert result is not None
        assert result.success
        assert executor.get_quarantined_contracts() == []
        assert mock_fn.await_count == 1

    @pytest.mark.asyncio
    async def test_handshake_no_hook_configured(
        self, executor: LifecycleHookExecutor, base_context_kwargs: dict[str, str]
    ) -> None:
        hooks = ModelLifecycleHooks()
        result = await executor.execute_handshake(hooks, base_context_kwargs)
        assert result is None

    @pytest.mark.asyncio
    async def test_handshake_success_on_retry(
        self, executor: LifecycleHookExecutor, base_context_kwargs: dict[str, str]
    ) -> None:
        hooks = ModelLifecycleHooks(
            validate_handshake=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.validate",
            ),
            handshake_config=ModelHandshakeConfig(
                max_retries=2,
                retry_delay_seconds=0.01,
            ),
        )

        mock_fn = AsyncMock(
            side_effect=[
                ModelLifecycleHookResult.failed("validate_handshake", "Not ready"),
                ModelLifecycleHookResult.succeeded("validate_handshake"),
            ]
        )

        with patch(
            "omnibase_infra.runtime.auto_wiring.lifecycle.resolve_hook_callable",
            return_value=mock_fn,
        ):
            result = await executor.execute_handshake(hooks, base_context_kwargs)

        assert result is not None
        assert result.success
        assert executor.get_quarantined_contracts() == []
        assert mock_fn.await_count == 2

    @pytest.mark.asyncio
    async def test_handshake_quarantine_after_retries_exhausted(
        self, executor: LifecycleHookExecutor, base_context_kwargs: dict[str, str]
    ) -> None:
        hooks = ModelLifecycleHooks(
            validate_handshake=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.validate",
            ),
            handshake_config=ModelHandshakeConfig(
                max_retries=1,
                retry_delay_seconds=0.01,
            ),
        )

        mock_fn = AsyncMock(
            return_value=ModelLifecycleHookResult.failed(
                "validate_handshake", "DB ownership check failed"
            )
        )

        with patch(
            "omnibase_infra.runtime.auto_wiring.lifecycle.resolve_hook_callable",
            return_value=mock_fn,
        ):
            result = await executor.execute_handshake(hooks, base_context_kwargs)

        assert result is not None
        assert not result.success
        quarantined = executor.get_quarantined_contracts()
        assert len(quarantined) == 1
        assert quarantined[0].handler_id == "test.handler"
        assert (
            quarantined[0].failure_reason
            == HandshakeFailureReason.HOOK_RETURNED_FAILURE
        )
        assert quarantined[0].attempts == 2  # 1 initial + 1 retry
        # 2 attempts: initial + 1 retry
        assert mock_fn.await_count == 2

    @pytest.mark.asyncio
    async def test_handshake_quarantine_on_total_timeout(
        self, executor: LifecycleHookExecutor, base_context_kwargs: dict[str, str]
    ) -> None:
        hooks = ModelLifecycleHooks(
            validate_handshake=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.validate",
                timeout_seconds=10.0,
            ),
            handshake_config=ModelHandshakeConfig(
                max_retries=5,
                retry_delay_seconds=0.01,
                total_timeout_seconds=1.0,
            ),
        )

        async def slow_hook(_ctx: ModelAutoWiringContext) -> ModelLifecycleHookResult:
            await asyncio.sleep(10)
            return ModelLifecycleHookResult.succeeded("validate_handshake")

        with patch(
            "omnibase_infra.runtime.auto_wiring.lifecycle.resolve_hook_callable",
            return_value=slow_hook,
        ):
            result = await executor.execute_handshake(hooks, base_context_kwargs)

        assert result is not None
        assert not result.success
        assert "total timeout" in result.error_message.lower()
        quarantined = executor.get_quarantined_contracts()
        assert len(quarantined) == 1
        assert quarantined[0].failure_reason == HandshakeFailureReason.TIMEOUT

    @pytest.mark.asyncio
    async def test_handshake_quarantine_timeout_classification(
        self, executor: LifecycleHookExecutor, base_context_kwargs: dict[str, str]
    ) -> None:
        """Per-attempt timeout should classify as TIMEOUT."""
        hooks = ModelLifecycleHooks(
            validate_handshake=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.validate",
                timeout_seconds=1.0,
            ),
            handshake_config=ModelHandshakeConfig(
                max_retries=0,
                retry_delay_seconds=0.0,
                total_timeout_seconds=30.0,
            ),
        )

        async def slow_hook(_ctx: ModelAutoWiringContext) -> ModelLifecycleHookResult:
            await asyncio.sleep(10)
            return ModelLifecycleHookResult.succeeded("validate_handshake")

        with patch(
            "omnibase_infra.runtime.auto_wiring.lifecycle.resolve_hook_callable",
            return_value=slow_hook,
        ):
            result = await executor.execute_handshake(hooks, base_context_kwargs)

        assert result is not None
        assert not result.success
        quarantined = executor.get_quarantined_contracts()
        assert len(quarantined) == 1
        assert quarantined[0].failure_reason == HandshakeFailureReason.TIMEOUT

    @pytest.mark.asyncio
    async def test_quarantine_registry_accumulates(
        self, executor: LifecycleHookExecutor
    ) -> None:
        """Multiple failed handshakes accumulate in the quarantine registry."""
        hooks = ModelLifecycleHooks(
            validate_handshake=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.validate",
            ),
            handshake_config=ModelHandshakeConfig(
                max_retries=0,
                retry_delay_seconds=0.0,
            ),
        )

        mock_fn = AsyncMock(
            return_value=ModelLifecycleHookResult.failed(
                "validate_handshake", "Not ready"
            )
        )

        with patch(
            "omnibase_infra.runtime.auto_wiring.lifecycle.resolve_hook_callable",
            return_value=mock_fn,
        ):
            await executor.execute_handshake(
                hooks, {"handler_id": "handler.a", "node_kind": "COMPUTE"}
            )
            await executor.execute_handshake(
                hooks, {"handler_id": "handler.b", "node_kind": "EFFECT"}
            )

        quarantined = executor.get_quarantined_contracts()
        assert len(quarantined) == 2
        assert quarantined[0].handler_id == "handler.a"
        assert quarantined[1].handler_id == "handler.b"

    @pytest.mark.asyncio
    async def test_quarantine_registry_returns_copy(
        self, executor: LifecycleHookExecutor
    ) -> None:
        """get_quarantined_contracts returns a copy, not the internal list."""
        q1 = executor.get_quarantined_contracts()
        q2 = executor.get_quarantined_contracts()
        assert q1 is not q2


# ---------------------------------------------------------------------------
# execute_startup integration with handshake
# ---------------------------------------------------------------------------


class TestExecuteStartupWithHandshake:
    """Tests for the full startup flow including handshake."""

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
    async def test_startup_on_start_then_handshake(
        self, executor: LifecycleHookExecutor, base_context_kwargs: dict[str, str]
    ) -> None:
        hooks = ModelLifecycleHooks(
            on_start=ModelLifecycleHookConfig(callable_ref="pkg.hooks.start"),
            validate_handshake=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.validate"
            ),
            handshake_config=ModelHandshakeConfig(max_retries=0),
        )

        call_order: list[str] = []

        async def mock_start(ctx: ModelAutoWiringContext) -> ModelLifecycleHookResult:
            call_order.append("on_start")
            return ModelLifecycleHookResult.succeeded("on_start")

        async def mock_validate(
            ctx: ModelAutoWiringContext,
        ) -> ModelLifecycleHookResult:
            call_order.append("validate_handshake")
            return ModelLifecycleHookResult.succeeded("validate_handshake")

        call_count = 0

        def mock_resolve(ref: str) -> AsyncMock:
            nonlocal call_count
            call_count += 1
            if "start" in ref:
                return mock_start  # type: ignore[return-value]
            return mock_validate  # type: ignore[return-value]

        with patch(
            "omnibase_infra.runtime.auto_wiring.lifecycle.resolve_hook_callable",
            side_effect=mock_resolve,
        ):
            results = await executor.execute_startup(hooks, base_context_kwargs)

        assert len(results) == 2
        assert all(r.success for r in results)
        assert call_order == ["on_start", "validate_handshake"]

    @pytest.mark.asyncio
    async def test_startup_on_start_failure_skips_handshake(
        self, executor: LifecycleHookExecutor, base_context_kwargs: dict[str, str]
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
            return_value=ModelLifecycleHookResult.failed("on_start", "Startup failed")
        )

        with patch(
            "omnibase_infra.runtime.auto_wiring.lifecycle.resolve_hook_callable",
            return_value=mock_fn,
        ):
            results = await executor.execute_startup(hooks, base_context_kwargs)

        assert len(results) == 1
        assert not results[0].success
        assert results[0].hook_name == "on_start"
        # Handshake was never called, so no quarantine
        assert executor.get_quarantined_contracts() == []

    @pytest.mark.asyncio
    async def test_startup_optional_on_start_failure_continues_to_handshake(
        self, executor: LifecycleHookExecutor, base_context_kwargs: dict[str, str]
    ) -> None:
        hooks = ModelLifecycleHooks(
            on_start=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.start",
                required=False,
            ),
            validate_handshake=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.validate"
            ),
            handshake_config=ModelHandshakeConfig(max_retries=0),
        )

        mock_fn = AsyncMock(
            side_effect=[
                ModelLifecycleHookResult.failed("on_start", "Non-critical"),
                ModelLifecycleHookResult.succeeded("validate_handshake"),
            ]
        )

        with patch(
            "omnibase_infra.runtime.auto_wiring.lifecycle.resolve_hook_callable",
            return_value=mock_fn,
        ):
            results = await executor.execute_startup(hooks, base_context_kwargs)

        assert len(results) == 2
        assert not results[0].success  # on_start failed (optional)
        assert results[1].success  # handshake succeeded

    @pytest.mark.asyncio
    async def test_startup_no_hooks(
        self, executor: LifecycleHookExecutor, base_context_kwargs: dict[str, str]
    ) -> None:
        hooks = ModelLifecycleHooks()
        results = await executor.execute_startup(hooks, base_context_kwargs)
        assert results == []

    @pytest.mark.asyncio
    async def test_startup_handshake_failure_quarantines(
        self, executor: LifecycleHookExecutor, base_context_kwargs: dict[str, str]
    ) -> None:
        hooks = ModelLifecycleHooks(
            validate_handshake=ModelLifecycleHookConfig(
                callable_ref="pkg.hooks.validate"
            ),
            handshake_config=ModelHandshakeConfig(
                max_retries=0,
                retry_delay_seconds=0.0,
            ),
        )

        mock_fn = AsyncMock(
            return_value=ModelLifecycleHookResult.failed(
                "validate_handshake", "Schema mismatch"
            )
        )

        with patch(
            "omnibase_infra.runtime.auto_wiring.lifecycle.resolve_hook_callable",
            return_value=mock_fn,
        ):
            results = await executor.execute_startup(hooks, base_context_kwargs)

        assert len(results) == 1
        assert not results[0].success
        assert len(executor.get_quarantined_contracts()) == 1
