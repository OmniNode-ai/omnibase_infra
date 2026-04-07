# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for DLQ lifecycle hooks (OMN-7662).

Proves that:
1. on_start succeeds when DLQ is not enabled (graceful skip).
2. on_start succeeds when DLQ is enabled but DB URL is missing.
3. on_shutdown cleans up state.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from omnibase_infra.dlq import lifecycle_dlq
from omnibase_infra.runtime.auto_wiring.context import ModelAutoWiringContext


def _make_context(**kwargs: object) -> ModelAutoWiringContext:
    defaults = {
        "handler_id": "service_dlq_retry_worker",
        "node_kind": "EFFECT_GENERIC",
        "phase": "on_start",
    }
    defaults.update(kwargs)
    return ModelAutoWiringContext(**defaults)


class TestDlqLifecycleOnStart:
    """Tests for lifecycle_dlq.on_start."""

    @pytest.fixture(autouse=True)
    def _reset_state(self) -> None:
        lifecycle_dlq._state.clear()

    @pytest.mark.asyncio
    async def test_not_enabled_succeeds_gracefully(self) -> None:
        """When OMNIBASE_INFRA_DLQ_ENABLED is not set, on_start succeeds."""
        ctx = _make_context()
        with patch.dict("os.environ", {}, clear=True):
            result = await lifecycle_dlq.on_start(ctx)
        assert result.success is True
        assert result.background_workers == []
        assert len(lifecycle_dlq._state) == 0

    @pytest.mark.asyncio
    async def test_enabled_no_db_url_succeeds_gracefully(self) -> None:
        """When DLQ enabled but no DB URL, on_start succeeds gracefully."""
        ctx = _make_context()
        env = {"OMNIBASE_INFRA_DLQ_ENABLED": "true"}
        with patch.dict("os.environ", env, clear=True):
            result = await lifecycle_dlq.on_start(ctx)
        assert result.success is True
        assert result.background_workers == []


class TestDlqLifecycleOnShutdown:
    """Tests for lifecycle_dlq.on_shutdown."""

    @pytest.fixture(autouse=True)
    def _reset_state(self) -> None:
        lifecycle_dlq._state.clear()

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up_state(self) -> None:
        """on_shutdown stops workers and clears state."""
        mock_pool = AsyncMock()
        mock_pool.close = AsyncMock()
        mock_tracking = AsyncMock()
        mock_tracking.shutdown = AsyncMock()
        mock_worker = AsyncMock()
        mock_worker.stop = AsyncMock()

        lifecycle_dlq._state["pool"] = mock_pool
        lifecycle_dlq._state["dlq_tracking"] = mock_tracking
        lifecycle_dlq._state["retry_worker"] = mock_worker
        lifecycle_dlq._state["dsn"] = "postgres://..."

        ctx = _make_context(phase="on_shutdown")
        result = await lifecycle_dlq.on_shutdown(ctx)

        assert result.success is True
        assert len(lifecycle_dlq._state) == 0
        mock_worker.stop.assert_awaited_once()
        mock_tracking.shutdown.assert_awaited_once()
        mock_pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_empty_state_succeeds(self) -> None:
        """on_shutdown succeeds even when nothing was started."""
        ctx = _make_context(phase="on_shutdown")
        result = await lifecycle_dlq.on_shutdown(ctx)
        assert result.success is True
