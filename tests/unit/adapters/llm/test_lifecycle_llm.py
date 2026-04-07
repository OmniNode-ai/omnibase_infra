# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for LLM lifecycle hooks (OMN-7662).

Proves that:
1. on_start succeeds with no LLM endpoints configured (graceful skip).
2. on_start creates health probes when endpoints are configured.
3. on_shutdown cleans up state.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from omnibase_infra.adapters.llm import lifecycle_llm
from omnibase_infra.runtime.auto_wiring.context import ModelAutoWiringContext


def _make_context(**kwargs: object) -> ModelAutoWiringContext:
    defaults = {
        "handler_id": "node_llm_inference_effect",
        "node_kind": "EFFECT_GENERIC",
        "phase": "on_start",
    }
    defaults.update(kwargs)
    return ModelAutoWiringContext(**defaults)


class TestLlmLifecycleOnStart:
    """Tests for lifecycle_llm.on_start."""

    @pytest.fixture(autouse=True)
    def _reset_state(self) -> None:
        lifecycle_llm._state.clear()

    @pytest.mark.asyncio
    async def test_no_endpoints_succeeds_with_empty_workers(self) -> None:
        """When no LLM_*_URL env vars are set, on_start succeeds gracefully."""
        ctx = _make_context()
        with patch.dict("os.environ", {}, clear=True):
            result = await lifecycle_llm.on_start(ctx)
        assert result.success is True
        assert result.background_workers == []
        assert "router" not in lifecycle_llm._state

    @pytest.mark.asyncio
    async def test_with_endpoints_creates_router_and_health(self) -> None:
        """When LLM endpoints are configured, on_start creates health probes."""
        ctx = _make_context()

        mock_health = AsyncMock()
        mock_health.start = AsyncMock()
        mock_health.stop = AsyncMock()

        env = {"LLM_CODER_URL": "http://localhost:8000/v1"}

        with (
            patch.dict("os.environ", env, clear=True),
            patch(
                "omnibase_infra.services.service_llm_endpoint_health.ServiceLlmEndpointHealth",
                return_value=mock_health,
            ),
        ):
            result = await lifecycle_llm.on_start(ctx)

        assert result.success is True
        assert "llm-health-probe" in result.background_workers
        assert "router" in lifecycle_llm._state
        assert "health_service" in lifecycle_llm._state
        mock_health.start.assert_awaited_once()


class TestLlmLifecycleOnShutdown:
    """Tests for lifecycle_llm.on_shutdown."""

    @pytest.fixture(autouse=True)
    def _reset_state(self) -> None:
        lifecycle_llm._state.clear()

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up_state(self) -> None:
        """on_shutdown stops health service and clears state."""
        mock_health = AsyncMock()
        mock_health.stop = AsyncMock()
        lifecycle_llm._state["health_service"] = mock_health
        lifecycle_llm._state["router"] = "mock_router"
        lifecycle_llm._state["endpoints"] = {}

        ctx = _make_context(phase="on_shutdown")
        result = await lifecycle_llm.on_shutdown(ctx)

        assert result.success is True
        assert len(lifecycle_llm._state) == 0
        mock_health.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_empty_state_succeeds(self) -> None:
        """on_shutdown succeeds even when nothing was started."""
        ctx = _make_context(phase="on_shutdown")
        result = await lifecycle_llm.on_shutdown(ctx)
        assert result.success is True
