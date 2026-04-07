# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the dashboard sweep runner used by the verify phase."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from omnibase_infra.nodes.node_verify_effect.runners.runner_dashboard_sweep import (
    DashboardSweepResult,
    run_dashboard_sweep,
)


@pytest.mark.unit
class TestRunnerDashboardSweep:
    """Tests for run_dashboard_sweep."""

    @pytest.mark.asyncio
    async def test_unreachable_returns_not_reachable(self) -> None:
        """When omnidash is down, result is advisory (reachable=False)."""
        with patch(
            "omnibase_infra.nodes.node_verify_effect.runners.runner_dashboard_sweep.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client_cls.return_value = mock_client

            result = await run_dashboard_sweep("http://localhost:9999")

        assert isinstance(result, DashboardSweepResult)
        assert result.reachable is False
        assert (
            "unreachable" in result.summary.lower()
            or "refused" in result.summary.lower()
        )

    @pytest.mark.asyncio
    async def test_server_error_returns_not_reachable(self) -> None:
        """When omnidash returns 500, result is advisory."""
        with patch(
            "omnibase_infra.nodes.node_verify_effect.runners.runner_dashboard_sweep.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_response = AsyncMock()
            mock_response.status_code = 500

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = await run_dashboard_sweep()

        assert result.reachable is False
        assert "500" in result.summary

    @pytest.mark.asyncio
    async def test_healthy_pages_with_data(self) -> None:
        """When pages return 200 with expected content, pages_with_data > 0."""
        with patch(
            "omnibase_infra.nodes.node_verify_effect.runners.runner_dashboard_sweep.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            # Body > 500 chars containing various data signals
            mock_response.text = (
                "x" * 400
                + "Summary Agent Event Intelligence Drift Pipeline Metric Settings"
                + "x" * 200
            )

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = await run_dashboard_sweep()

        assert result.reachable is True
        assert result.pages_with_data > 0
        assert len(result.pages) == 8  # 8 routes checked

    @pytest.mark.asyncio
    async def test_empty_pages_detected(self) -> None:
        """When pages return 200 but short body, pages_no_data is counted."""
        with patch(
            "omnibase_infra.nodes.node_verify_effect.runners.runner_dashboard_sweep.httpx.AsyncClient"
        ) as mock_client_cls:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.text = "<html>Loading...</html>"  # Short, no data signals

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = await run_dashboard_sweep()

        assert result.reachable is True
        assert result.pages_no_data == 8
        assert result.pages_with_data == 0
