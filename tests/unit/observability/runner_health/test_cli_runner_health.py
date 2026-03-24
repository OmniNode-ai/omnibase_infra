# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for runner health CLI."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.observability.runner_health.cli_runner_health import main
from omnibase_infra.observability.runner_health.model_runner_health_snapshot import (
    ModelRunnerHealthSnapshot,
)


def _make_snapshot(**overrides: object) -> ModelRunnerHealthSnapshot:
    """Create a snapshot with sensible defaults."""
    defaults = {
        "correlation_id": uuid4(),
        "collected_at": datetime.now(tz=UTC),
        "runners": (),
        "expected_runners": 10,
        "observed_runners": 0,
        "healthy_count": 10,
        "degraded_count": 0,
        "host": "192.168.86.201",
        "host_disk_percent": 25.0,
    }
    defaults.update(overrides)
    return ModelRunnerHealthSnapshot(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cli_missing_host(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI exits 1 with clear message when host is not set."""
    with patch(
        "omnibase_infra.observability.runner_health.cli_runner_health.RUNNER_HOST", ""
    ):
        result = await main([])
    assert result == 1
    captured = capsys.readouterr()
    assert "RUNNER_HEALTH_HOST" in captured.out


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cli_default_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """Default mode prints human-readable summary."""
    mock_snapshot = _make_snapshot()
    with patch(
        "omnibase_infra.observability.runner_health.cli_runner_health.CollectorRunnerHealth"
    ) as mock_cls:
        mock_cls.return_value.collect = AsyncMock(return_value=mock_snapshot)
        result = await main(["--host", "192.168.86.201"])

    assert result == 0
    captured = capsys.readouterr()
    assert "Runner Health:" in captured.out


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cli_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    """--json flag outputs valid JSON."""
    mock_snapshot = _make_snapshot()
    with patch(
        "omnibase_infra.observability.runner_health.cli_runner_health.CollectorRunnerHealth"
    ) as mock_cls:
        mock_cls.return_value.collect = AsyncMock(return_value=mock_snapshot)
        result = await main(["--host", "192.168.86.201", "--json"])

    assert result == 0
    captured = capsys.readouterr()
    # Verify JSON round-trips
    parsed = ModelRunnerHealthSnapshot.model_validate_json(captured.out)
    assert parsed.host == "192.168.86.201"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cli_alert_all_healthy(capsys: pytest.CaptureFixture[str]) -> None:
    """--alert prints healthy summary when no runners are degraded."""
    mock_snapshot = _make_snapshot()
    with patch(
        "omnibase_infra.observability.runner_health.cli_runner_health.CollectorRunnerHealth"
    ) as mock_cls:
        mock_cls.return_value.collect = AsyncMock(return_value=mock_snapshot)
        result = await main(["--host", "192.168.86.201", "--alert"])

    assert result == 0
    captured = capsys.readouterr()
    assert "healthy" in captured.out.lower()
