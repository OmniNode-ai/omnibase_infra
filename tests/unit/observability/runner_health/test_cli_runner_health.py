# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for runner health CLI."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.observability.runner_health.cli_runner_health import main
from omnibase_infra.observability.runner_health.model_runner_health_snapshot import (
    ModelRunnerHealthSnapshot,
)


def _write_runner_fleet_config(path: Path, *, expected_count: int = 10) -> None:
    path.write_text(
        "\n".join(
            [
                'version: "1.0"',
                "github_org: OmniNode-ai",
                "runner_host: 192.168.86.201",
                "runner_group: omnibase-ci",
                "runner_name_prefix: omninode-runner",
                f"expected_count: {expected_count}",
                "",
            ]
        ),
        encoding="utf-8",
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
    return ModelRunnerHealthSnapshot(**defaults)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cli_missing_config(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI exits 1 with clear message when fleet config is missing."""
    monkeypatch.setenv("RUNNER_FLEET_CONFIG_PATH", str(tmp_path / "missing.yaml"))
    result = await main([])
    assert result == 1
    captured = capsys.readouterr()
    assert "Runner fleet config not found" in captured.out


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cli_default_summary(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default mode prints human-readable summary."""
    config_path = tmp_path / "runner_fleet.yaml"
    _write_runner_fleet_config(config_path)
    monkeypatch.setenv("RUNNER_FLEET_CONFIG_PATH", str(config_path))
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
async def test_cli_json_output(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--json flag outputs valid JSON."""
    config_path = tmp_path / "runner_fleet.yaml"
    _write_runner_fleet_config(config_path)
    monkeypatch.setenv("RUNNER_FLEET_CONFIG_PATH", str(config_path))
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
async def test_cli_alert_all_healthy(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--alert prints healthy summary when no runners are degraded."""
    config_path = tmp_path / "runner_fleet.yaml"
    _write_runner_fleet_config(config_path)
    monkeypatch.setenv("RUNNER_FLEET_CONFIG_PATH", str(config_path))
    mock_snapshot = _make_snapshot()
    with patch(
        "omnibase_infra.observability.runner_health.cli_runner_health.CollectorRunnerHealth"
    ) as mock_cls:
        mock_cls.return_value.collect = AsyncMock(return_value=mock_snapshot)
        result = await main(["--host", "192.168.86.201", "--alert"])

    assert result == 0
    captured = capsys.readouterr()
    assert "healthy" in captured.out.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cli_emit_skips_when_no_bootstrap(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--emit skips gracefully when KAFKA_BOOTSTRAP_SERVERS is not set."""
    config_path = tmp_path / "runner_fleet.yaml"
    _write_runner_fleet_config(config_path)
    monkeypatch.setenv("RUNNER_FLEET_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    mock_snapshot = _make_snapshot()
    with patch(
        "omnibase_infra.observability.runner_health.cli_runner_health.CollectorRunnerHealth"
    ) as mock_cls:
        mock_cls.return_value.collect = AsyncMock(return_value=mock_snapshot)
        result = await main(["--host", "192.168.86.201", "--emit"])

    assert result == 0
    captured = capsys.readouterr()
    assert "KAFKA_BOOTSTRAP_SERVERS not set" in captured.out


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cli_emit_publishes_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--emit calls _emit_to_kafka with the collected snapshot."""
    config_path = tmp_path / "runner_fleet.yaml"
    _write_runner_fleet_config(config_path)
    monkeypatch.setenv("RUNNER_FLEET_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    mock_snapshot = _make_snapshot()

    with (
        patch(
            "omnibase_infra.observability.runner_health.cli_runner_health.CollectorRunnerHealth"
        ) as mock_cls,
        patch(
            "omnibase_infra.observability.runner_health.cli_runner_health._emit_to_kafka",
            new=AsyncMock(),
        ) as mock_emit,
    ):
        mock_cls.return_value.collect = AsyncMock(return_value=mock_snapshot)
        result = await main(["--host", "192.168.86.201", "--emit"])

    assert result == 0
    mock_emit.assert_awaited_once_with(mock_snapshot)


@pytest.mark.unit
def test_runner_health_topic_in_all_omnibase_infra_specs() -> None:
    """SUFFIX_RUNNER_HEALTH_SNAPSHOT is registered in ALL_OMNIBASE_INFRA_TOPIC_SPECS with retention config."""
    from omnibase_infra.topics.platform_topic_suffixes import (
        ALL_OMNIBASE_INFRA_TOPIC_SPECS,
        SUFFIX_RUNNER_HEALTH_SNAPSHOT,
    )

    suffixes = {spec.suffix for spec in ALL_OMNIBASE_INFRA_TOPIC_SPECS}
    assert SUFFIX_RUNNER_HEALTH_SNAPSHOT in suffixes
    matched = next(
        spec
        for spec in ALL_OMNIBASE_INFRA_TOPIC_SPECS
        if spec.suffix == SUFFIX_RUNNER_HEALTH_SNAPSHOT
    )
    assert matched.partitions == 1
    assert matched.kafka_config is not None
    assert matched.kafka_config.get("retention.ms") == "604800000"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cli_uses_configured_runner_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI passes the fleet config count into the collector."""
    config_path = tmp_path / "runner_fleet.yaml"
    _write_runner_fleet_config(config_path, expected_count=12)
    monkeypatch.setenv("RUNNER_FLEET_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("RUNNER_HEALTH_EXPECTED_COUNT", "10")
    mock_snapshot = _make_snapshot(expected_runners=12, healthy_count=12)
    with patch(
        "omnibase_infra.observability.runner_health.cli_runner_health.CollectorRunnerHealth"
    ) as mock_cls:
        mock_cls.return_value.collect = AsyncMock(return_value=mock_snapshot)
        result = await main(["--host", "192.168.86.201"])

    assert result == 0
    mock_cls.assert_called_once_with(
        github_org="OmniNode-ai",
        runner_host="192.168.86.201",
        runner_count=12,
        runner_prefix="omninode-runner",
    )
