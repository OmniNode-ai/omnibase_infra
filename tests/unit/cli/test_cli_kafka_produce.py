# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for onex kafka produce CLI command (OMN-8435)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_kafka import kafka

pytestmark = pytest.mark.unit


class TestKafkaProduceDryRun:
    """Tests for --dry-run mode — no Kafka connection required."""

    def test_dry_run_prints_topic_and_payload(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            kafka,
            [
                "produce",
                "onex.cmd.test.v1",
                "--payload",
                '{"key": "value"}',
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "onex.cmd.test.v1" in result.output
        assert "key" in result.output

    def test_dry_run_with_envelope_wraps_payload(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            kafka,
            [
                "produce",
                "onex.cmd.deploy.rebuild-requested.v1",
                "--payload",
                '{"scope": "full", "services": []}',
                "--dry-run",
                "--envelope",
            ],
        )
        assert result.exit_code == 0, result.output
        output_json = _extract_json(result.output)
        assert output_json is not None, f"No JSON in output: {result.output}"
        assert "correlation_id" in output_json
        assert "requested_by" in output_json
        assert "timestamp" in output_json
        assert output_json.get("scope") == "full"

    def test_dry_run_indicates_skipping_publish(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            kafka,
            ["produce", "onex.cmd.test.v1", "--payload", "{}", "--dry-run"],
        )
        assert result.exit_code == 0
        assert "dry-run" in result.output.lower() or "skipping" in result.output.lower()

    def test_invalid_json_payload_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            kafka,
            ["produce", "onex.cmd.test.v1", "--payload", "not-json", "--dry-run"],
        )
        assert result.exit_code != 0

    def test_help_shows_produce_subcommand(self) -> None:
        runner = CliRunner()
        result = runner.invoke(kafka, ["--help"])
        assert result.exit_code == 0
        assert "produce" in result.output

    def test_envelope_uses_custom_requested_by(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            kafka,
            [
                "produce",
                "onex.cmd.test.v1",
                "--payload",
                "{}",
                "--dry-run",
                "--envelope",
                "--requested-by",
                "my-agent",
            ],
        )
        assert result.exit_code == 0
        output_json = _extract_json(result.output)
        assert output_json is not None
        assert output_json.get("requested_by") == "my-agent"


class TestKafkaProducePublish:
    """Tests for actual publish path — Kafka producer mocked."""

    def test_produce_calls_publisher_with_correct_topic(self) -> None:
        published: list[dict[str, object]] = []

        async def mock_send(topic: str, value: bytes) -> None:
            published.append({"topic": topic, "payload": json.loads(value)})

        mock_producer = MagicMock()
        mock_producer.start = AsyncMock()
        mock_producer.stop = AsyncMock()
        mock_producer.send_and_wait = AsyncMock(side_effect=mock_send)

        with patch(
            "omnibase_infra.cli.cli_kafka.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            runner = CliRunner()
            result = runner.invoke(
                kafka,
                ["produce", "onex.cmd.test.v1", "--payload", '{"scope": "full"}'],
                env={"KAFKA_BOOTSTRAP_SERVERS": "localhost:19092"},
            )

        assert result.exit_code == 0, result.output
        assert len(published) == 1
        assert published[0]["topic"] == "onex.cmd.test.v1"

    def test_envelope_flag_produces_valid_envelope_shape(self) -> None:
        published: list[dict[str, object]] = []

        async def mock_send(topic: str, value: bytes) -> None:
            published.append({"topic": topic, "payload": json.loads(value)})

        mock_producer = MagicMock()
        mock_producer.start = AsyncMock()
        mock_producer.stop = AsyncMock()
        mock_producer.send_and_wait = AsyncMock(side_effect=mock_send)

        with patch(
            "omnibase_infra.cli.cli_kafka.AIOKafkaProducer",
            return_value=mock_producer,
        ):
            runner = CliRunner()
            result = runner.invoke(
                kafka,
                [
                    "produce",
                    "onex.cmd.deploy.rebuild-requested.v1",
                    "--payload",
                    '{"scope": "full", "services": [], "git_ref": "origin/main"}',
                    "--envelope",
                    "--requested-by",
                    "test-agent",
                ],
                env={"KAFKA_BOOTSTRAP_SERVERS": "localhost:19092"},
            )

        assert result.exit_code == 0, result.output
        assert len(published) == 1
        env = published[0]["payload"]
        assert "correlation_id" in env
        assert env.get("requested_by") == "test-agent"
        assert "timestamp" in env
        assert env.get("scope") == "full"

    def test_missing_bootstrap_servers_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            kafka,
            ["produce", "onex.cmd.test.v1", "--payload", "{}"],
            env={"KAFKA_BOOTSTRAP_SERVERS": ""},
        )
        assert result.exit_code != 0


def _extract_json(output: str) -> dict[str, object] | None:
    brace_pos = output.find("{")
    if brace_pos == -1:
        return None
    candidate = output[brace_pos:]
    for end in range(len(candidate), 0, -1):
        try:
            result = json.loads(candidate[:end])
            if isinstance(result, dict):
                return result  # type: ignore[return-value]
        except json.JSONDecodeError:
            continue
    return None
