# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for the lab runtime log bridge transport (OMN-19718)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from omnibase_infra.runtime.service_kernel import (
    _create_runtime_log_bridge_producer,
    _runtime_log_bridge_allowlist,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INFRA_COMPOSE = _REPO_ROOT / "docker" / "docker-compose.infra.yml"


@pytest.mark.asyncio
async def test_bridge_producer_uses_runtime_sasl_transport() -> None:
    """The dedicated producer authenticates exactly like the runtime event bus."""
    producer = AsyncMock()
    producer_type = MagicMock(return_value=producer)
    sasl_env = {
        "KAFKA_SECURITY_PROTOCOL": "SASL_PLAINTEXT",
        "KAFKA_SASL_MECHANISM": "SCRAM-SHA-256",
        "KAFKA_SASL_USERNAME": "runtime-log-bridge",
        "KAFKA_SASL_PASSWORD": "test-password",
    }

    with (
        patch.dict("os.environ", sasl_env, clear=True),
        patch("aiokafka.AIOKafkaProducer", producer_type),
    ):
        result = await _create_runtime_log_bridge_producer("redpanda:9092")

    assert result is producer
    producer_type.assert_called_once_with(
        bootstrap_servers="redpanda:9092",
        security_protocol="SASL_PLAINTEXT",
        sasl_mechanism="SCRAM-SHA-256",
        sasl_plain_username="runtime-log-bridge",
        sasl_plain_password="test-password",
    )
    producer.start.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_bridge_producer_defaults_to_plaintext_without_sasl_env() -> None:
    """A lane without auth settings keeps aiokafka's PLAINTEXT default."""
    producer = AsyncMock()
    producer_type = MagicMock(return_value=producer)

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("aiokafka.AIOKafkaProducer", producer_type),
    ):
        result = await _create_runtime_log_bridge_producer("redpanda:9092")

    assert result is producer
    producer_type.assert_called_once_with(bootstrap_servers="redpanda:9092")
    producer.start.assert_awaited_once_with()


def test_default_allowlist_includes_runtime_auto_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Projection and handler wiring failures are captured by default."""
    monkeypatch.delenv("RUNTIME_LOG_BRIDGE_ALLOWLIST", raising=False)

    assert "omnibase_infra.runtime.auto_wiring" in _runtime_log_bridge_allowlist()


def test_dev_lane_runtime_services_enable_log_bridge() -> None:
    """The dev-lane base enables the bridge on both catalog-equivalent runtimes."""
    compose = yaml.safe_load(_INFRA_COMPOSE.read_text(encoding="utf-8"))

    for service_name in ("omninode-runtime", "runtime-effects"):
        environment = compose["services"][service_name]["environment"]
        assert environment["ENABLE_RUNTIME_LOG_BRIDGE"] == "true"
