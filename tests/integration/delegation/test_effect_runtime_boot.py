# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration proof that the remote-agent effect boots under RuntimeHostProcess."""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

from omnibase_infra.errors import InfraConnectionError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from omnibase_infra.utils import compute_consumer_group_id

REMOTE_EFFECT_CONTRACT = (
    Path("src")
    / "omnibase_infra"
    / "nodes"
    / "node_remote_agent_invoke_effect"
    / "contract.yaml"
)


def _rpk_groups() -> set[str]:
    result = subprocess.run(
        ["docker", "exec", "omnibase-infra-redpanda", "rpk", "group", "list"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "GROUP" not in line.upper()
    }


async def _wait_for_group(expected_group: str, timeout_seconds: float) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if expected_group in _rpk_groups():
            return
        await asyncio.sleep(1.0)
    msg = f"consumer group did not appear in rpk group list: {expected_group}"
    raise AssertionError(msg)


@pytest.mark.integration
@pytest.mark.kafka
@pytest.mark.asyncio
async def test_effect_boots_via_runtime_host() -> None:
    """RuntimeHostProcess package-node wiring boots the effect subscription."""
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").strip()
    if not bootstrap_servers:
        pytest.skip("KAFKA_BOOTSTRAP_SERVERS is not configured")
    if not REMOTE_EFFECT_CONTRACT.exists():
        pytest.skip(f"missing contract: {REMOTE_EFFECT_CONTRACT}")

    event_bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers=bootstrap_servers,
            environment="dev",
            timeout_seconds=30,
            max_retry_attempts=2,
            retry_backoff_base=0.5,
            circuit_breaker_threshold=5,
            circuit_breaker_reset_timeout=10.0,
        )
    )
    runtime = RuntimeHostProcess(
        event_bus=event_bus,
        input_topic="onex.cmd.omnibase-infra.delegation-request.v1",
        output_topic="onex.evt.omnibase-infra.delegation-completed.v1",
        config={
            "service_name": "omnibase-infra",
            "node_name": "runtime-host",
            "env": "dev",
            "version": "v0.1.0",
        },
    )

    expected_group = compute_consumer_group_id(
        ModelNodeIdentity(
            env="dev",
            service="omnibase-infra",
            node_name="node_remote_agent_invoke_effect",
            version="v0.1.0",
        )
    )

    try:
        try:
            await runtime.start()
        except InfraConnectionError as exc:
            pytest.skip(f"Kafka bootstrap unavailable for runtime boot test: {exc}")
        await _wait_for_group(expected_group, timeout_seconds=15.0)
        assert runtime.is_running is True
        health = await event_bus.health_check()
        assert health["consumer_count"] >= 1
        assert expected_group in _rpk_groups()
    finally:
        await runtime.stop()
