# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for runtime health-server startup ordering."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_health_server_starts_before_kafka_subscription_bootstrap() -> None:
    """Kernel must bind /health before long effects Kafka subscription work."""
    source = (
        Path(__file__).parents[3]
        / "src"
        / "omnibase_infra"
        / "runtime"
        / "service_kernel.py"
    ).read_text()

    freeze_idx = source.index("MessageDispatchEngine frozen after all")
    health_start_idx = source.index("await health_server.start()")
    subscribe_idx = source.index("await subscribe_wired_contract_topics(")
    plugin_consumers_idx = source.index("await plugin.start_consumers(plugin_config)")
    runtime_create_idx = source.index("runtime = RuntimeHostProcess(")
    runtime_attach_idx = source.index("health_server.attach_runtime(runtime)")

    assert freeze_idx < health_start_idx < subscribe_idx < plugin_consumers_idx
    assert subscribe_idx < runtime_create_idx < runtime_attach_idx


@pytest.mark.unit
def test_health_server_uses_runtime_pending_mode_before_attach() -> None:
    """Early ServiceHealth construction must not require RuntimeHostProcess."""
    source = (
        Path(__file__).parents[3]
        / "src"
        / "omnibase_infra"
        / "runtime"
        / "service_kernel.py"
    ).read_text()

    start_block = source[
        source.index("health_server = ServiceHealth(") : source.index(
            "await health_server.start()"
        )
    ]

    assert "container=container" in start_block
    assert "runtime=runtime" not in start_block
