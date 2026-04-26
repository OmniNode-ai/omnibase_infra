# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration regression proof for OMN-9741 health-server startup ordering."""

from __future__ import annotations

from pathlib import Path

import pytest


def _read_service_kernel_source() -> str:
    return (
        Path(__file__).parents[2]
        / "src"
        / "omnibase_infra"
        / "runtime"
        / "service_kernel.py"
    ).read_text()


@pytest.mark.integration
def test_health_server_starts_before_kafka_subscription_bootstrap() -> None:
    """Kernel must bind /health before long effects Kafka subscription work."""
    source = _read_service_kernel_source()

    freeze_idx = source.index("dispatch_engine.freeze()")
    health_start_idx = source.index("await health_server.start()")
    subscribe_idx = source.index("await subscribe_wired_contract_topics(")
    plugin_consumers_idx = source.index("await plugin.start_consumers(plugin_config)")
    runtime_create_idx = source.index("runtime = RuntimeHostProcess(")
    runtime_attach_idx = source.index("health_server.attach_runtime(runtime)")

    assert freeze_idx < health_start_idx < subscribe_idx < plugin_consumers_idx
    assert subscribe_idx < runtime_create_idx < runtime_attach_idx


@pytest.mark.integration
def test_health_server_uses_runtime_pending_mode_before_attach() -> None:
    """Early ServiceHealth construction must not require RuntimeHostProcess."""
    source = _read_service_kernel_source()

    start_block = source[
        source.index("health_server = ServiceHealth(") : source.index(
            "await health_server.start()"
        )
    ]

    assert "container=container" in start_block
    assert "runtime=runtime" not in start_block
