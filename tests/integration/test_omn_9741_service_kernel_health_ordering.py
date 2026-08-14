# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration regression proof for OMN-9741/OMN-13768 health-server startup ordering."""

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
    """Kernel must bind /health before long effects Kafka subscription work.

    OMN-13768: RuntimeHostProcess is now created, registered, and attached to
    the health server immediately after ``health_server.start()`` and BEFORE
    ``subscribe_wired_contract_topics``/plugin consumer start (previously the
    opposite order, which left ``ServiceHealth._runtime`` as ``None`` for the
    entire — potentially 10+ minute, per this module's docstring — Kafka
    subscription window, so ``GET /ready`` raised
    ``ProtocolConfigurationError [ONEX_CORE_041]`` the whole time even though
    the runtime was already processing events via the auto-wired consumers).
    """
    source = _read_service_kernel_source()

    freeze_idx = source.index("dispatch_engine.freeze()")
    health_start_idx = source.index("await health_server.start()")
    runtime_create_idx = source.index("runtime = RuntimeHostProcess(")
    runtime_attach_idx = source.index("health_server.attach_runtime(runtime)")
    subscribe_idx = source.index("await subscribe_wired_contract_topics(")
    plugin_consumers_idx = source.index("await plugin.start_consumers(plugin_config)")

    assert (
        freeze_idx
        < health_start_idx
        < runtime_create_idx
        < runtime_attach_idx
        < subscribe_idx
        < plugin_consumers_idx
    )


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


@pytest.mark.integration
def test_failed_plugin_consumer_result_is_not_reported_as_started() -> None:
    """A returned failure must branch before the consumer-started log."""
    source = _read_service_kernel_source()
    pass_two_start = source.index(
        "# --- Pass 2: Start consumers for ready plugins only ---"
    )
    pass_two = source[
        pass_two_start : source.index("plugin_activation_duration =", pass_two_start)
    ]

    result_idx = pass_two.index("consumer_result = await plugin.start_consumers")
    failure_idx = pass_two.index("if not consumer_result.success:")
    continue_idx = pass_two.index("continue", failure_idx)
    started_idx = pass_two.index("consumers started", continue_idx)

    assert result_idx < failure_idx < continue_idx < started_idx
