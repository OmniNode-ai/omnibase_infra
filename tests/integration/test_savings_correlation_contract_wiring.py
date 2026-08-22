# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-manifest wiring proof for node_savings_estimation_compute (OMN-16293).

Repo rule ("Runtime Startup is a First-Class CI Gate", CLAUDE.md): any PR that
touches service_kernel.py or handler-level kernel registration must include a
test that loads the real contract manifest from disk and runs
``wire_from_manifest`` with zero failures for required handlers.

This asserts the rewritten contract (node_type EFFECT_GENERIC, self-only
``savings.correlation_batch_compute`` command topic, no more auto-wired
raw-topic dead subscription — see OMN-16293) loads and wires cleanly through
the real auto-wiring path, not a synthetic contract.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.report import EnumWiringOutcome
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "src/omnibase_infra/nodes/node_savings_estimation_compute/contract.yaml"
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_savings_estimation_contract_wires_with_zero_failures() -> None:
    manifest = discover_contracts_from_paths([CONTRACT])
    assert len(manifest.contracts) == 1
    contract = manifest.contracts[0]
    assert contract.node_type == "EFFECT_GENERIC"

    engine = MessageDispatchEngine()
    mock_container = MagicMock()
    mock_container.get_service_async = AsyncMock(
        side_effect=Exception("no DI resolution needed for a wiring-only proof")
    )

    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=engine,
        event_bus=None,
        environment="test",
        container=mock_container,
        subscribe_immediately=False,
    )

    assert report.total_failed == 0, [
        (r.outcome, r.reason)
        for r in report.results
        if r.outcome != EnumWiringOutcome.WIRED
    ]
    assert report.total_wired == 1


def test_savings_estimation_contract_declares_only_self_command_topic() -> None:
    """The dead auto-wired 7-raw-topic subscription (OMN-16292/OMN-16293
    finding) must not come back. Raw signal ingestion is wired directly by
    service_kernel.py, bypassing event_bus.subscribe_topics entirely."""
    manifest = discover_contracts_from_paths([CONTRACT])
    contract = manifest.contracts[0]
    assert contract.event_bus is not None
    assert list(contract.event_bus.subscribe_topics) == [
        "onex.cmd.omnibase-infra.savings-correlation-batch-compute.v1"
    ]
