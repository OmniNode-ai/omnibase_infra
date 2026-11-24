# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-startup gate for the savings estimation node (OMN-26418).

``omnibase_infra/CLAUDE.md`` ("Runtime Startup is a First-Class CI Gate") asks
every PR that touches ``service_kernel.py`` for a test that loads the real
contract from disk, runs ``wire_from_manifest`` the way the kernel does, and
asserts zero failures. This one covers the node whose consumers the kernel's
savings step starts. The compose-sandbox half of the gate (healthy inside the
compose ``start_period``, ``RestartCount == 0``) runs in CI on the worker
container, not here.

Ticket: OMN-26418
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    wire_from_manifest,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = [pytest.mark.integration]

_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_savings_estimation_compute/contract.yaml"
)


def test_the_savings_contract_exists_and_is_discoverable() -> None:
    assert _CONTRACT.exists(), f"missing {_CONTRACT}"
    manifest = discover_contracts_from_paths([_CONTRACT])
    assert manifest.total_errors == 0, f"contract parse errors: {manifest.errors!r}"
    assert {c.name for c in manifest.contracts} == {"node_savings_estimation_compute"}


@pytest.mark.asyncio
async def test_the_savings_node_wires_from_the_manifest_with_zero_failures() -> None:
    manifest = discover_contracts_from_paths([_CONTRACT])
    assert manifest.total_errors == 0
    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=MessageDispatchEngine(logger=MagicMock()),
        event_bus=None,  # no live Kafka in this gate; auto-wiring is in-process
        environment="test",
        container=MagicMock(),
        result_appliers_by_contract={},
    )
    assert report.total_failed == 0, (
        f"wiring failures: {[r for r in report.results if r.outcome != 'wired']!r}"
    )
    assert len(report.results) == 1
