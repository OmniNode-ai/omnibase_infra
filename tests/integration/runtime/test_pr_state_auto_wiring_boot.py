# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""OMN-14375 first-class CI gate for the pr_state projection chain.

This test loads the real contract.yaml files for the two new nodes from disk,
runs `wire_from_manifest` exactly the way the kernel does in production, and
asserts both nodes wire successfully with zero failures.

Required by `omnibase_infra/CLAUDE.md` § "Runtime Startup is a First-Class CI
Gate" for any PR touching auto_wiring/, service_kernel.py, handler __init__
signatures, or kernel-level registration. This PR adds two new nodes that
register handlers via the auto-wiring path AND a new kernel-level
DispatchResultApplier registration (`service_kernel.py`), so the gate
applies. Mirrors test_build_loop_auto_wiring_boot.py (OMN-9774), the closest
healthy sibling pattern for a COMPUTE-projects-intent -> EFFECT-persists
pair — the exact class of "wired in the contract but silently unreachable"
defect this repo already hit twice this week (OMN-14134/14139/14140).

The boot of `omninode-runtime` in a compose sandbox + RestartCount==0 assertion
is the second half of the gate; it runs in CI on the worker container, not
locally.

Ticket: OMN-14375
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    wire_from_manifest,
)
from omnibase_infra.runtime.message_dispatch_engine import (
    MessageDispatchEngine,
)
from omnibase_infra.runtime.service_intent_routing_loader import (
    load_intent_routing_table,
)

pytestmark = [pytest.mark.integration]


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    msg = "Could not locate project root from test file path."
    raise RuntimeError(msg)


_ROOT = _project_root()
COMPUTE_CONTRACT = (
    _ROOT / "src/omnibase_infra/nodes/node_pr_state_projection_compute/contract.yaml"
)
EFFECT_CONTRACT = (
    _ROOT / "src/omnibase_infra/nodes/node_pr_state_write_effect/contract.yaml"
)


class _StubResultApplier:
    async def apply(self, *args: object, **kwargs: object) -> None:
        return None


def test_pr_state_projection_contract_files_exist() -> None:
    """Sanity: both contract.yaml files exist on disk."""
    assert COMPUTE_CONTRACT.exists(), f"missing {COMPUTE_CONTRACT}"
    assert EFFECT_CONTRACT.exists(), f"missing {EFFECT_CONTRACT}"


def test_pr_state_projection_contracts_discoverable() -> None:
    """Manifest discovery parses both contracts without errors."""
    manifest = discover_contracts_from_paths([COMPUTE_CONTRACT, EFFECT_CONTRACT])
    assert manifest.total_errors == 0, f"contract parse errors: {manifest.errors!r}"
    assert manifest.total_discovered == 2
    names = {c.name for c in manifest.contracts}
    assert names == {
        "node_pr_state_projection_compute",
        "node_pr_state_write_effect",
    }


@pytest.mark.asyncio
async def test_pr_state_projection_chain_wires_from_manifest() -> None:
    """Both nodes wire into the dispatch engine with zero failures.

    Loads the real manifest from disk and runs `wire_from_manifest` with
    kernel-equivalent args, asserting zero failures for required handlers
    (the OMN-14134/OMN-14140 invariant: a handler present in Python but
    unconstructable or unrouted from the contract is a silent-drop defect,
    not a passing gate).
    """
    manifest = discover_contracts_from_paths([COMPUTE_CONTRACT, EFFECT_CONTRACT])
    assert manifest.total_errors == 0

    engine = MessageDispatchEngine(logger=MagicMock())
    result_appliers = (
        {"node_pr_state_projection_compute": _StubResultApplier()}
        if load_intent_routing_table(COMPUTE_CONTRACT)
        else {}
    )
    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=engine,
        event_bus=None,  # No live Kafka in this gate; auto-wiring is in-process.
        environment="test",
        container=MagicMock(),
        result_appliers_by_contract=result_appliers,
    )

    assert report.total_failed == 0, (
        f"wiring failures: {[r for r in report.results if r.outcome != 'wired']!r}"
    )
    assert len(report.results) == 2
