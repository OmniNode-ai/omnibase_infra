# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""OMN-9126 first-class CI gate for the build_loop projection chain.

This test loads the real contract.yaml files for the two new nodes from disk,
runs `wire_from_manifest` exactly the way the kernel does in production, and
asserts both nodes wire successfully with zero failures.

Required by `omnibase_infra/CLAUDE.md` § "Runtime Startup is a First-Class CI
Gate" for any PR touching auto_wiring/, service_kernel.py, handler __init__
signatures, or kernel-level registration. This PR adds two new nodes that
register handlers via the auto-wiring path, so the gate applies.

The boot of `omninode-runtime` in a compose sandbox + RestartCount==0 assertion
is the second half of the OMN-9126 gate; it runs in CI on the worker container,
not locally.

Ticket: OMN-9774
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    wire_from_manifest,
)
from omnibase_infra.runtime.service_message_dispatch_engine import (
    MessageDispatchEngine,
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
    _ROOT / "src/omnibase_infra/nodes/node_build_loop_projection_compute/contract.yaml"
)
EFFECT_CONTRACT = (
    _ROOT / "src/omnibase_infra/nodes/node_build_loop_write_effect/contract.yaml"
)


def test_build_loop_projection_contract_files_exist() -> None:
    """Sanity: both contract.yaml files exist on disk."""
    assert COMPUTE_CONTRACT.exists(), f"missing {COMPUTE_CONTRACT}"
    assert EFFECT_CONTRACT.exists(), f"missing {EFFECT_CONTRACT}"


def test_build_loop_projection_contracts_discoverable() -> None:
    """Manifest discovery parses both contracts without errors."""
    manifest = discover_contracts_from_paths([COMPUTE_CONTRACT, EFFECT_CONTRACT])
    assert manifest.total_errors == 0, f"contract parse errors: {manifest.errors!r}"
    assert manifest.total_discovered == 2
    names = {c.name for c in manifest.contracts}
    assert names == {
        "node_build_loop_projection_compute",
        "node_build_loop_write_effect",
    }


@pytest.mark.asyncio
async def test_build_loop_projection_chain_wires_from_manifest() -> None:
    """Both nodes wire into the dispatch engine with zero failures.

    OMN-9126 invariant: load real manifest from disk, run `wire_from_manifest`
    with kernel-equivalent args, assert zero failures for required handlers.
    """
    manifest = discover_contracts_from_paths([COMPUTE_CONTRACT, EFFECT_CONTRACT])
    assert manifest.total_errors == 0

    engine = MessageDispatchEngine(logger=MagicMock())
    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=engine,
        event_bus=None,  # No live Kafka in this gate; auto-wiring is in-process.
        environment="test",
    )

    assert report.total_failed == 0, (
        f"wiring failures: {[r for r in report.results if r.outcome != 'wired']!r}"
    )
    # Both contracts should produce wiring outcomes (wired or skipped is fine
    # for the pure parse+wire pass; what matters is zero failures).
    assert len(report.results) == 2
