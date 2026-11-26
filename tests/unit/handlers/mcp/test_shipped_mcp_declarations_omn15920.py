# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shipped contract ``mcp`` blocks list the same tools after the tools-list move.

OMN-15920: the contract ``mcp`` block is ``expose`` plus a ``tools`` list. The
declarations shipped under ``src/omnibase_infra/nodes`` were moved to it; the
consumer must list the tools those declarations listed before the move.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import omnibase_infra
from omnibase_infra.handlers.mcp.adapter_onex_to_mcp import ONEXToMCPAdapter

NODES_ROOT = Path(omnibase_infra.__file__).parent / "nodes"

# Tools the shipped declarations listed before the move: name -> timeout.
EXPECTED_SHIPPED_TOOLS = {"register_node": 30}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shipped_declarations_list_the_same_tools_as_before() -> None:
    adapter = ONEXToMCPAdapter()

    tools = await adapter.discover_tools(contracts_root=NODES_ROOT)

    assert {t.name: t.timeout_seconds for t in tools} == EXPECTED_SHIPPED_TOOLS
    register = next(t for t in tools if t.name == "register_node")
    assert register.description.startswith("Register a new ONEX node")
    # The input model path resolved to a real schema, not the bare fallback.
    assert register.input_schema is not None
