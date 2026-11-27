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
import yaml

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


@pytest.mark.unit
def test_shipped_declarations_use_tools_list_not_single_tool_block() -> None:
    """Every shipped ``mcp`` block is ``expose`` plus a ``tools`` list.

    The listing test above passes for either block shape the adapter once read,
    so this pins the shape itself: a shipped block left on the old single-tool
    keys (``tool_name`` beside ``expose``) fails here.
    """
    blocks = {}
    for contract in sorted(NODES_ROOT.glob("*/contract.yaml")):
        mcp = (yaml.safe_load(contract.read_text(encoding="utf-8")) or {}).get("mcp")
        if mcp is not None:
            blocks[contract.parent.name] = mcp

    assert blocks, "no shipped contract declares an mcp block"
    for node_name, mcp in blocks.items():
        assert "tool_name" not in mcp, f"{node_name}: old single-tool mcp block"
        assert isinstance(mcp.get("tools"), list) and mcp["tools"], node_name
    names = [entry["tool_name"] for mcp in blocks.values() for entry in mcp["tools"]]
    assert names == list(EXPECTED_SHIPPED_TOOLS)
