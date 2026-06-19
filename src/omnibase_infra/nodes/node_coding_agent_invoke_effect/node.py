# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative coding-agent invoke EFFECT node (OMN-13247).

The ONLY I/O node in the workflow: runs the claude/codex subprocess in the
workspace and captures the git-derived diff. Its contract is the ONLY one that
declares ``descriptor.agent_invocation: true`` — the contract fact the OMN-13219
gate reads to tell an agent effect from a banned inference tier. contract.yaml +
the handler drive behavior; the node body stays empty per the declarative-node
invariant.
"""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeCodingAgentInvokeEffect(NodeEffect):
    """Declarative coding-agent invoke effect node (zero custom logic)."""

    # Declarative node — all behavior defined in contract.yaml.


__all__: list[str] = ["NodeCodingAgentInvokeEffect"]
