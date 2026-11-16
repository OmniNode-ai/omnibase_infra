# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""NodeSiblingClonePinEffect - Declarative EFFECT node for sibling-clone pin resolution.

Declarative EFFECT node that resolves the actual pin (version + git SHA) of a
sibling clone. The handler performs I/O (reads ``pyproject.toml``, runs ``git``),
so the node extends ``NodeEffect``.

Design Rationale:
    ONEX nodes are declarative shells driven by contract.yaml. The node class
    extends the archetype base class and contains no custom logic; all effect
    behavior is defined in handlers configured via handler_routing in the contract.

Ticket: OMN-23947
"""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeSiblingClonePinEffect(NodeEffect):
    """Declarative EFFECT node for sibling-clone pin resolution.

    All behavior is defined in contract.yaml and delegated to
    HandlerSiblingClonePin. This node contains no custom logic.

    See Also:
        - handlers/handler_sibling_clone_pin.py: the pyproject/git I/O logic
        - contract.yaml: node I/O and handler routing configuration
    """

    # Declarative node - all behavior defined in contract.yaml


__all__: list[str] = ["NodeSiblingClonePinEffect"]
