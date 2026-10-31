# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Binding site for NodeSiblingShaPinRatchetCompute (OMN-21683).

``runtime_sibling_sha_pin_ratchet`` runs the sibling SHA-pin ratchet
through ``NodeSiblingShaPinRatchetCompute``. This module is the one place that builds it, so the
command-line entry point takes the handler from here and never constructs it by
hand.
"""

from __future__ import annotations

from omnibase_infra.nodes.node_sibling_sha_pin_ratchet_compute.handler import (
    NodeSiblingShaPinRatchetCompute,
)

__all__: list[str] = ["bind_node_sibling_sha_pin_ratchet"]


def bind_node_sibling_sha_pin_ratchet() -> NodeSiblingShaPinRatchetCompute:
    """Build the the sibling SHA-pin ratchet COMPUTE handler."""
    return NodeSiblingShaPinRatchetCompute()
