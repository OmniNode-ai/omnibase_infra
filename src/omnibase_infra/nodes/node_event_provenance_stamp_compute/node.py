# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declarative compute node for the event-provenance stamp (OMN-23981)."""

from omnibase_core.nodes.node_compute import NodeCompute


class NodeEventProvenanceStampCompute(NodeCompute):
    """Stamp an envelope with its deployment's provenance through the contract's pure handler."""
