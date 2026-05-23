# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Declarative compute node for contract validation."""

from __future__ import annotations

from omnibase_core.nodes.node_compute import NodeCompute


class NodeContractValidateCompute(NodeCompute):
    """Declarative shell; behavior is routed by contract.yaml."""


__all__ = ["NodeContractValidateCompute"]
