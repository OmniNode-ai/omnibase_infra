# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Contract Resolver Bridge node package.

Exposes NodeContractResolveCompute via synchronous HTTP for dashboard use.

Ticket: OMN-2756
"""

from omnibase_infra.nodes.node_contract_resolver_bridge.node import (
    NodeContractResolverBridge,
    load_contract_resolver_bridge_config,
)

__all__ = [
    "NodeContractResolverBridge",
    "load_contract_resolver_bridge_config",
]
