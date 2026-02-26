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
from omnibase_infra.nodes.node_contract_resolver_bridge.registry.registry_infra_contract_resolver_bridge import (
    RegistryInfraContractResolverBridge,
)

__all__ = [
    "NodeContractResolverBridge",
    "RegistryInfraContractResolverBridge",
    "load_contract_resolver_bridge_config",
]
