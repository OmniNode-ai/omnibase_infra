# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry module for NodeAuthGateCompute.

Provides the RegistryInfraAuthGateCompute class for dependency injection
registration and factory methods.

Ticket: OMN-2125
"""

from omnibase_infra.nodes.node_auth_gate_compute.registry.registry_infra_auth_gate_compute import (
    RegistryInfraAuthGateCompute,
)

__all__: list[str] = ["RegistryInfraAuthGateCompute"]
