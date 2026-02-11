# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Auth Gate Compute — Work authorization decision node.

This package provides the NodeAuthGateCompute, a declarative COMPUTE node
that evaluates a 10-step authorization cascade for tool invocations.

Architecture:
    This node follows the ONEX declarative pattern where:
    - NodeAuthGateCompute is a declarative shell (no custom logic)
    - HandlerAuthGate contains the 10-step cascade logic
    - contract.yaml defines behavior via handler_routing

Core Purpose:
    Pure COMPUTE node for authorization decisions. Receives auth state +
    tool request, returns allow/deny/soft_deny. No I/O.

Decision Cascade:
     1. Whitelisted paths -> allow (plans, memory)
     2. Emergency override active -> soft_deny (with banner) / deny if no reason
     3. No run_id determinable -> deny
     4. Run context not found -> deny
     5. Auth not granted (run_id mismatch) -> deny
     6. Tool not in allowed_tools -> deny
     7. Path not matching allowed_paths glob -> deny
     8. Repo not in repo_scopes -> deny
     9. Auth expired -> deny
    10. All checks pass -> allow

Related Tickets:
    - OMN-2125: Auth Gate Nodes — Work Authorization Compute Node

Example:
    >>> from unittest.mock import MagicMock
    >>> from omnibase_infra.nodes.node_auth_gate_compute import (
    ...     HandlerAuthGate,
    ...     NodeAuthGateCompute,
    ...     RegistryInfraAuthGateCompute,
    ... )
    >>>
    >>> container = MagicMock()
    >>> handler = HandlerAuthGate(container)
"""

from omnibase_infra.nodes.node_auth_gate_compute.handlers import (
    HandlerAuthGate,
)
from omnibase_infra.nodes.node_auth_gate_compute.node import (
    NodeAuthGateCompute,
)
from omnibase_infra.nodes.node_auth_gate_compute.registry import (
    RegistryInfraAuthGateCompute,
)

__all__: list[str] = [
    "HandlerAuthGate",
    "NodeAuthGateCompute",
    "RegistryInfraAuthGateCompute",
]
