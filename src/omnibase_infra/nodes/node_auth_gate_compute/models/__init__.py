# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Models for the auth gate compute node.

Provides the authorization scope contract, gate request input,
and gate decision output models.

Ticket: OMN-2125
"""

from omnibase_infra.nodes.node_auth_gate_compute.models.model_auth_gate_decision import (
    ModelAuthGateDecision,
)
from omnibase_infra.nodes.node_auth_gate_compute.models.model_auth_gate_request import (
    ModelAuthGateRequest,
)
from omnibase_infra.nodes.node_auth_gate_compute.models.model_contract_work_authorization import (
    ModelContractWorkAuthorization,
)

__all__: list[str] = [
    "ModelAuthGateDecision",
    "ModelAuthGateRequest",
    "ModelContractWorkAuthorization",
]
