# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Models for the delegation orchestrator node."""

from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_result import (
    ModelDelegationResult,
)

__all__: list[str] = [
    "ModelDelegationRequest",
    "ModelDelegationResult",
]
