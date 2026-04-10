# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Onboarding orchestrator node (OMN-8274).

Declarative node — all behavior defined in contract.yaml.
"""

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.nodes import NodeOrchestrator


class NodeOnboardingOrchestrator(NodeOrchestrator):
    """Declarative onboarding orchestrator — driven entirely by contract.yaml."""

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__ = ["NodeOnboardingOrchestrator"]
