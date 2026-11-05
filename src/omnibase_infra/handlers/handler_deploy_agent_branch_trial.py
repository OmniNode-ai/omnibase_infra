# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canonical request adapter for the dev agent's trusted host supervisor."""

from pathlib import Path

from omnibase_core.container import ModelONEXContainer
from omnibase_core.utils.util_run_coro_sync import run_coro_sync
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.deploy_agent_branch_trial_effect import effect
from omnibase_infra.models.model_deploy_agent_branch_trial_request import (
    ModelDeployAgentBranchTrialRequest,
)
from omnibase_infra.models.model_deploy_agent_branch_trial_result import (
    ModelDeployAgentBranchTrialResult,
)


class HandlerDeployAgentBranchTrial:
    def __init__(self, container: ModelONEXContainer) -> None:
        self._state_root = container.get_service(Path)

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelDeployAgentBranchTrialRequest
    ) -> ModelDeployAgentBranchTrialResult:
        return effect.queue_request(request, self._state_root)


def queue_from_host(
    state_root: Path, request: ModelDeployAgentBranchTrialRequest
) -> ModelDeployAgentBranchTrialResult:
    container = ModelONEXContainer()
    run_coro_sync(container.service_registry.register_instance(Path, state_root))
    handler = HandlerDeployAgentBranchTrial(container)
    run_coro_sync(
        container.service_registry.register_instance(
            HandlerDeployAgentBranchTrial, handler
        )
    )
    return container.get_service(HandlerDeployAgentBranchTrial).handle(request)
