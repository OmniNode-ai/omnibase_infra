# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host composition root: inject state and contract policy into the effect."""

from pathlib import Path

from omnibase_core.container import ModelONEXContainer
from omnibase_core.utils.util_run_coro_sync import run_coro_sync
from omnibase_infra.handlers.handler_deploy_agent_branch_trial import (
    HandlerDeployAgentBranchTrial,
)
from omnibase_infra.models.model_deploy_agent_branch_trial_policy import (
    ModelDeployAgentBranchTrialPolicy,
)
from omnibase_infra.models.model_deploy_agent_branch_trial_request import (
    ModelDeployAgentBranchTrialRequest,
)
from omnibase_infra.models.model_deploy_agent_branch_trial_result import (
    ModelDeployAgentBranchTrialResult,
)


def load_policy(contract_path: Path) -> ModelDeployAgentBranchTrialPolicy:
    from yaml import safe_load

    return ModelDeployAgentBranchTrialPolicy.model_validate(
        safe_load(contract_path.read_text())["config"]["branch_trial"]
    )


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


def trial_result_topic(contract_path: Path) -> str:
    from yaml import safe_load

    return str(safe_load(contract_path.read_text())["terminal_event"])
