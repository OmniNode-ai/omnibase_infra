# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read contract-owned host-effect policy."""

from pathlib import Path

from omnibase_infra.models.model_deploy_agent_branch_trial_policy import (
    ModelDeployAgentBranchTrialPolicy,
)


def load_policy(contract_path: Path) -> ModelDeployAgentBranchTrialPolicy:
    from yaml import safe_load

    return ModelDeployAgentBranchTrialPolicy.model_validate(
        safe_load(contract_path.read_text())["config"]["branch_trial"]
    )
