# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for interactive onboarding policy Pydantic models."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

pytestmark = pytest.mark.unit

POLICY_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "omnibase_infra"
    / "onboarding"
    / "policies"
    / "interactive_onboarding.yaml"
)


def test_interactive_policy_loads_into_model() -> None:
    from omnibase_infra.onboarding.model_interactive_policy import (
        ModelInteractivePolicy,
    )

    raw = yaml.safe_load(POLICY_PATH.read_text())
    policy = ModelInteractivePolicy.model_validate(raw)
    assert policy.policy_name == "interactive_onboarding"
    assert policy.policy_type == "interactive"
    assert len(policy.steps) == 9
    assert len(policy.transitions) >= 7


def test_step_types_are_valid() -> None:
    from omnibase_infra.onboarding.model_interactive_policy import (
        ModelInteractivePolicy,
    )

    raw = yaml.safe_load(POLICY_PATH.read_text())
    policy = ModelInteractivePolicy.model_validate(raw)
    valid_types = {"choice", "multi_choice", "text", "action"}
    for step in policy.steps:
        assert step.type in valid_types


def test_terminal_steps_have_env_output() -> None:
    from omnibase_infra.onboarding.model_interactive_policy import (
        ModelInteractivePolicy,
    )

    raw = yaml.safe_load(POLICY_PATH.read_text())
    policy = ModelInteractivePolicy.model_validate(raw)
    terminal_ids = {t.from_step for t in policy.transitions if t.terminal}
    for tid in terminal_ids:
        assert tid in policy.env_output


def test_start_step_field() -> None:
    from omnibase_infra.onboarding.model_interactive_policy import (
        ModelInteractivePolicy,
    )

    raw = yaml.safe_load(POLICY_PATH.read_text())
    policy = ModelInteractivePolicy.model_validate(raw)
    assert policy.start_step == "choose_deployment_mode"


def test_invalid_step_type_rejected() -> None:
    from omnibase_infra.onboarding.model_interactive_step import ModelInteractiveStep

    with pytest.raises(ValidationError):
        ModelInteractiveStep(id="x", prompt="x", type="bogus")


def test_duplicate_step_ids_rejected() -> None:
    from omnibase_infra.onboarding.model_interactive_policy import (
        ModelInteractivePolicy,
    )

    raw = yaml.safe_load(POLICY_PATH.read_text())
    raw["steps"].append(raw["steps"][0])
    with pytest.raises(ValueError, match=r"[Dd]uplicate"):
        ModelInteractivePolicy.model_validate(raw)


def test_transition_to_unknown_step_rejected() -> None:
    from omnibase_infra.onboarding.model_interactive_policy import (
        ModelInteractivePolicy,
    )

    raw = yaml.safe_load(POLICY_PATH.read_text())
    raw["transitions"][0]["responses"]["local"]["next"] = "nonexistent_step"
    with pytest.raises(ValueError, match="unknown step"):
        ModelInteractivePolicy.model_validate(raw)


def test_missing_terminal_env_output_rejected() -> None:
    from omnibase_infra.onboarding.model_interactive_policy import (
        ModelInteractivePolicy,
    )

    raw = yaml.safe_load(POLICY_PATH.read_text())
    del raw["env_output"]["write_config_local"]
    with pytest.raises(ValueError, match="env_output"):
        ModelInteractivePolicy.model_validate(raw)


def test_extra_fields_rejected() -> None:
    from omnibase_infra.onboarding.model_interactive_step import ModelInteractiveStep

    with pytest.raises(ValidationError):
        ModelInteractiveStep(id="x", prompt="x", type="choice", bogus_field="y")
