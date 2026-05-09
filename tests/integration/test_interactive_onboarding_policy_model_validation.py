# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for interactive onboarding policy model validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.onboarding.model_interactive_policy import ModelInteractivePolicy

pytestmark = pytest.mark.integration

POLICY_PATH = (
    Path(__file__).parent.parent.parent
    / "src"
    / "omnibase_infra"
    / "onboarding"
    / "policies"
    / "interactive_onboarding.yaml"
)


def _load_policy_raw() -> dict[str, object]:
    raw = yaml.safe_load(POLICY_PATH.read_text())
    assert isinstance(raw, dict)
    return raw


def test_interactive_policy_rejects_empty_steps_before_defaulting_start() -> None:
    raw = deepcopy(_load_policy_raw())
    raw["steps"] = []
    raw["transitions"] = []
    raw["env_output"] = {}

    with pytest.raises(ValidationError, match="at least one step"):
        ModelInteractivePolicy.model_validate(raw)


def test_interactive_policy_rejects_unknown_explicit_start_step() -> None:
    raw = deepcopy(_load_policy_raw())
    raw["start_step"] = "missing_start_step"

    with pytest.raises(ValidationError, match="start_step references unknown step"):
        ModelInteractivePolicy.model_validate(raw)
