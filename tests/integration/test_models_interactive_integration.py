# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for interactive onboarding Pydantic models (OMN-10778).

Validates the re-export shim (models_interactive.py) and the canonical
interactive_onboarding.yaml policy against the full model stack.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

POLICY_PATH = (
    Path(__file__).parent.parent.parent
    / "src"
    / "omnibase_infra"
    / "onboarding"
    / "policies"
    / "interactive_onboarding.yaml"
)


def test_models_interactive_shim_exports_all_four_models() -> None:
    from omnibase_infra.onboarding.models_interactive import (
        ModelInteractivePolicy,
        ModelInteractiveStep,
        ModelTransition,
        ModelTransitionBranch,
    )

    assert ModelInteractivePolicy is not None
    assert ModelInteractiveStep is not None
    assert ModelTransition is not None
    assert ModelTransitionBranch is not None


def test_canonical_policy_loads_via_shim() -> None:
    from omnibase_infra.onboarding.models_interactive import ModelInteractivePolicy

    raw = yaml.safe_load(POLICY_PATH.read_text())
    policy = ModelInteractivePolicy.model_validate(raw)
    assert policy.policy_name == "interactive_onboarding"
    assert policy.start_step == "choose_deployment_mode"
    assert len(policy.steps) == 9


def test_transition_branch_set_state_roundtrip() -> None:
    from omnibase_infra.onboarding.models_interactive import ModelTransitionBranch

    branch = ModelTransitionBranch(
        next="next_step",
        set_state={"mode": "local", "flag": True, "count": 3},
    )
    dumped = branch.model_dump()
    restored = ModelTransitionBranch.model_validate(dumped)
    assert restored.set_state == branch.set_state


def test_canonical_policy_graph_integrity() -> None:
    from omnibase_infra.onboarding.models_interactive import ModelInteractivePolicy

    raw = yaml.safe_load(POLICY_PATH.read_text())
    policy = ModelInteractivePolicy.model_validate(raw)

    step_ids = {s.id for s in policy.steps}
    terminal_ids = {t.from_step for t in policy.transitions if t.terminal}

    for t in policy.transitions:
        assert t.from_step in step_ids
        if not t.terminal:
            for branch in (t.responses or {}).values():
                assert branch.next in step_ids
    for tid in terminal_ids:
        assert tid in policy.env_output
