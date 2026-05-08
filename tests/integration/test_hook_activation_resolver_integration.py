# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for hook activation mask resolution."""

from __future__ import annotations

import pytest

from omnibase_core.enums.enum_hook_bit import EnumHookBit, hook_enabled
from omnibase_core.models.contracts.subcontracts.model_hook_activation import (
    ModelHookActivation,
)
from omnibase_infra.hooks.hook_activation_resolver import HookActivationResolver

pytestmark = pytest.mark.integration


def test_hook_activations_resolve_to_runtime_hook_mask() -> None:
    """Validate hook activation contracts resolve into a runtime hook mask."""
    activations = [
        ModelHookActivation.model_validate(
            {
                "hook_bit": "CI_REMINDER",
                "enabled_by_default": True,
                "description": "Enable CI reminder hook by default.",
            }
        ),
        ModelHookActivation.model_validate(
            {
                "hook_bit": "AUTO_HOSTILE_REVIEW",
                "enabled_by_default": False,
                "description": "Keep hostile review opt-in for this package.",
            }
        ),
    ]

    mask = HookActivationResolver(activations=activations).compute_mask()

    assert hook_enabled(EnumHookBit.CI_REMINDER, mask=mask)
    assert not hook_enabled(EnumHookBit.AUTO_HOSTILE_REVIEW, mask=mask)
