# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

import pytest

from omnibase_core.enums.enum_hook_bit import EnumHookBit
from omnibase_core.models.contracts.subcontracts.model_hook_activation import (
    ModelHookActivation,
)
from omnibase_infra.hooks.hook_activation_resolver import HookActivationResolver


@pytest.mark.unit
def test_resolver_returns_activated_bits_for_installed_packages() -> None:
    activations = [
        ModelHookActivation(hook_bit=EnumHookBit.CI_REMINDER, enabled_by_default=True),
        ModelHookActivation(
            hook_bit=EnumHookBit.AUTO_HOSTILE_REVIEW, enabled_by_default=False
        ),
    ]
    resolver = HookActivationResolver(activations=activations)
    mask = resolver.compute_mask()
    assert mask & int(EnumHookBit.CI_REMINDER) != 0
    assert mask & int(EnumHookBit.AUTO_HOSTILE_REVIEW) == 0


@pytest.mark.unit
def test_resolver_empty_activations_returns_zero_mask() -> None:
    resolver = HookActivationResolver(activations=[])
    mask = resolver.compute_mask()
    assert mask == 0


@pytest.mark.unit
def test_resolver_all_enabled_by_default_sets_all_bits() -> None:
    activations = [
        ModelHookActivation(hook_bit=EnumHookBit.CI_REMINDER, enabled_by_default=True),
        ModelHookActivation(hook_bit=EnumHookBit.RUFF_FIX, enabled_by_default=True),
    ]
    resolver = HookActivationResolver(activations=activations)
    mask = resolver.compute_mask()
    assert mask & int(EnumHookBit.CI_REMINDER) != 0
    assert mask & int(EnumHookBit.RUFF_FIX) != 0


@pytest.mark.unit
def test_resolver_none_enabled_by_default_returns_zero_mask() -> None:
    activations = [
        ModelHookActivation(hook_bit=EnumHookBit.CI_REMINDER, enabled_by_default=False),
        ModelHookActivation(hook_bit=EnumHookBit.RUFF_FIX, enabled_by_default=False),
    ]
    resolver = HookActivationResolver(activations=activations)
    mask = resolver.compute_mask()
    assert mask == 0
