# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for core-owned runtime profile registry parity."""

from __future__ import annotations

from omnibase_core.constants.constants_runtime_profiles import (
    CONSUMER_ATTACHED_RUNTIME_PROFILES,
    REGISTERED_RUNTIME_PROFILES,
)
from omnibase_infra.runtime.runtime_profile import _PROFILES, load_runtime_profile


def test_infra_runtime_profiles_match_core_registry() -> None:
    """Infra must be able to boot every profile accepted by core validators."""
    assert frozenset(_PROFILES) == REGISTERED_RUNTIME_PROFILES


def test_consumer_attached_runtime_profiles_preserve_identity() -> None:
    """Consumer lanes declared by core must not silently fall back to default."""
    for profile_name in sorted(CONSUMER_ATTACHED_RUNTIME_PROFILES):
        assert load_runtime_profile(profile_name).name == profile_name
