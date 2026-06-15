# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime profile identity tests."""

from __future__ import annotations

from omnibase_core.constants.constants_runtime_profiles import (
    CONSUMER_ATTACHED_RUNTIME_PROFILES,
    REGISTERED_RUNTIME_PROFILES,
)
from omnibase_infra.runtime.runtime_profile import _PROFILES, load_runtime_profile


def test_runtime_lane_profiles_preserve_identity() -> None:
    """Runtime lane names must not fall back to default."""
    for profile_name in ("main", "effects", "workers", "projection-api", "canary"):
        assert load_runtime_profile(profile_name).name == profile_name


def test_unknown_runtime_profile_still_falls_back_to_default() -> None:
    assert load_runtime_profile("unknown-lane").name == "default"


def test_profiles_match_core_registry() -> None:
    """OMN-12957: infra _PROFILES keys are the canonical core registry.

    Core owns the profile name set so contract validation can enforce
    ``runtime_profiles ⊆ REGISTERED_RUNTIME_PROFILES`` without a core→infra
    dependency. If this drifts, a contract could name a profile core blesses
    but infra cannot boot (or vice versa) — a silent-orphan hazard. The module
    import already asserts this; the test makes the contract explicit.
    """
    assert frozenset(_PROFILES) == REGISTERED_RUNTIME_PROFILES


def test_consumer_attached_profiles_actually_load() -> None:
    """Every consumer-attached profile must be a real, loadable runtime lane."""
    for profile_name in CONSUMER_ATTACHED_RUNTIME_PROFILES:
        assert load_runtime_profile(profile_name).name == profile_name
