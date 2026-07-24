# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime profile identity tests."""

from __future__ import annotations

import pytest

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


class TestOnexSecretPolicyOverride:
    """OMN-14951: ONEX_SECRET_POLICY is lane-scoped, independent of
    RUNTIME_PROFILE's role identity. Without this override, prefetch_policy
    "required" is structurally unreachable on every role-based profile
    (main/effects/workers/projection-api/canary all hardcode "disabled")."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ONEX_SECRET_POLICY", raising=False)

    def test_role_profiles_default_to_disabled_without_override(self) -> None:
        """Baseline (RED without this feature): a role-based profile alone
        can never reach prefetch_policy='required'."""
        for profile_name in ("main", "effects", "workers", "projection-api", "canary"):
            assert load_runtime_profile(profile_name).prefetch_policy == "disabled"

    def test_override_promotes_role_profile_to_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_SECRET_POLICY", "required")
        for profile_name in ("main", "effects", "workers", "projection-api", "canary"):
            profile = load_runtime_profile(profile_name)
            # Role identity is preserved -- only prefetch_policy changes.
            assert profile.name == profile_name
            assert profile.prefetch_policy == "required"

    def test_override_can_downgrade_production_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The override is a real override, not just an upgrade path."""
        monkeypatch.setenv("ONEX_SECRET_POLICY", "best_effort")
        assert load_runtime_profile("production").prefetch_policy == "best_effort"

    def test_case_insensitive_and_whitespace_tolerant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_SECRET_POLICY", "  REQUIRED  ")
        assert load_runtime_profile("main").prefetch_policy == "required"

    def test_invalid_value_is_ignored_not_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_SECRET_POLICY", "not-a-real-policy")
        # Falls back to the profile's own policy rather than crashing boot.
        assert load_runtime_profile("main").prefetch_policy == "disabled"

    def test_empty_value_is_a_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ONEX_SECRET_POLICY", "")
        assert load_runtime_profile("main").prefetch_policy == "disabled"

    def test_unset_leaves_profile_unchanged(self) -> None:
        assert load_runtime_profile("production").prefetch_policy == "required"
        assert load_runtime_profile("main").prefetch_policy == "disabled"
