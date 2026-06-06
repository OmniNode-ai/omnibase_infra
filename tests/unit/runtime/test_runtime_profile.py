# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime profile identity tests."""

from __future__ import annotations

from omnibase_infra.runtime.runtime_profile import load_runtime_profile


def test_runtime_lane_profiles_preserve_identity() -> None:
    """Runtime lane names must not fall back to default."""
    for profile_name in ("main", "effects", "workers", "projection-api", "canary"):
        assert load_runtime_profile(profile_name).name == profile_name


def test_unknown_runtime_profile_still_falls_back_to_default() -> None:
    assert load_runtime_profile("unknown-lane").name == "default"
