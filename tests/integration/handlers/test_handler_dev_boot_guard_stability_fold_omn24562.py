# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The stability lane retires its folded effects service from the real policy (OMN-24562).

Reads the committed ``docker/runtime-policy.env`` rather than a fixture, so the
retirement the handler computes is the one the compose gates share.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.handlers import handler_dev_boot_guard as guard_mod

pytestmark = pytest.mark.integration

POLICY_PATH = Path(__file__).resolve().parents[3] / "docker" / "runtime-policy.env"


def _policy() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in POLICY_PATH.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def test_stability_prefix_retires_only_the_mapped_effects_service() -> None:
    retired = guard_mod.folded_dev_services(
        _policy(), frozenset({"effects"}), "STABILITY_TEST"
    )
    assert "runtime-effects" in retired


def test_a_lane_that_folds_nothing_retires_nothing() -> None:
    assert (
        guard_mod.folded_dev_services(_policy(), frozenset(), "STABILITY_TEST") == set()
    )
