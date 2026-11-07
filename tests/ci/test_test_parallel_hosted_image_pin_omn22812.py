# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The Tests (Split) job must be pinned by literal to a fixed hosted image (OMN-22812).

``ubuntu-latest`` became Ubuntu 26.04 on 2026-11-06. That image carries no
PostgreSQL 16 binaries, so the ephemeral-PostgreSQL tests skip there, and Skip
Count Ratchet (OMN-18776) reads the new skips as growth of the skipped set.

A pinned fallback behind ``vars.OMNI_TRUSTED_CI_RUNS_ON_JSON`` does not help:
that variable is set to ``["ubuntu-latest"]`` at organisation scope and on this
repository (``config/runner_routing_policy.yaml``), so the fallback literal never
applies. Only a literal ``runs-on`` answers to no variable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
ROUTING_POLICY = REPO_ROOT / "config" / "runner_routing_policy.yaml"
PINNED_IMAGE = "ubuntu-24.04"


def is_pinned_literal(runs_on: object) -> bool:
    """True only for the pinned image as a bare literal, with no expression."""
    return runs_on == PINNED_IMAGE


def _test_parallel_runs_on() -> object:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["test-parallel"]["runs-on"]


def test_the_probe_refuses_the_selector_with_a_pinned_fallback() -> None:
    """Control: a variable-read expression, even with a pinned fallback, is refused."""
    behind_variable = (
        "${{ fromJSON(vars.OMNI_TRUSTED_CI_RUNS_ON_JSON || '[\"ubuntu-24.04\"]') }}"
    )
    assert not is_pinned_literal(behind_variable)
    assert not is_pinned_literal("ubuntu-latest")


def test_the_trusted_variable_is_asserted_to_resolve_to_a_floating_image() -> None:
    """The reason a fallback pin is dead code: the variable is set, to ubuntu-latest."""
    policy = yaml.safe_load(ROUTING_POLICY.read_text(encoding="utf-8"))
    assert (
        policy["probe_placement_variables"]["omnibase_infra"][
            "OMNI_TRUSTED_CI_RUNS_ON_JSON"
        ]
        == '["ubuntu-latest"]'
    )


def test_test_parallel_is_pinned_by_literal() -> None:
    """The Tests (Split) job runs on the pinned image whatever the variables say."""
    assert is_pinned_literal(_test_parallel_runs_on())
