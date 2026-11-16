# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23333: the dev lane's receipt writer reads the container that carries the effects role.

Recorded case, 2026-11-09. The per-merge verify job that writes the lane's lab-pass receipt
(``runtime-rebuild-trigger-reusable.yml``) took its effects container from
``verify.effects_container`` of ``config/deploy_lane_routing.yaml``. That row still named the
separate effects container after the dev lane folded the effects role into main
(``DEV_RUNTIME_MAIN_FOLDED_PROFILES`` in ``docker/runtime-policy.env``), so the converge step
ended ``no such object`` and ``INDETERMINATE``, the receipt was never PASS, and the reading of
rebuild piece 11 found no PASS receipt for any merge after 2026-11-08T12:05Z.

The deploy agent's own health target already follows the fold (OMN-22930); this keeps the
verify row on the same fact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.ci.deploy_lane_verify_route import (
    ROUTING_TABLE,
    load_table,
    verify_targets,
)

pytestmark = pytest.mark.unit

POLICY = Path(__file__).resolve().parents[2] / "docker" / "runtime-policy.env"
FOLDED_KEY = "DEV_RUNTIME_MAIN_FOLDED_PROFILES"


def folded_profiles(policy_text: str) -> frozenset[str]:
    for line in policy_text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == FOLDED_KEY:
            return frozenset(p.strip().lower() for p in value.split(",") if p.strip())
    return frozenset()


def effects_target_problem(table: dict[str, Any], policy_text: str) -> str | None:
    """Why the default instance's verify row names an effects container the lane does not run."""
    targets = verify_targets(table, str(table["default_instance"]))
    folded = "effects" in folded_profiles(policy_text)
    if folded and targets.effects_container != targets.runtime_container:
        return (
            f"the dev lane folds effects into main ({FOLDED_KEY}) but verify.effects_container "
            f"is {targets.effects_container!r}, not the main container {targets.runtime_container!r}"
        )
    if not folded and targets.effects_container == targets.runtime_container:
        return "effects is not folded but verify.effects_container names the main container"
    return None


def test_golden_committed_table_reads_the_container_that_carries_effects() -> None:
    assert ROUTING_TABLE.exists()
    problem = effects_target_problem(load_table(), POLICY.read_text(encoding="utf-8"))
    assert problem is None, problem


def test_error_the_retired_container_name_is_refused_while_effects_is_folded() -> None:
    """FAILING CONTROL: the row as it stood on 2026-11-09 reads as the recorded defect."""
    table = load_table()
    verify = table["instances"][table["default_instance"]]["verify"]
    # The separate effects container of a lane is "<main>-effects" (lane_runtime_containers).
    retired = f"{verify['runtime_container']}-effects"
    verify["effects_container"] = retired
    problem = effects_target_problem(table, POLICY.read_text(encoding="utf-8"))
    assert retired
    assert problem is not None
    assert retired in problem


def test_error_an_unfolded_policy_keeps_the_separate_container() -> None:
    """FAILING CONTROL: the check follows the policy, it does not hard-code the fold."""
    table = load_table()
    unfolded = f"{FOLDED_KEY}=workers,projection-api\n"
    problem = effects_target_problem(table, unfolded)
    assert problem is not None
    assert "not folded" in problem
