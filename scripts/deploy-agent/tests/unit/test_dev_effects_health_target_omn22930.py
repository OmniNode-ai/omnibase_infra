# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22930: primary dev's effects health target reads the runtime that carries the role.

The retired ``runtime-effects`` container answered on the effects port. With the
role folded into main, a verify that still read that container would pass on a
stale one or fail on an absent one. The target keeps the effects port and moves
to main's container.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from deploy_agent.events import EnumRuntimeLane
from deploy_agent.executor import (
    SERVICE_HEALTH_TARGET_INDEX,
    lane_config_for,
    lane_folded_profiles,
)

pytestmark = pytest.mark.unit

POLICY = Path(__file__).resolve().parents[4] / "docker/runtime-policy.env"


def _targets(policy: Path) -> dict[str, tuple[str, int]]:
    with patch("deploy_agent.executor.RUNTIME_POLICY_ENV_FILE", policy):
        config = lane_config_for(EnumRuntimeLane.DEV)
    names = {index: name for name, index in SERVICE_HEALTH_TARGET_INDEX.items()}
    return {names[i]: target for i, target in enumerate(config.runtime_health_targets)}


def test_committed_policy_reads_the_effects_port_from_main() -> None:
    with patch("deploy_agent.executor.RUNTIME_POLICY_ENV_FILE", POLICY):
        assert "effects" in lane_folded_profiles(EnumRuntimeLane.DEV)
    targets = _targets(POLICY)
    main_container, main_port = targets["omninode-runtime"]
    effects_container, effects_port = targets["runtime-effects"]
    assert effects_container == main_container
    assert effects_port != main_port  # the effects port is kept, served by main


def test_without_the_fold_the_target_is_the_effects_container(tmp_path: Path) -> None:
    """FAILING CONTROL: unfolded policy keeps the separate container as the target."""
    unfolded = tmp_path / "runtime-policy.env"
    unfolded.write_text(
        "\n".join(
            line
            for line in POLICY.read_text().splitlines()
            if not line.startswith(
                ("DEV_RUNTIME_MAIN_FOLDED_PROFILES", "DEV_RUNTIME_MAIN_FOLD_")
            )
        )
        + "\nDEV_RUNTIME_MAIN_FOLDED_PROFILES=\n"
    )
    targets = _targets(unfolded)
    assert targets["runtime-effects"][0] != targets["omninode-runtime"][0]
