# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22930: the dev refresh gate on a lane that folds the effects role into main.

No ``runtime-effects`` container exists on such a lane, so the refresh script
leaves it out of the containers it hands the gate, the gate checks exactly the
containers it is handed, and the effects port (served by main) is still read:
that read is how a lost effects role shows.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_dev_effects_health_omn17427 import (
    _DEV,
    _HEALTHY_BODY,
    _Clock,
    _gate,
    _Lane,
    _refused,
)

REFRESH = Path(__file__).resolve().parents[1] / "refresh_dev_lane.sh"
FOLDED = ["omninode-runtime", "runtime-worker", "projection-api"]


def _core_services(folded_profiles: str) -> list[str]:
    """CORE_SERVICES as the refresh script's own line computes it, from the policy value."""
    line = next(
        line
        for line in REFRESH.read_text().splitlines()
        if line.startswith("readonly CORE_SERVICES=")
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'DEV_RUNTIME_MAIN_FOLDED_PROFILES="{folded_profiles}"\n{line}\n'
            'printf "%s\\n" "${CORE_SERVICES[@]}"',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.split()


@pytest.mark.unit
def test_the_refresh_script_leaves_the_retired_container_out_when_the_policy_folds_it() -> (
    None
):
    assert _core_services("tenant-projection,effects") == FOLDED
    # FAILING CONTROL: a policy that does not fold effects keeps the container.
    assert _core_services("tenant-projection") == [
        "omninode-runtime",
        "runtime-effects",
        "runtime-worker",
        "projection-api",
    ]
    assert _core_services("") == _core_services("tenant-projection")


@pytest.mark.unit
def test_the_gate_checks_exactly_the_containers_it_is_handed() -> None:
    ids = {name: f"c-{name}" for name in FOLDED}
    report = _gate(_Lane([_HEALTHY_BODY]), _Clock(), container_ids=ids)
    assert [s.service for s in report.services] == FOLDED
    assert report.effects_health_ok is True


@pytest.mark.unit
def test_failing_control_with_none_named_the_gate_demands_the_effects_container() -> (
    None
):
    report = _gate(_Lane([_HEALTHY_BODY]), _Clock(), container_ids={})
    assert "runtime-effects" in [s.service for s in report.services]
    assert report.core_services_running is False  # no container was given


@pytest.mark.unit
def test_a_folded_lane_still_fails_when_the_effects_port_does_not_answer() -> None:
    ids = {name: f"c-{name}" for name in FOLDED}
    report = _gate(_Lane([_refused()]), _Clock(), container_ids=ids)
    assert report.health_ok is True
    assert report.effects_health_ok is False
    assert report.overall != "PASS"


@pytest.mark.unit
def test_the_unfolded_service_set_is_still_the_verifiers_default() -> None:
    assert "runtime-effects" in _DEV.CORE_SERVICE_NAMES
