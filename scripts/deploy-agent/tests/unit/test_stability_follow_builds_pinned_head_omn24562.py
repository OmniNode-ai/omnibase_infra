# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24562: a stability follow builds the live pin's head, not the tracking branch.

THE FAILURE
-----------
A code-ref pin moves the agent's own code. A deploy job's build reads a second clone,
the deploy-source clone, which the job's git phase resets to the command's ``git_ref``.
The stability follow carried the tracking ref, so with the pin registered the agent ran
the pinned code and built from ``origin/dev``'s compose files: the overlay's composed
build anchor was absent, the second build named eleven services and built the infra
Dockerfile eleven times (job ca7a5efe, buildx history: the build context was the
deploy-source clone at the dev head, not the omnimarket context), and the stability
image kept its infra-only layers.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from deploy_agent.executor import EnumRuntimeLane, load_dev_boot_guard
from deploy_agent.idle_converge import stability_follow_command

pytestmark = pytest.mark.unit

TAG = "dev-20261115134817-ebb006955dd0"
HEAD = "6013569daf5e0328a733858f9f7c49d639f90a87"
TRACKING = "origin/dev"


def _register(state_dir: Path, *, expires_in: timedelta) -> None:
    pin_handler = load_dev_boot_guard("handler_dev_lane_code_ref_pin")
    now = datetime.now(UTC)
    pin_handler.write_pin(
        state_dir,
        pin_handler.ModelDevLaneCodeRefPin(
            branch="omn-24562-stability-integ-1740",
            head=HEAD,
            pr=6011,
            ticket="OMN-24562",
            registered_by="stab-integ2-1750",
            registered_at=now - timedelta(hours=1),
            expires_at=now + expires_in,
        ),
    )


@pytest.fixture(autouse=True)
def _tracking_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_TRACKING_REF", "dev")


def test_a_live_pin_names_its_head_as_the_build_ref(tmp_path: Path) -> None:
    _register(tmp_path, expires_in=timedelta(hours=2))
    assert stability_follow_command(TAG, "omnimarket", tmp_path).git_ref == HEAD


def test_no_pin_builds_the_tracking_ref(tmp_path: Path) -> None:
    assert stability_follow_command(TAG, "omnimarket", tmp_path).git_ref == TRACKING


def test_an_expired_pin_builds_the_tracking_ref(tmp_path: Path) -> None:
    _register(tmp_path, expires_in=timedelta(seconds=-1))
    assert stability_follow_command(TAG, "omnimarket", tmp_path).git_ref == TRACKING


def test_a_missing_state_dir_has_no_pin(tmp_path: Path) -> None:
    command = stability_follow_command(TAG, "omnimarket", tmp_path / "absent")
    assert command.git_ref == TRACKING


def test_the_follow_still_names_the_tag_and_the_stability_lane(tmp_path: Path) -> None:
    _register(tmp_path, expires_in=timedelta(hours=2))
    cmd = stability_follow_command(TAG, "omnimarket", tmp_path)
    assert cmd.sibling_tags == {"omnimarket": TAG}
    assert cmd.runtime_lane is EnumRuntimeLane.STABILITY_TEST
