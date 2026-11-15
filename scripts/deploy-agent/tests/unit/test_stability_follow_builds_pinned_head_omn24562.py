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
    command = stability_follow_command(TAG, "omnimarket", tmp_path)
    assert command.git_ref == HEAD
    # The first-party source audit builds a branch only for a declared open pull
    # request, and the declaration carries a rollback declaration.
    assert command.open_pr == f"omnibase_infra#6011@{HEAD}"
    assert command.rollback is not None
    assert command.rollback.actor == "stab-integ2-1750"


def test_no_pin_builds_the_tracking_ref(tmp_path: Path) -> None:
    command = stability_follow_command(TAG, "omnimarket", tmp_path)
    assert command.git_ref == TRACKING
    assert command.open_pr == ""
    assert command.rollback is None


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


# The stability lane builds only the open pull request its own live pin declares.


def _admit(state_dir: Path, lane: EnumRuntimeLane, open_pr: str) -> None:
    load_dev_boot_guard("handler_dev_lane_code_ref_pin").admit_open_pr(
        lane is EnumRuntimeLane.STABILITY_TEST, open_pr, state_dir
    )


def test_the_follow_s_own_declaration_is_admitted(tmp_path: Path) -> None:
    _register(tmp_path, expires_in=timedelta(hours=2))
    command = stability_follow_command(TAG, "omnimarket", tmp_path)
    _admit(tmp_path, EnumRuntimeLane.STABILITY_TEST, command.open_pr)


@pytest.mark.parametrize(
    "open_pr",
    [f"omnibase_infra#6010@{HEAD}", f"omnibase_infra#6011@{'b' * 40}"],
    ids=["a_different_declared_open_pr", "another_head_of_the_pinned_pr"],
)
def test_a_different_declared_open_pr_is_refused_on_stability(
    tmp_path: Path, open_pr: str
) -> None:
    _register(tmp_path, expires_in=timedelta(hours=2))
    with pytest.raises(ValueError, match="builds only the open pull request"):
        _admit(tmp_path, EnumRuntimeLane.STABILITY_TEST, open_pr)


def test_a_declared_open_pr_with_no_live_pin_is_refused_on_stability(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no pin is live"):
        _admit(
            tmp_path,
            EnumRuntimeLane.STABILITY_TEST,
            f"omnibase_infra#6011@{HEAD}",
        )


def test_dev_and_undeclared_builds_are_not_read(tmp_path: Path) -> None:
    _admit(tmp_path, EnumRuntimeLane.DEV, f"omnibase_infra#6010@{HEAD}")
    _admit(tmp_path, EnumRuntimeLane.STABILITY_TEST, "")


def test_preflight_refuses_a_foreign_declaration_before_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    from deploy_agent import agent as agent_mod

    _register(tmp_path, expires_in=timedelta(hours=2))
    monkeypatch.setattr(agent_mod, "STATE_DIR", tmp_path)
    assert sys.modules["deploy_agent.agent"] is agent_mod
    executor = Mock()
    command = SimpleNamespace(
        runtime_lane=EnumRuntimeLane.STABILITY_TEST,
        open_pr=f"omnibase_infra#6010@{HEAD}",
    )
    with pytest.raises(ValueError, match="builds only the open pull request"):
        load_dev_boot_guard().run_preflight(executor, command, Mock())
    executor.preflight.assert_not_called()
    own = stability_follow_command(TAG, "omnimarket", tmp_path)
    load_dev_boot_guard().run_preflight(executor, own, Mock())
    executor.preflight.assert_called_once()
