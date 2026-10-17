# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20935: no workflow reads the CI bus lane table from omnimarket.

The lane table (lane -> broker, transport, projection readback) is private
deployment config. It lives in the CI overlay repository
(``config/lab/ci_bus_lanes.yaml``); every workflow that publishes to, or probes,
the CI bus fetches it through ``.github/actions/fetch-lane-table``. These
cases pin the composite action and refuse a workflow that goes back to a sparse
checkout of omnimarket's ``config/ci_bus_lanes.yaml``, which is the file
omnimarket removes once its readers have moved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = sorted((_REPO_ROOT / ".github" / "workflows").glob("*.yml"))
_ACTION = _REPO_ROOT / ".github" / "actions" / "fetch-lane-table" / "action.yml"
_ACTION_USES = "./.github/actions/fetch-lane-table"
_OMNIMARKET = "OmniNode-ai/omnimarket"
_LANE_TABLE = "config/ci_bus_lanes.yaml"
_REUSABLES_WITH_OCC_CREDENTIALS = (
    "runtime-rebuild-trigger-reusable.yml",
    "onex-api-lab-delivery-reusable.yml",
)


def _steps(workflow: Path) -> list[dict[str, Any]]:
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = []
    for job in (document.get("jobs") or {}).values():
        steps.extend(job.get("steps") or [])
    return steps


@pytest.mark.unit
@pytest.mark.parametrize("workflow", _WORKFLOWS, ids=lambda p: p.name)
def test_no_workflow_checks_out_the_lane_table_from_omnimarket(
    workflow: Path,
) -> None:
    offenders = [
        step.get("name", "<unnamed>")
        for step in _steps(workflow)
        if step.get("with", {}).get("repository") == _OMNIMARKET
        and _LANE_TABLE in str(step.get("with", {}).get("sparse-checkout", ""))
    ]
    assert not offenders, (
        f"{workflow.name} sparse-checks-out {_LANE_TABLE} from omnimarket in "
        f"{offenders}; read it through {_ACTION_USES} instead (OMN-20935)"
    )


@pytest.mark.unit
@pytest.mark.parametrize("workflow", _WORKFLOWS, ids=lambda p: p.name)
def test_every_fetch_step_names_the_overlay_repo_credentials_and_destination(
    workflow: Path,
) -> None:
    for step in _steps(workflow):
        if step.get("uses") != _ACTION_USES:
            continue
        with_block = step["with"]
        assert with_block["overlay-repo"] == "${{ vars.OMNI_CI_OVERLAY_REPO }}"
        assert "private-key" in with_block
        assert str(with_block["app-id"]).startswith("${{ secrets.ONEXBOT_")
        assert str(with_block["dest"]).endswith("ci_bus_lanes.yaml")


@pytest.mark.unit
@pytest.mark.parametrize("name", _REUSABLES_WITH_OCC_CREDENTIALS)
def test_a_reusable_that_fetches_the_table_declares_the_app_secrets(
    name: str,
) -> None:
    """A called workflow only sees the secrets its ``workflow_call`` declares."""
    document = yaml.safe_load(
        (_REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
    )
    triggers = document.get("on", document.get(True))
    secrets = triggers["workflow_call"]["secrets"]
    assert {"ONEXBOT_OCC_APP_ID", "ONEXBOT_OCC_PRIVATE_KEY"} <= set(secrets)
    fetch = next(
        step
        for step in _steps(_REPO_ROOT / ".github" / "workflows" / name)
        if step.get("uses") == _ACTION_USES
    )
    assert fetch["with"]["app-id"] == "${{ secrets.ONEXBOT_OCC_APP_ID }}"
    assert fetch["with"]["private-key"] == "${{ secrets.ONEXBOT_OCC_PRIVATE_KEY }}"


@pytest.mark.unit
class TestTheFetchAction:
    """The composite action reads one file, read-only, and fails loud."""

    @staticmethod
    def _action() -> dict[str, Any]:
        return yaml.safe_load(_ACTION.read_text(encoding="utf-8"))

    def test_it_reads_only_the_lane_table_from_the_overlay_repository(self) -> None:
        steps = self._action()["runs"]["steps"]
        checkout = next(s for s in steps if "actions/checkout" in s.get("uses", ""))
        assert checkout["with"]["sparse-checkout"].split() == [
            "config/lab/ci_bus_lanes.yaml"
        ]
        assert checkout["with"]["persist-credentials"] is False
        assert checkout["with"]["repository"].startswith(
            "${{ github.repository_owner }}/"
        )

    def test_the_token_is_read_only_and_scoped_to_the_overlay_repository(self) -> None:
        steps = self._action()["runs"]["steps"]
        mint = next(s for s in steps if "create-github-app-token" in s.get("uses", ""))
        assert mint["with"]["permission-contents"] == "read"
        assert mint["with"]["repositories"] == "${{ steps.overlay.outputs.name }}"
        assert not [
            k
            for k in mint["with"]
            if k.startswith("permission-") and k != "permission-contents"
        ]

    def test_an_unset_variable_a_bad_name_and_a_missing_file_fail_the_step(
        self,
    ) -> None:
        steps = self._action()["runs"]["steps"]
        resolve = next(s for s in steps if s.get("id") == "overlay")["run"]
        place = next(s for s in steps if s.get("id") == "place")["run"]
        assert 'if [ -z "${CI_OVERLAY_REPO}" ]' in resolve
        assert "OMNI_CI_OVERLAY_REPO is unset" in resolve
        assert "bare repository name" in resolve
        assert resolve.count("exit 1") == 2
        assert 'if [ ! -s "$src" ]' in place
        assert "exit 1" in place
        assert "continue-on-error" not in _ACTION.read_text(encoding="utf-8")
