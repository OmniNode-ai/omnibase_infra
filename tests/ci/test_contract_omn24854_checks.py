# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The DoD checks of contracts/OMN-24854.yaml can fail, and its lab criteria stay open (OMN-24854).

The ticket's AC1 to AC3 are lab observations on the h201 dev runtime (the run line, the
STATUS row, the integration pass line). No repository check can both pass on the hosted
runner and read the lab, so the contract binds none of them: a check that passes whatever
the lab shows would mark them met while the dev runtime still lacks the declaration folder.
What it does bind is the mount source (AC4), with checks that can fail on the defect they
guard: a test run and one anchored grep per compose file for the mount line itself.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "contracts" / "OMN-24854.yaml"
EVIDENCE_TESTS = (
    "tests/ci/test_dev_lane_ledger_source_binding.py",
    "scripts/deploy-agent/tests/unit/test_llm_endpoint_env_compose_omn24025.py",
)
LAB_CRITERIA = {"AC1", "AC2", "AC3"}
MOUNT_TARGET = ":/etc/onex/workspace-config/config/lab:ro"
PRE_FIX_MOUNT = "- ${ONEX_LANE_OVERLAY_DIR}/.." + MOUNT_TARGET


def _contract() -> dict[str, Any]:
    loaded = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _checks() -> list[dict[str, str]]:
    return [check for item in _contract()["dod_evidence"] for check in item["checks"]]


def _grep_checks() -> list[list[str]]:
    return [
        shlex.split(check["check_value"])
        for check in _checks()
        if check["check_value"].startswith("grep ")
    ]


def _can_fail(check: dict[str, str]) -> bool:
    """Whether ``check`` reads the repository tree: a test run, or an anchored grep of one file.

    ``test -f``, ``true`` and the like pass whatever the tree or the lab shows.
    """
    value = check["check_value"]
    if check["check_type"] == "test_passes":
        return any(path in value for path in EVIDENCE_TESTS)
    argv = shlex.split(value)
    return (
        argv[:2] == ["grep", "-qE"]
        and len(argv) == 4
        and argv[2].startswith("^")
        and argv[2].endswith("$")
        and (REPO_ROOT / argv[3]).is_file()
    )


def test_the_evidence_command_runs_both_tests() -> None:
    commands = [r["command"] for r in _contract()["evidence_requirements"]]
    assert len(commands) == 1
    for path in EVIDENCE_TESTS:
        assert path in commands[0]
    test_checks = [c for c in _checks() if c["check_type"] == "test_passes"]
    assert len(test_checks) == 1
    assert test_checks[0]["check_value"] == commands[0]


def test_every_declared_check_can_fail() -> None:
    assert _checks()
    for check in _checks():
        assert _can_fail(check), check


@pytest.mark.parametrize(
    "check",
    [
        {
            "check_type": "command",
            "check_value": "test -f docker/docker-compose.dev-lane.yml",
        },
        {"check_type": "command", "check_value": "true"},
        {
            "check_type": "command",
            "check_value": "grep -q ONEX_LANE_DECLARATION_DIR:? docker/docker-compose.dev-lane.yml",
        },
        {"check_type": "test_passes", "check_value": "uv run pytest tests/unit -q"},
    ],
)
def test_control_checks_that_pass_whatever_the_tree_or_the_lab_shows_are_refused(
    check: dict[str, str],
) -> None:
    assert not _can_fail(check)


def test_the_lab_criteria_are_left_open_not_bound_to_a_check_that_passes_regardless() -> (
    None
):
    declared = {
        ac["id"]
        for requirement in _contract()["requirements"]
        for ac in requirement["acceptance"]
    }
    bound = {ac for item in _contract()["dod_evidence"] for ac in item["binds_ac"]}

    assert declared == bound == {"AC4"}
    assert not declared & LAB_CRITERIA


def test_every_grep_check_names_one_file_and_matches_the_mount_line_there() -> None:
    greps = _grep_checks()
    assert len(greps) == 2
    assert {argv[3] for argv in greps} == {
        "docker/docker-compose.dev-lane.yml",
        "docker/docker-compose.dev-tenant-fold.yml",
    }
    for argv in greps:
        here = subprocess.run(argv, cwd=REPO_ROOT, check=False)
        assert here.returncode == 0, argv


@pytest.mark.parametrize(
    "damage",
    ["revert_to_overlay_parent", "comment_out", "drop_the_required_form"],
)
def test_every_grep_check_goes_red_when_its_mount_line_is_damaged(
    tmp_path: Path, damage: str
) -> None:
    mount = re.compile(
        r"^(\s+)- \$\{ONEX_LANE_DECLARATION_DIR:\?[^}]*\}"
        + re.escape(MOUNT_TARGET)
        + "$"
    )
    for argv in _grep_checks():
        lines = (REPO_ROOT / argv[3]).read_text(encoding="utf-8").splitlines()
        hits = [i for i, line in enumerate(lines) if mount.match(line)]
        assert len(hits) == 1, argv
        indent = lines[hits[0]][: len(lines[hits[0]]) - len(lines[hits[0]].lstrip())]
        lines[hits[0]] = {
            "revert_to_overlay_parent": f"{indent}{PRE_FIX_MOUNT}",
            "comment_out": f"{indent}# {lines[hits[0]].strip()}",
            "drop_the_required_form": f"{indent}- ${{ONEX_LANE_DECLARATION_DIR:-x}}{MOUNT_TARGET}",
        }[damage]
        copy = tmp_path / Path(argv[3]).name
        copy.write_text("\n".join(lines) + "\n", encoding="utf-8")

        gone = subprocess.run([*argv[:3], copy.name], cwd=tmp_path, check=False)

        assert gone.returncode == 1, (damage, argv)
