# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The DoD checks of contracts/OMN-24854.yaml can fail (OMN-24854).

Mount checks name one compose file each and fail when its mount source changes.
Lab observations have separate checks that fail without recorded evidence; a
synthetic run record exercises their positive and negative controls in isolation.
"""

from __future__ import annotations

import shlex
import shutil
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


def _contract() -> dict[str, Any]:
    loaded = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _checks() -> list[dict[str, str]]:
    return [check for item in _contract()["dod_evidence"] for check in item["checks"]]


def _mount_grep_checks() -> list[list[str]]:
    return [
        shlex.split(check["check_value"])
        for check in _checks()
        if check["check_value"].startswith("grep -q ")
    ]


def test_the_evidence_command_runs_both_tests() -> None:
    commands = [r["command"] for r in _contract()["evidence_requirements"]]
    assert len(commands) == 1
    for path in EVIDENCE_TESTS:
        assert path in commands[0]
    test_checks = [c for c in _checks() if c["check_type"] == "test_passes"]
    assert len(test_checks) == 1
    assert test_checks[0]["check_value"] == commands[0]


def test_no_check_passes_whatever_the_lab_shows() -> None:
    for check in _checks():
        assert not check["check_value"].startswith("test -f"), check
    declared = {
        ac["id"]
        for requirement in _contract()["requirements"]
        for ac in requirement["acceptance"]
    }
    bound = {ac for item in _contract()["dod_evidence"] for ac in item["binds_ac"]}
    # Retain the ticket's lab criteria. Their checks must fail until real
    # observations are recorded; only AC4 is established by the mount tests.
    assert declared == bound
    assert {"AC1", "AC2", "AC3", "AC4"} <= declared


def test_lab_observation_checks_refuse_a_tree_without_h201_evidence(
    tmp_path: Path,
) -> None:
    contract = _contract()
    lab = next(
        ac
        for req in contract["requirements"]
        for ac in req["acceptance"]
        if ac["id"] == "AC1"
    )
    assert "h201" in lab["statement"]
    checks = [
        c
        for item in contract["dod_evidence"]
        if "AC1" in item["binds_ac"]
        for c in item["checks"]
    ]
    assert checks
    for check in checks:
        assert check["check_type"] == "command", (
            "unit tests cannot establish a live h201 run"
        )
        argv = shlex.split(check["check_value"])
        result = subprocess.run(
            argv, cwd=tmp_path, check=False, capture_output=True, text=True
        )
        assert result.returncode != 0
        target = tmp_path / argv[-1]
        target.parent.mkdir(parents=True, exist_ok=True)
        line = (
            "Lab: host=h201 revision="
            + "a" * 40
            + " read_at=2026-11-16T23:00:00Z nodes="
            "node_commitment_check_runner_effect,node_commitment_ledger_compute,"
            "node_commitment_writer_effect,node_work_state_projection_compute,"
            "node_work_state_projection_effect observed=handled\n"
        )
        target.write_text(line, encoding="utf-8")
        assert subprocess.run(argv, cwd=tmp_path, check=False).returncode == 0
        for wrong in (
            line.replace("h201", "h202"),
            line.replace("node_commitment_writer_effect", "unhandled_node"),
        ):
            target.write_text(wrong, encoding="utf-8")
            assert subprocess.run(argv, cwd=tmp_path, check=False).returncode != 0


def test_every_mount_grep_check_names_exactly_one_file() -> None:
    greps = _mount_grep_checks()
    assert len(greps) == 2
    for argv in greps:
        assert argv[:2] == ["grep", "-q"], argv
        assert len(argv) == 4, argv


def test_mount_grep_checks_pass_here_and_fail_for_a_different_source(
    tmp_path: Path,
) -> None:
    for argv in _mount_grep_checks():
        pattern, target = argv[2], argv[3]
        here = subprocess.run(argv, cwd=REPO_ROOT, check=False)
        assert here.returncode == 0, argv
        copy = tmp_path / Path(target).name
        shutil.copyfile(REPO_ROOT / target, copy)
        # Mutate the mount's source while preserving comments and all other
        # content. This checks the deployment property, not deletion of the
        # grep's own pattern from a copy.
        copy.write_text(
            "\n".join(
                line.replace("ONEX_LANE_DECLARATION_DIR", "WRONG_DECLARATION_DIR")
                if ":/etc/onex/workspace-config/config/lab:ro" in line
                and not line.lstrip().startswith("#")
                else line
                for line in copy.read_text(encoding="utf-8").splitlines()
            )
            + "\n",
            encoding="utf-8",
        )
        gone = subprocess.run(
            ["grep", "-q", pattern, copy.name], cwd=tmp_path, check=False
        )
        assert gone.returncode == 1, argv
