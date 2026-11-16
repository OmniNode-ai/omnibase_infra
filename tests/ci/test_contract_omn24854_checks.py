# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The DoD checks of contracts/OMN-24854.yaml can fail (OMN-24854).

A ``grep -c`` over two files exits 0 when only one matches, and ``test -f`` passes
whatever the lab shows. Every grep check names one file, passes at this tree, and goes
red when its target text is gone from that file. The evidence command runs both tests.
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


def _grep_checks() -> list[list[str]]:
    return [
        shlex.split(check["check_value"])
        for check in _checks()
        if check["check_value"].startswith("grep ")
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
    bound = {ac for item in _contract()["dod_evidence"] for ac in item["binds_ac"]}
    # AC2 and AC3 are lab observations: unmet until the STATUS row and the
    # SEEN RUNNING line exist, so no check binds them here.
    assert bound == {"AC1"}


def test_every_grep_check_names_exactly_one_file() -> None:
    greps = _grep_checks()
    assert len(greps) == 2
    for argv in greps:
        assert argv[:2] == ["grep", "-q"], argv
        assert len(argv) == 4, argv


def test_every_grep_check_passes_here_and_fails_when_its_target_is_gone(
    tmp_path: Path,
) -> None:
    for argv in _grep_checks():
        pattern, target = argv[2], argv[3]
        here = subprocess.run(argv, cwd=REPO_ROOT, check=False)
        assert here.returncode == 0, argv
        # Positive control: the same grep over a copy without the text must exit 1.
        copy = tmp_path / Path(target).name
        shutil.copyfile(REPO_ROOT / target, copy)
        copy.write_text(
            copy.read_text(encoding="utf-8").replace(pattern, "removed"),
            encoding="utf-8",
        )
        gone = subprocess.run(
            ["grep", "-q", pattern, copy.name], cwd=tmp_path, check=False
        )
        assert gone.returncode == 1, argv
