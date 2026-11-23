# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dev roll is good only when the work ledger answers inside its declared bound (OMN-25942).

Failure first: the dev boot guard judged a roll by Docker health and the runtime health
ports, so a dev runtime that answered both and could not answer a ledger read ended
``success`` and kept its image. The command line and the bound are read from the
``post_roll_proof`` declaration of the deploy-source clone, never typed in the guard.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from deploy_agent import executor as executor_mod
from deploy_agent.executor import DeployExecutor

from tests.unit import test_dev_boot_guard_omn21279 as boot_guard_tests

pytestmark = [pytest.mark.unit, pytest.mark.dev_boot_guard]

docker = boot_guard_tests.docker

CONTRACT = (
    Path(__file__).resolve().parents[4]
    / "src/omnibase_infra/contracts/work_ledger_query_client.contract.yaml"
)
DECLARED = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))["post_roll_proof"]
JOB = "0f6a2b1c-2fd2-4c52-8a02-25942f2f2f2f"
COMMIT = boot_guard_tests.GOOD_COMMIT


def _roll(
    tmp_path: Path,
) -> tuple[boot_guard_tests.guard_mod.HandlerDevBootGuard, Mock]:
    executor = DeployExecutor()
    executor.verify = Mock(return_value=[])
    guard = executor_mod.DevBootGuard(executor, tmp_path)
    guard.prepare()
    return guard, Mock()


def _ledger_commands(docker: dict) -> list[list[str]]:
    return [argv for argv in docker["commands"] if argv[0].endswith("/scripts/onex")]


def test_declaration_is_the_ticketed_command_and_bound() -> None:
    assert DECLARED == {
        "argv": ["ledger", "query", "--query", "rows", "--limit", "1"],
        "timeout_s": 10,
    }


def test_a_ledger_that_does_not_answer_in_time_fails_the_roll_and_restores(
    tmp_path: Path, docker: dict
) -> None:
    guard, mark_failed = _roll(tmp_path)
    docker["ledger_raises"] = subprocess.TimeoutExpired(
        cmd="onex", timeout=DECLARED["timeout_s"]
    )
    with pytest.raises(boot_guard_tests.guard_mod.DevBootRecoveryError):
        with guard.protect(JOB, boot_guard_tests.BAD_COMMIT, mark_failed):
            # The roll replaced the runtime: healthy, on another image.
            docker["containers"] = [
                {**container, "Image": "sha256:" + "9" * 64}
                for container in docker["containers"]
            ]
    mark_failed.assert_called_once()
    assert f"work ledger: no answer within {DECLARED['timeout_s']}s" in str(
        mark_failed.call_args
    )
    up = next(argv for argv in docker["commands"] if "up" in argv)
    recovered = json.loads(Path(up[up.index("-f") + 1]).read_text())
    assert all("boot-good" in s["image"] for s in recovered["services"].values()), (
        "the previous image is kept"
    )
    assert len(_ledger_commands(docker)) == 1, "recovery is not asked for the proof"


def test_a_ledger_that_answers_keeps_the_roll_and_runs_the_declared_command(
    tmp_path: Path, docker: dict
) -> None:
    guard, mark_failed = _roll(tmp_path)
    commands_before = len(docker["commands"])
    with guard.protect(JOB, COMMIT, mark_failed):
        pass
    mark_failed.assert_not_called()
    assert not any("up" in argv for argv in docker["commands"][commands_before:])
    (command,) = _ledger_commands(docker)
    assert command == [
        f"{executor_mod.REPO_DIR}/scripts/onex",
        *DECLARED["argv"],
        "--timeout",
        str(DECLARED["timeout_s"]),
    ]
    assert guard.point.current is not None and guard.point.current.revision == COMMIT


@pytest.mark.parametrize(
    ("ledger", "ledger_exit", "cause"),
    [
        ("", 1, "exited 1"),
        ("not json", 0, "not one JSON object"),
        ("[]", 0, "not one JSON object"),
    ],
)
def test_a_command_that_is_not_an_answer_fails_the_roll(
    tmp_path: Path, docker: dict, ledger: str, ledger_exit: int, cause: str
) -> None:
    guard, mark_failed = _roll(tmp_path)
    docker["ledger"], docker["ledger_exit"] = ledger, ledger_exit
    with pytest.raises(boot_guard_tests.guard_mod.DevBootRecoveryError):
        with guard.protect(JOB, COMMIT, mark_failed):
            pass
    assert cause in str(mark_failed.call_args)


def test_a_command_that_cannot_start_fails_the_roll(
    tmp_path: Path, docker: dict
) -> None:
    guard, mark_failed = _roll(tmp_path)
    docker["ledger_raises"] = FileNotFoundError("onex")
    with pytest.raises(boot_guard_tests.guard_mod.DevBootRecoveryError):
        with guard.protect(JOB, COMMIT, mark_failed):
            pass
    assert "could not start" in str(mark_failed.call_args)


def test_the_command_and_bound_come_from_the_declaration_not_the_guard(
    tmp_path: Path, docker: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = tmp_path / "clone"
    declaration = clone / boot_guard_tests.guard_mod.WORK_LEDGER_CONTRACT
    declaration.parent.mkdir(parents=True)
    declaration.write_text(
        yaml.safe_dump(
            {
                "post_roll_proof": {
                    "argv": ["ledger", "query", "--query", "holds"],
                    "timeout_s": 3,
                }
            }
        )
    )
    guard, mark_failed = _roll(tmp_path / "state")
    monkeypatch.setattr(executor_mod, "REPO_DIR", str(clone))
    with guard.protect(JOB, COMMIT, mark_failed):
        pass
    (command,) = _ledger_commands(docker)
    assert command == [
        f"{clone}/scripts/onex",
        "ledger",
        "query",
        "--query",
        "holds",
        "--timeout",
        "3",
    ]


def test_a_failed_roll_is_not_asked_for_a_ledger_proof(
    tmp_path: Path, docker: dict
) -> None:
    guard, mark_failed = _roll(tmp_path)
    with pytest.raises(boot_guard_tests.guard_mod.DevBootRecoveryError):
        with guard.protect(JOB, COMMIT, mark_failed):
            docker["containers"] = boot_guard_tests._containers(healthy=False)
            raise RuntimeError("compose up failed")
    assert _ledger_commands(docker) == []
