# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The post-roll ledger proof runs the operator command with the operator switch (OMN-26316).

Measured on dev job d7149818 (2026-11-25T23:27Z): omnibase_core 0.47.77 and later list the
internal ``ledger`` group only when ``ONEX_CLI_SHOW_ALL_COMMANDS`` is set, so the proof
exited 2 ("No such command 'ledger'") on a healthy new runtime and the boot guard
restored the previous image. The variable belongs to that one call, never the process.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from deploy_agent import executor as executor_mod
from deploy_agent.executor import DeployExecutor

from tests.unit import test_dev_boot_guard_omn21279 as boot_guard_tests

pytestmark = [pytest.mark.unit, pytest.mark.dev_boot_guard]

docker = boot_guard_tests.docker

SWITCH = "ONEX_CLI_SHOW_ALL_COMMANDS=1"
JOB = "5d0f1c7e-26316-4a0e-9a77-0f6a2b1c2fd2"


def _ledger_raw(docker: dict) -> list[list[str]]:
    return [
        raw
        for raw in docker["raw_commands"]
        if any(part.endswith("/scripts/onex") for part in raw)
    ]


def test_ledger_proof_carries_the_operator_switch_on_that_call_only(
    tmp_path: Path, docker: dict
) -> None:
    executor = DeployExecutor()
    executor.verify = Mock(return_value=[])
    guard = executor_mod.DevBootGuard(executor, tmp_path)
    guard.prepare()
    with guard.protect(JOB, boot_guard_tests.GOOD_COMMIT, Mock()):
        pass
    (raw,) = _ledger_raw(docker)
    entry = raw.index(f"{executor_mod.deploy_repo_dir()}/scripts/onex")
    assert raw[0] == "env"
    assert SWITCH in raw[:entry], "the switch is set for the ledger call"
    others = [r for r in docker["raw_commands"] if r is not raw]
    assert others and not any(SWITCH in r for r in others), (
        "no other command carries the switch"
    )
