# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The OMN-25106 contract binds AC2 to a reading it can fail on (OMN-25169, OMN-25191).

AC2 is the dev runtime answering ``/ready`` 200 after a redeploy, with a node
run line in its log written after the redeploy time. A curl that passes on any
code under 400, a boot-line read that a runtime which boots and then loses its
consumers also passes, a fixed ``--since 1h`` window, or unit tests that never
touch h201 cannot show that criterion's falsifier.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "contracts" / "OMN-25106.yaml"
CONTRACTS = (CONTRACT, REPO_ROOT / "contracts" / "OMN-25169.yaml")
HANDLER = (
    REPO_ROOT / "src" / "omnibase_infra" / "handlers" / "handler_dev_boot_guard.py"
)
DLQ_HANDLER = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_dlq_replay_effect"
    / "handlers"
    / "handler_dlq_replay.py"
)
BOOT_LINE = "ONEX runtime started successfully"
NODE_RUN_LINE = "DLQ replay run finished"
AGENT_TEST = "scripts/deploy-agent/tests/unit/test_wedged_restore_omn25106.py"


def _contract() -> dict[str, Any]:
    loaded = yaml.safe_load(CONTRACT.read_text())
    assert isinstance(loaded, dict)
    return loaded


def _checks_bound_to(ac: str) -> list[str]:
    return [
        check["check_value"]
        for entry in _contract()["dod_evidence"]
        if ac in entry["binds_ac"]
        for check in entry["checks"]
    ]


def _curl_arguments(check: str) -> list[str]:
    found = re.search(r"curl\s+([^)]*)\)", check)
    assert found, "no curl command in the check"
    return shlex.split(found.group(1))


def _curl_fails_on_http_error(arguments: list[str]) -> bool:
    """Whether any flag makes curl exit non-zero on an HTTP error status."""
    return any(
        token in ("--fail", "--fail-with-body")
        or (re.fullmatch(r"-[A-Za-z]+", token) is not None and "f" in token)
        for token in arguments
    )


def _logs_line(source: str, line: str) -> bool:
    return line in source


def _roll_step_probes() -> list[tuple[Path, str]]:
    """The python snippet of every contract probe that reads the roll step."""
    probes: list[tuple[Path, str]] = []
    for path in CONTRACTS:
        loaded = yaml.safe_load(path.read_text())
        for entry in loaded["dod_evidence"]:
            for check in entry["checks"]:
                value = check["check_value"]
                if "run_runtime_roll_step" in value:
                    found = re.search(r"python3 -c '(.*)'\s*$", value, re.DOTALL)
                    assert found, f"{path.name}: no python probe in the check"
                    probes.append((path, found.group(1)))
    return probes


def _run_probe(probe: str, source: str) -> int:
    return subprocess.run(
        [sys.executable, "-c", probe],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    ).returncode


def _without_the_roll_step_call(source: str) -> str:
    start = source.index("def run_runtime_roll_step")
    end = source.index("\ndef ", start + 1)
    body = source[start:end].replace("restart_if_wedged(", "no_restart(")
    assert body != source[start:end]
    return source[:start] + body + source[end:]


def test_ac2_is_bound_only_to_readings_not_to_unit_tests() -> None:
    checks = _checks_bound_to("AC2")

    assert checks, "AC2 is bound to no check"
    assert not [c for c in checks if "pytest" in c]


def test_the_ready_check_fails_on_any_code_but_200() -> None:
    ready = [c for c in _checks_bound_to("AC2") if "/ready" in c]

    assert len(ready) == 1
    # ``curl -sf`` passes on any status under 400; the code must be compared.
    assert not _curl_fails_on_http_error(_curl_arguments(ready[0]))
    assert "%{http_code}" in ready[0]
    assert "= 200" in ready[0]


@pytest.mark.parametrize("flags", ["-f", "-sf", "-fsS", "--fail", "--fail-with-body"])
def test_the_curl_flag_check_sees_every_fail_flag_spelling(flags: str) -> None:
    """Positive control: the case the comment names, ``-sf``, is caught."""
    assert _curl_fails_on_http_error([*flags.split(), "http://localhost/ready"])


def test_the_curl_flag_check_passes_the_flags_the_contract_uses() -> None:
    arguments = shlex.split("-s -o /dev/null -w '%{http_code}' http://localhost/ready")
    assert not _curl_fails_on_http_error(arguments)


def test_a_check_reads_a_node_run_line_after_the_redeploy_time() -> None:
    reads = [c for c in _checks_bound_to("AC2") if "docker logs" in c]

    assert len(reads) == 1
    assert NODE_RUN_LINE in reads[0]
    assert "grep -c" in reads[0]
    assert '--since "$REDEPLOY_AT"' in reads[0]
    # A fixed window is not "after the redeploy time".
    assert not re.search(r"--since\s+\d+[smhd]\b", reads[0])
    # The count must be compared: ``grep -c`` prints 0 and the test passes on it.
    assert "-ge 1" in reads[0]


def test_the_boot_line_is_not_what_ac2_reads() -> None:
    """A runtime that boots and then loses its consumers also logs the boot line."""
    assert not [c for c in _checks_bound_to("AC2") if BOOT_LINE in c]


def test_the_node_run_line_the_check_greps_is_still_logged_by_the_handler() -> None:
    """Removing the grep target's log line turns this red, not the check unreachable."""
    assert _logs_line(DLQ_HANDLER.read_text(), NODE_RUN_LINE)


def test_the_node_run_line_gate_goes_red_when_the_target_is_removed() -> None:
    source = DLQ_HANDLER.read_text()
    assert _logs_line(source, NODE_RUN_LINE), "positive control"

    assert not _logs_line(
        source.replace(NODE_RUN_LINE, "DLQ replay finished"), NODE_RUN_LINE
    )


def test_the_agent_restore_test_is_in_the_evidence_commands() -> None:
    commands = [r["command"] for r in _contract()["evidence_requirements"]]

    agent = [c for c in commands if AGENT_TEST in c]
    assert len(agent) == 1
    assert "--project scripts/deploy-agent" in agent[0]


def test_both_contracts_carry_a_roll_step_probe() -> None:
    assert sorted(path.name for path, _ in _roll_step_probes()) == [
        "OMN-25106.yaml",
        "OMN-25169.yaml",
    ]


@pytest.mark.parametrize("path", CONTRACTS, ids=lambda p: p.name)
def test_the_roll_step_probe_passes_on_the_real_handler(path: Path) -> None:
    """Positive control: the probe is satisfied by the committed source."""
    probe = next(p for owner, p in _roll_step_probes() if owner == path)

    assert _run_probe(probe, HANDLER.read_text()) == 0


@pytest.mark.parametrize("path", CONTRACTS, ids=lambda p: p.name)
def test_the_roll_step_probe_fails_when_the_call_is_deleted(path: Path) -> None:
    """The later ``def restart_if_wedged(`` once satisfied the probe with the call gone."""
    probe = next(p for owner, p in _roll_step_probes() if owner == path)
    mutated = _without_the_roll_step_call(HANDLER.read_text())
    # The old slice, to the end of the file, still holds the call text.
    assert "restart_if_wedged(" in mutated.split("def run_runtime_roll_step")[1]

    assert _run_probe(probe, mutated) != 0
