# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The OMN-25106 contract binds only what this change evidences, and its roll-step probe can fail (OMN-25169, OMN-25191).

AC1, AC2 and AC4 are open criteria no check at this head can read, so the
contract does not bind them; the roll-step probe must fail when the restart call
is deleted from the roll step.
"""

from __future__ import annotations

import re
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
AGENT_TEST = "scripts/deploy-agent/tests/unit/test_wedged_restore_omn25106.py"


def _contract() -> dict[str, Any]:
    loaded = yaml.safe_load(CONTRACT.read_text())
    assert isinstance(loaded, dict)
    return loaded


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
