# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The core dependency-pin check is a CI Summary member here (OMN-22360).

This repository's ``CI Summary`` is a no-``needs`` poller, so a job is a member
when ``scripts/ci/ci_summary_gate.py`` registers its context in
``STRICT_GATE_JOBS``: a skipped or absent job then fails closed instead of
reading green.
"""

from __future__ import annotations

import re
import shlex
from importlib.metadata import entry_points
from pathlib import Path

import pytest
import yaml

from scripts.ci.ci_summary_gate import STRICT_GATE_JOBS

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CORE = "OmniNode-ai/omnibase_core"
WORKFLOW = f"{CORE}/.github/workflows/dependency-pins.yml@"
CONTEXT = "dependency-pins / Dependency Pins"
# The one core hook block held at an earlier in-bound rev; see its comment in
# .pre-commit-config.yaml. The validator itself bounds how far back it may sit.
HELD_BACK_HOOK = "check-public-runner-placement"
VALIDATOR_ENTRY_POINT = (
    "omnibase_core.nodes.node_dependency_pins_check_effect."
    "runtime_dependency_pins_check:main"
)


def _jobs() -> dict[str, dict[str, object]]:
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    jobs = ci["jobs"]
    assert isinstance(jobs, dict)
    return jobs


def _core_hook_revs() -> dict[str, str]:
    """Rev of every omnibase_core hook, keyed by each hook id in its block."""
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    return {
        str(hook["id"]): str(entry["rev"])
        for entry in config["repos"]
        if str(entry.get("repo", "")).rstrip("/").endswith(CORE)
        for hook in entry["hooks"]
    }


def test_dependency_pins_job_calls_the_core_workflow() -> None:
    assert str(_jobs()["dependency-pins"]["uses"]).startswith(WORKFLOW)


def test_dependency_pins_job_uses_the_core_hook_tag() -> None:
    tag = str(_jobs()["dependency-pins"]["uses"]).removeprefix(WORKFLOW)
    revs = _core_hook_revs()
    assert HELD_BACK_HOOK in revs, f"no omnibase_core block holds {HELD_BACK_HOOK}"
    held_back = revs.pop(HELD_BACK_HOOK)
    # Only an immutable commit may sit off the tag: a branch or moving ref would
    # escape the release-count bound the required dependency-pins job applies.
    assert re.fullmatch(r"[0-9a-f]{40}", held_back), held_back
    assert held_back != tag
    assert set(revs.values()) == {tag}


def test_held_back_block_holds_only_the_placement_hook() -> None:
    """No other hook rides on the held-back rev."""
    config = yaml.safe_load(
        (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    blocks = [
        [str(hook["id"]) for hook in entry["hooks"]]
        for entry in config["repos"]
        if any(hook["id"] == HELD_BACK_HOOK for hook in entry.get("hooks", []))
    ]
    assert blocks == [[HELD_BACK_HOOK]]


def test_dependency_pins_job_is_unconditional() -> None:
    job = _jobs()["dependency-pins"]
    assert "if" not in job
    assert "needs" not in job


def test_ci_summary_requires_the_dependency_pins_context() -> None:
    assert CONTEXT in STRICT_GATE_JOBS


def test_ac8_is_bound_to_the_validator_itself() -> None:
    """AC8 is bound to a check that runs the validator over this tree.

    A pin or lock check that already passes on the merge base is carried
    evidence, not a check that fails against the control; only the validator
    fails there until the pin findings are fixed.
    """
    contract = yaml.safe_load(
        (ROOT / "contracts/OMN-22360.yaml").read_text(encoding="utf-8")
    )
    commands = [
        str(check["check_value"])
        for entry in contract["dod_evidence"]
        if "AC8" in entry.get("binds_ac", [])
        for check in entry["checks"]
    ]
    validator_commands = [
        command
        for command in commands
        if shlex.split(command)[-2:] == ["check-dependency-pins", "."]
        and shlex.split(command)[:2] == ["uv", "run"]
    ]
    assert validator_commands, commands
    # The command runs core's validator itself, not a same-named local script.
    scripts = {
        entry.name: entry.value for entry in entry_points(group="console_scripts")
    }
    assert scripts.get("check-dependency-pins") == VALIDATOR_ENTRY_POINT
