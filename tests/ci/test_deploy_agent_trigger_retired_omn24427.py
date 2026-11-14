# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The hand-run deploy agent trigger is retired (OMN-24427).

The trigger script read a signing key from whichever machine ran it, so a command
signed with a key that differed from the agent's was published and silently dropped.
The one sender is now ``node_branch_deploy_command_effect`` (omnimarket), which
refuses loudly when the key is missing. This file fails while the script, its python
module or any reference that tells a reader to run them is still in the tree.

Failing control: ``test_the_reference_scan_finds_a_planted_reference`` runs the same
scan over a tree that carries one, so a scan that matches nothing cannot pass.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

REPO_ROOT = Path(__file__).resolve().parents[2]

# What the retirement removed. Built from parts so this file is not its own match.
SCRIPT_NAME = "deploy-agent" + "-trigger"
MODULE_NAME = "deploy_agent" + ".trigger"
RETIRED_PATHS = (
    "scripts/deploy-agent/" + SCRIPT_NAME + ".sh",
    "scripts/deploy-agent/deploy_agent/trigger.py",
)
# Decision records and ticket contracts keep the history of why the trigger existed and
# what retired it; they may name it.
DECISION_TEXT_DIRS = ("docs/decisions/", "contracts/")
SKIPPED_SUFFIXES = (".lock",)


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout
    return [name for name in out.split("\0") if name]


def references_to_retired_trigger(root: Path, names: list[str]) -> list[str]:
    """Return ``path:line`` for each tracked file outside decision text that names the trigger."""
    found: list[str] = []
    for name in names:
        if name.startswith(DECISION_TEXT_DIRS) or name.endswith(SKIPPED_SUFFIXES):
            continue
        if name == "tests/ci/test_deploy_agent_trigger_retired_omn24427.py":
            continue
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if SCRIPT_NAME in line or MODULE_NAME in line:
                found.append(f"{name}:{number}")
    return found


@pytest.mark.unit
@pytest.mark.parametrize("relative", RETIRED_PATHS)
def test_the_retired_trigger_files_are_gone(relative: str) -> None:
    assert not (REPO_ROOT / relative).exists(), f"{relative} is retired (OMN-24427)"


@pytest.mark.unit
def test_no_tracked_file_outside_decision_text_names_the_trigger() -> None:
    found = references_to_retired_trigger(REPO_ROOT, tracked_files(REPO_ROOT))
    assert found == [], (
        "these lines still name the retired trigger; send the deploy command through "
        f"node_branch_deploy_command_effect instead: {found}"
    )


@pytest.mark.unit
def test_the_reference_scan_finds_a_planted_reference(tmp_path: Path) -> None:
    """Control: the scan reports a reference where one exists and skips decision text."""
    (tmp_path / "docs" / "decisions").mkdir(parents=True)
    (tmp_path / "docs" / "decisions" / "why.yaml").write_text(f"note: {SCRIPT_NAME}\n")
    (tmp_path / "runbook.md").write_text(f"run\n{MODULE_NAME} --dry-run\n")
    names = ["docs/decisions/why.yaml", "runbook.md"]

    assert references_to_retired_trigger(tmp_path, names) == ["runbook.md:2"]


@pytest.mark.unit
def test_the_file_shape_baseline_lists_no_retired_path() -> None:
    baseline = (
        REPO_ROOT / ".onex_ratchets" / "canonical_file_shape_baseline.txt"
    ).read_text()
    listed = {line.strip() for line in baseline.splitlines()}
    assert not listed.intersection(RETIRED_PATHS)
