# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for scripts/validate-pr-contract-sync.sh [OMN-8915].

TDD-first: these tests were written before the script existed.
Each test invokes the script with a --files argument (newline-separated list
of changed file paths) so no live gh CLI is needed in the test suite.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parent.parent.parent / "scripts" / "validate-pr-contract-sync.sh"
)


def run_script(
    changed_files: list[str], commit_messages: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Invoke the gate script with a synthetic file list.

    Args:
        changed_files: Paths as they would appear in `gh pr diff --name-only`.
        commit_messages: Commit messages to simulate [skip-contract-sync] exemption.
    """
    file_input = "\n".join(changed_files)
    env_overrides = {"VALIDATE_PR_FILES": file_input}
    if commit_messages is not None:
        env_overrides["VALIDATE_PR_COMMIT_MESSAGES"] = "\n".join(commit_messages)

    import os

    env = {**os.environ, **env_overrides}

    return subprocess.run(
        ["bash", str(SCRIPT), "--from-env"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.mark.unit
def test_handler_change_with_contract_change_passes():
    """Handler change accompanied by contract.yaml change must PASS."""
    result = run_script(
        [
            "src/omnibase_infra/nodes/node_registration_orchestrator/handlers/handler_register.py",
            "src/omnibase_infra/nodes/node_registration_orchestrator/contract.yaml",
        ]
    )
    assert result.returncode == 0, (
        f"Expected PASS.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.unit
def test_handler_change_without_contract_change_fails():
    """Handler change without a matching contract.yaml change must FAIL with structured message."""
    result = run_script(
        [
            "src/omnibase_infra/nodes/node_registration_orchestrator/handlers/handler_register.py",
        ]
    )
    assert result.returncode != 0, (
        f"Expected FAIL.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert (
        "node_registration_orchestrator" in result.stdout
        or "node_registration_orchestrator" in result.stderr
    )
    assert "contract.yaml" in result.stdout or "contract.yaml" in result.stderr


@pytest.mark.unit
def test_skill_prompt_change_with_skill_md_passes():
    """Skill prompt.md change accompanied by SKILL.md change must PASS."""
    result = run_script(
        [
            "plugins/onex/skills/handoff/prompt.md",
            "plugins/onex/skills/handoff/SKILL.md",
        ]
    )
    assert result.returncode == 0, (
        f"Expected PASS.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.unit
def test_skill_prompt_change_without_skill_md_fails():
    """Skill prompt.md change without matching SKILL.md change must FAIL."""
    result = run_script(
        [
            "plugins/onex/skills/handoff/prompt.md",
        ]
    )
    assert result.returncode != 0, (
        f"Expected FAIL.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "handoff" in result.stdout or "handoff" in result.stderr
    assert (
        "SKILL.md" in result.stdout
        or "contract.yaml" in result.stdout
        or "SKILL.md" in result.stderr
        or "contract.yaml" in result.stderr
    )


@pytest.mark.unit
def test_skip_token_in_commit_passes_without_contract():
    """[skip-contract-sync] in commit message must exempt the PR even without contract change."""
    result = run_script(
        changed_files=[
            "src/omnibase_infra/nodes/node_registration_orchestrator/handlers/handler_register.py",
        ],
        commit_messages=[
            "fix(register): typo fix [skip-contract-sync] no contract drift"
        ],
    )
    assert result.returncode == 0, (
        f"Expected PASS with skip token.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.unit
def test_skill_py_change_without_skill_md_fails():
    """Skill .py file change without matching SKILL.md must FAIL."""
    result = run_script(
        [
            "plugins/onex/skills/handoff/skill_handoff.py",
        ]
    )
    assert result.returncode != 0, (
        f"Expected FAIL.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.unit
def test_unrelated_file_change_passes():
    """Changes to non-handler, non-skill files must not trigger the gate."""
    result = run_script(
        [
            "docs/plans/some-plan.md",
            "pyproject.toml",
            "README.md",
        ]
    )
    assert result.returncode == 0, (
        f"Expected PASS for unrelated files.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


@pytest.mark.unit
def test_skill_contract_yaml_alternative_passes():
    """Skill change accompanied by skill-level contract.yaml (not SKILL.md) must also PASS."""
    result = run_script(
        [
            "plugins/onex/skills/handoff/prompt.md",
            "plugins/onex/skills/handoff/contract.yaml",
        ]
    )
    assert result.returncode == 0, (
        f"Expected PASS with contract.yaml alternative.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
