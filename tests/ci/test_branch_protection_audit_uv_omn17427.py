# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Keep the hosted branch-protection audit runnable (OMN-17427).

omnibase_infra#5066 (27ccd1b6f) changed the runs-on fallback to ubuntu-latest.
With no repository Actions variables set, the job uses that hosted image, which
ships no uv. Run 39492835153, job 118870175515, failed at the required-context
parity report with "uv: command not found" (exit 127). Install uv before use and
run the report with only its locked PyYAML dependency, avoiding project sync.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "branch-protection-audit.yml"


def _audit_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    steps: list[dict[str, Any]] = workflow["jobs"]["audit"]["steps"]
    return steps


def test_setup_uv_is_pinned_to_a_full_sha() -> None:
    assert any(
        re.fullmatch(r"astral-sh/setup-uv@[0-9a-fA-F]{40}", step.get("uses", ""))
        for step in _audit_steps()
    ), "The audit must install uv with a setup-uv action pinned to a full SHA"


def test_setup_uv_precedes_every_uv_run() -> None:
    steps = _audit_steps()
    setup_indices = [
        index
        for index, step in enumerate(steps)
        if step.get("uses", "").startswith("astral-sh/setup-uv@")
    ]
    assert setup_indices, "The audit must install uv before running it"
    for index, step in enumerate(steps):
        if "uv run" in step.get("run", ""):
            assert min(setup_indices) < index, step.get("name", "<unnamed>")


def test_every_uv_run_skips_project_sync() -> None:
    for step in _audit_steps():
        run = step.get("run", "")
        for match in re.finditer(r"\buv run\b", run):
            assert re.match(r" --no-project(?:\s|$)", run[match.end() :]), (
                f"{step.get('name', '<unnamed>')} must avoid syncing the project"
            )


def test_parity_report_uses_the_locked_pyyaml_version() -> None:
    lock = tomllib.loads((_REPO_ROOT / "uv.lock").read_text())
    version = next(
        package["version"] for package in lock["package"] if package["name"] == "pyyaml"
    )
    parity_step = next(
        step
        for step in _audit_steps()
        if step.get("name") == "Required-context parity report (REPORT-ONLY)"
    )
    match = re.search(r"(?:^|\s)--with\s+pyyaml==([^\s]+)(?:\s|$)", parity_step["run"])
    assert match is not None, "The parity report must explicitly install pinned PyYAML"
    assert match.group(1) == version


@pytest.mark.parametrize(
    "filename",
    ["audit_required_context_parity_cli.py", "audit_required_context_parity_lib.py"],
)
def test_parity_imports_need_only_pyyaml(filename: str) -> None:
    tree = ast.parse((_REPO_ROOT / "scripts" / filename).read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module.split(".")[0])

    allowed = sys.stdlib_module_names | {
        "__future__",
        "yaml",
        "audit_required_context_parity_lib",
    }
    assert not (unexpected := imports - allowed), (
        f"{filename} adds dependencies missing from the isolated report: {sorted(unexpected)}"
    )
