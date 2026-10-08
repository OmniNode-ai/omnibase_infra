# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Wiring of the omnibase_infra validator conversion batch (OMN-20568).

Each converted check runs as a node from the pre-commit config and from CI, and the
script it replaced is gone. This is the enforcement half of the conversion: a check
that is converted but not wired, or a replaced script that is still invoked, fails here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
NODES = "omnibase_infra.nodes"

# hook id -> the module the hook entry must run
PRE_COMMIT_NODE_ENTRIES = {
    "no-env-fallbacks": (
        "omnibase_core.nodes.node_no_env_fallbacks_check_compute"
        ".runtime_no_env_fallbacks_check"
    ),
    "onex-validate-migration-freeze": (
        f"{NODES}.node_migration_freeze_check_compute.runtime_migration_freeze_check"
    ),
    "onex-validate-migration-sequence": (
        f"{NODES}.node_migration_sequence_check_compute"
        ".runtime_migration_sequence_check"
    ),
    "onex-check-migration-append-only": (
        f"{NODES}.node_migration_append_only_check_compute"
        ".runtime_migration_append_only_check"
    ),
}

REPLACED_SCRIPTS = (
    "scripts/validate_no_env_fallbacks.py",
    "scripts/validation/validate_migration_freeze.py",
    "scripts/check_migration_freeze.sh",
    "scripts/validation/validate_migration_sequence.py",
    "scripts/validation/check_migration_append_only.py",
)


def _hooks() -> dict[str, dict[str, object]]:
    config = yaml.safe_load(
        (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    return {
        hook["id"]: hook for repo in config["repos"] for hook in repo.get("hooks", [])
    }


@pytest.mark.unit
@pytest.mark.parametrize(("hook_id", "module"), PRE_COMMIT_NODE_ENTRIES.items())
def test_converted_checks_run_as_nodes_from_pre_commit(
    hook_id: str, module: str
) -> None:
    entry = str(_hooks()[hook_id]["entry"])
    assert f"python -m {module}" in entry


@pytest.mark.unit
@pytest.mark.parametrize("script", REPLACED_SCRIPTS)
def test_replaced_scripts_are_deleted(script: str) -> None:
    assert not (REPO_ROOT / script).exists()


@pytest.mark.unit
def test_no_pre_commit_or_ci_entry_invokes_a_replaced_script_or_validate_py_subcommand() -> (
    None
):
    config = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    for text in (config, workflow):
        for script in REPLACED_SCRIPTS:
            assert f"python {script}" not in text
            assert f"./{script}" not in text
        assert "validate.py migration_freeze" not in text
        assert "validate.py migration_sequence" not in text


@pytest.mark.unit
def test_ci_runs_the_same_node_entries() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "uv run pre-commit run no-env-fallbacks --all-files" in workflow
    assert "runtime_migration_freeze_check --base" in workflow
    assert "runtime_migration_append_only_check --base" in workflow
