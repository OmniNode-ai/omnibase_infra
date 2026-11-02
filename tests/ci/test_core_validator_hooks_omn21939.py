# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21939: validators this repo used to copy now run from omnibase_core.

Four hook ids that ran a repository script now resolve to the omnibase_core
hook repository, and their scripts are gone. Two copies stay, each for a rule
the core node does not yet carry:

* ``skip-count-ratchet``: the core ``--selftest`` validates the baseline next to
  its own installed package, never the consumer's ``config/skip_count_baseline.yaml``.
* ``integration-skip-guard``: the core node ignores ``require_each_curated_module``,
  which ``scripts/ci/runtime_boot_suites_skip_guard.yaml`` depends on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_HOOK_REPO = "https://github.com/OmniNode-ai/omnibase_core"

CONSUMED_FROM_CORE = {
    "exposed-identifier-gate": "scripts/validation/check_exposed_identifiers.py",
    "check-release-identity": "scripts/check_release_identity.py",
    "check-ai-slop": "scripts/validation/check_ai_slop.py",
    "precommit-interpreter-resolution": (
        "scripts/validation/validate_precommit_interpreter.py"
    ),
}
KEPT_LOCAL = {
    "skip-count-ratchet": "scripts/ci/skip_count_ratchet.py",
    "integration-skip-guard": "scripts/ci/check_integration_skips.py",
}

pytestmark = pytest.mark.unit


def _hook_repos() -> dict[str, list[str]]:
    config = yaml.safe_load(
        (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    found: dict[str, list[str]] = {}
    for repo in config["repos"]:
        for hook in repo.get("hooks") or []:
            found.setdefault(hook["id"], []).append(str(repo["repo"]))
    return found


@pytest.mark.parametrize("hook_id", sorted(CONSUMED_FROM_CORE))
def test_hook_id_resolves_to_the_core_hook_repo(hook_id: str) -> None:
    repos = _hook_repos().get(hook_id)
    assert repos == [CORE_HOOK_REPO], (
        f"{hook_id} must be declared exactly once, under {CORE_HOOK_REPO}; got {repos}"
    )


@pytest.mark.parametrize("hook_id", sorted(CONSUMED_FROM_CORE))
def test_the_deleted_copy_is_not_back(hook_id: str) -> None:
    script = CONSUMED_FROM_CORE[hook_id]
    assert not (REPO_ROOT / script).exists(), f"{script} is a copy of a core hook"
    config_text = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert script not in config_text


@pytest.mark.parametrize("hook_id", sorted(KEPT_LOCAL))
def test_a_kept_copy_is_still_wired(hook_id: str) -> None:
    """Positive control: the parser sees local hooks, so the checks above can fail."""
    assert _hook_repos().get(hook_id) == ["local"]
    assert (REPO_ROOT / KEPT_LOCAL[hook_id]).is_file()
