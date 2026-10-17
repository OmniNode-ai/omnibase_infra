# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: the strict contract load runs at commit, not only in a CI test split.

OMN-20945 pinned every ``contracts/*.yaml`` to one value per mapping key, but only
inside the full test suite, so a merge that resolves an adjacent-addition conflict
in a ticket contract by keeping both sides' ``description`` and ``command`` lines
under one ``- kind: tests`` item still reached CI. On 2026-10-16 three PRs
(OMN-19998) carried that defect and each went red on the Test-Failure Ratchet Gate
and repo-evidence / dod-verify. The ``contracts-strict-load`` hook runs the same
test on any commit that touches a contract, a merge commit's conflicted
contract included.

Falsifier: drop the hook, take ``contracts/`` out of its ``files`` pattern, or
point its entry at another test, and the named test fails.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PRECOMMIT = REPO_ROOT / ".pre-commit-config.yaml"
HOOK_ID = "contracts-strict-load"
STRICT_LOAD_TEST = "tests/ci/test_contracts_strict_load_omn20945.py"


def _hook() -> dict[str, Any]:
    config = yaml.safe_load(PRECOMMIT.read_text(encoding="utf-8"))
    hooks = [
        hook
        for repo in config["repos"]
        for hook in repo.get("hooks", [])
        if hook.get("id") == HOOK_ID
    ]
    assert len(hooks) == 1, f"exactly one {HOOK_ID} hook is declared"
    return hooks[0]


def test_hook_runs_the_strict_load_test_at_commit() -> None:
    hook = _hook()
    assert hook["stages"] == ["pre-commit"]
    assert hook["pass_filenames"] is False
    assert STRICT_LOAD_TEST in hook["entry"].split()
    assert (REPO_ROOT / STRICT_LOAD_TEST).is_file()


@pytest.mark.parametrize(
    "path",
    [
        "contracts/OMN-19998.yaml",
        "contracts/OMN-17427.yaml",
        STRICT_LOAD_TEST,
        ".pre-commit-config.yaml",
    ],
)
def test_hook_fires_on_a_contract_change(path: str) -> None:
    assert re.search(_hook()["files"], path), f"{HOOK_ID} does not fire on {path}"


@pytest.mark.parametrize(
    "path",
    ["src/omnibase_infra/nodes/foo/contract.yaml", "contracts/sub/OMN-1.yaml"],
)
def test_hook_does_not_fire_off_the_ticket_contracts(path: str) -> None:
    assert not re.search(_hook()["files"], path)
