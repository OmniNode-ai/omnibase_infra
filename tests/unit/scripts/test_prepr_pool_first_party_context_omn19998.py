# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pool prover's image build names the first-party audit context (OMN-19998).

Since omnibase_infra#5472 the runtime image audits where each first-party package
came from, and the audit runs in the build context named by the
FIRST_PARTY_BUILD_CONTEXT build arg, else in dev-lane, which refuses the target
(omnibase-infra) built from a branch. The pool prover builds the test merge of an
unmerged pull request, so every build it made since then failed the audit at
"FAIL omnibase-infra: branch (...) staged from branch dev" and every lab proof
read INCONCLUSIVE. ``prepr_verify_lane.sh`` passes the pre-PR slot context; the
isolated pool member's build in ``prepr_pool_prove.sh`` did not.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from omnibase_infra.validation.checker_first_party_sources import load_contract

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
PROVE = ROOT / "scripts/runtime_build/prepr_pool_prove.sh"
BUILD_ARG = "--build-arg FIRST_PARTY_BUILD_CONTEXT="


def _build_command() -> str:
    """The isolated member's `docker compose ... build` line."""
    lines = [
        line
        for line in PROVE.read_text(encoding="utf-8").splitlines()
        if "docker compose -f docker/docker-compose.generated.yml" in line
        and " build " in line
    ]
    assert len(lines) == 1, lines
    return lines[0]


def test_isolated_build_passes_the_first_party_context() -> None:
    assert BUILD_ARG in _build_command()


def test_the_context_is_read_from_the_tree_under_test() -> None:
    """The substitution the build runs, run as written, yields the slot context."""
    match = re.search(
        re.escape(BUILD_ARG) + r'"(\$\(python3 .*?\.py\))"', _build_command()
    )
    assert match is not None, "the build reads no context from the policy"
    done = subprocess.run(
        ["bash", "-c", f'printf %s "{match.group(1)}"'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    contract = load_contract()
    assert done.stdout == contract.pre_pr_slot_context
    # The control: the context the build falls back to without the arg refuses
    # the branch build this one needs.
    assert "branch" in contract.context(done.stdout).target_accepts
    assert "branch" not in contract.context(contract.default_context).target_accepts
