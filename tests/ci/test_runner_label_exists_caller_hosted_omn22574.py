# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runner-label-exists caller runs the gate on a GitHub-hosted runner (OMN-22574).

The caller pins the omniclaude reusable by commit. The commit ed4300f8 names a self-hosted label, so a
public pull request here ran the gate on a lab runner. The commit 9a56b9fc (omniclaude#3171) selects a
hosted runner. This test pins the caller to a commit that does, and names the one that does not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

CALLER = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "runner-label-exists-caller.yml"
)
HOSTED_REUSABLE_SHA = "9a56b9fcee48e5221bfc2ae71b5a82bb9e11a751"
LAB_SELECTING_REUSABLE_SHA = "ed4300f8a427ead7f099daf69cedc012907cdbd3"
USES = re.compile(r"runner-label-exists-reusable\.yml@([0-9a-f]{40})$")


def _pin() -> str:
    document = yaml.safe_load(CALLER.read_text(encoding="utf-8"))
    uses = document["jobs"]["runner-label-exists"]["uses"]
    match = USES.search(uses)
    assert match is not None, f"the caller does not pin the reusable by commit: {uses}"
    return match.group(1)


def test_the_caller_pins_the_reusable_commit_that_selects_a_hosted_runner() -> None:
    assert _pin() == HOSTED_REUSABLE_SHA


def test_the_caller_does_not_pin_the_commit_that_selects_a_lab_runner() -> None:
    assert _pin() != LAB_SELECTING_REUSABLE_SHA
