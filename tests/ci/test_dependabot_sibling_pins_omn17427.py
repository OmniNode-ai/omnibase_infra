# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21351 removed the version-update schedules, so no Dependabot run can
propose an OmniNode-ai sibling pin bump.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_github_actions_version_update_entry() -> None:
    path = REPO_ROOT / ".github" / "dependabot.yml"
    if not path.exists():
        return
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert not any(
        entry["package-ecosystem"] == "github-actions"
        for entry in document.get("updates", []) or []
    )


def test_sibling_workflow_and_action_uses_remain_present() -> None:
    files = [
        *(REPO_ROOT / ".github" / "workflows").glob("*.yml"),
        *(REPO_ROOT / ".github" / "actions").glob("**/action.yml"),
    ]
    names = {
        match.group(1)
        for path in files
        for match in re.finditer(
            r"uses:\s*(OmniNode-ai/[^@\s]+)@", path.read_text(encoding="utf-8")
        )
    }
    assert names, "expected sibling workflow or action uses"
