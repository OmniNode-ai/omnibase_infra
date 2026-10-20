# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427 / OMN-20331: Dependabot ignores sibling workflow and action pins.

OmniNode-ai sibling pins move only in PRs passing the gates that read the new
sha; third-party actions retain their Dependabot updates.

Falsifier: delete the OmniNode-ai/* ignore from .github/dependabot.yml and
test_github_actions_updates_ignore_sibling_pins fails.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import cast

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _github_actions_ignores() -> list[dict[str, object]]:
    updates: list[dict[str, object]] = yaml.safe_load(
        (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    )["updates"]
    update = next(
        entry for entry in updates if entry["package-ecosystem"] == "github-actions"
    )
    return cast("list[dict[str, object]]", update.get("ignore", []))


def test_github_actions_updates_ignore_sibling_pins() -> None:
    ignores = [
        entry
        for entry in _github_actions_ignores()
        if entry["dependency-name"] == "OmniNode-ai/*"
    ]
    assert ignores, "github-actions updates must ignore OmniNode-ai/*"
    assert all(
        "versions" not in entry and "update-types" not in entry for entry in ignores
    )


def test_every_sibling_uses_is_covered_by_the_ignore() -> None:
    pattern = next(
        entry["dependency-name"]
        for entry in _github_actions_ignores()
        if entry["dependency-name"] == "OmniNode-ai/*"
    )
    assert isinstance(pattern, str)
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
    assert all(fnmatch.fnmatchcase(name, pattern) for name in names)


def test_third_party_actions_still_update() -> None:
    for entry in _github_actions_ignores():
        pattern = entry["dependency-name"]
        assert isinstance(pattern, str)
        assert not fnmatch.fnmatchcase("actions/setup-node", pattern), pattern
