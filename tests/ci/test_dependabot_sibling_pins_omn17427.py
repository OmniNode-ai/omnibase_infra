# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427 / OMN-21351: no Dependabot run can move a workflow or action pin.

OMN-21351 deleted .github/dependabot.yml, so no ecosystem has a version-update
schedule: neither OmniNode-ai sibling pins nor third-party action pins are
proposed by Dependabot any more. Both move only in a deliberate PR that passes
the gates reading the new sha. Every sibling ``uses:`` is pinned to a full
commit sha, so no tag or branch can float underneath it either.

Falsifier: restore a dependabot.yml, or pin any OmniNode-ai ``uses:`` to a tag
or branch, and a test here fails.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SIBLING_USES = re.compile(r"uses:\s*(OmniNode-ai/[^@\s]+)@(\S+)")


def _sibling_uses() -> dict[str, set[str]]:
    files = [
        *(REPO_ROOT / ".github" / "workflows").glob("*.yml"),
        *(REPO_ROOT / ".github" / "actions").glob("**/action.yml"),
    ]
    uses: dict[str, set[str]] = {}
    for path in files:
        for match in SIBLING_USES.finditer(path.read_text(encoding="utf-8")):
            uses.setdefault(match.group(1), set()).add(match.group(2))
    return uses


def test_no_dependabot_configuration() -> None:
    assert not (REPO_ROOT / ".github" / "dependabot.yml").exists()
    assert not (REPO_ROOT / ".github" / "dependabot.yaml").exists()


def test_every_sibling_uses_is_pinned_to_a_full_sha() -> None:
    uses = _sibling_uses()
    assert uses, "expected sibling workflow or action uses"
    floating = {
        name: sorted(ref for ref in refs if not re.fullmatch(r"[0-9a-f]{40}", ref))
        for name, refs in uses.items()
    }
    assert not {name: refs for name, refs in floating.items() if refs}
