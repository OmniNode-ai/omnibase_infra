# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21433: the documentation-surface item of contracts/OMN-17427.yaml holds.

``dod-omn17427-documentation-surface`` is a ``command`` check that asserts the
root README links to the knowledge base and is short. A README that grew past
the item's line bound (the dev work-ledger section, OMN-20740) left the item
false on dev, and an edit of the item inside a pull request that cites
OMN-17427 could never fail at its merge base: the receipt gate lays the cited
contract over the base, so the corrected item and the unchanged README pass at
both ends. This test runs the item's own commands from the repository root and
is bound by a pull request that does not cite OMN-17427, so the base keeps the
stale contract and fails here while the head passes.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "contracts" / "OMN-17427.yaml"
ITEM_ID = "dod-omn17427-documentation-surface"
KNOWLEDGE_BASE_URL = "https://github.com/OmniNode-ai/knowledge_base"


def _item_commands() -> list[str]:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = [
        item for item in contract["dod_evidence"] if item["id"] == ITEM_ID
    ]
    assert len(items) == 1, f"{ITEM_ID} must be declared exactly once"
    return [
        check["check_value"]
        for check in items[0]["checks"]
        if check["check_type"] == "command"
    ]


def _run(command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    argv = shlex.split(command)
    assert argv[0] == "python3", f"unexpected interpreter in the item: {argv[0]}"
    argv[0] = sys.executable
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)


def test_item_declares_a_command() -> None:
    assert _item_commands(), f"{ITEM_ID} declares no command check"


def test_item_commands_exit_zero_at_repo_root() -> None:
    for command in _item_commands():
        result = _run(command, REPO_ROOT)
        assert result.returncode == 0, (
            f"{ITEM_ID} fails at the repository root:\n{result.stdout}{result.stderr}"
        )


def _readme(link: bool = True, lines: int = 5) -> str:
    return (KNOWLEDGE_BASE_URL if link else "no link") + "\n" + "line\n" * lines


@pytest.mark.parametrize(
    ("case", "readme", "extra_file"),
    [
        ("oversized README", _readme(lines=200), None),
        ("README without the knowledge-base link", _readme(link=False), None),
        ("unused security policy present", _readme(), "SECURITY.md"),
        ("unused evidence text present", _readme(), "docs/evidence/note.txt"),
        ("unused fixture README present", _readme(), "tests/fixtures/README.md"),
    ],
)
def test_item_commands_refuse_each_violation(
    tmp_path: Path, case: str, readme: str, extra_file: str | None
) -> None:
    """Positive controls: the item still asserts each thing it exists to assert."""
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    if extra_file is not None:
        target = tmp_path / extra_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x\n", encoding="utf-8")
    for command in _item_commands():
        result = _run(command, tmp_path)
        assert result.returncode != 0, f"{ITEM_ID} accepted: {case}"
        assert "AssertionError" in result.stderr
