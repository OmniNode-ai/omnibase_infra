# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The release writer's contract item survives the yamlfmt fold (OMN-25638).

The handler is real and writes into a real clone directory on disk; the
written contract is read back as YAML, the way the release pull request's
Pre-commit Suite reads it. The fold google/yamlfmt applies to a ``>-`` block
scalar (``max_line_length: 100``, content at indent 10) is restated here as an
independent oracle, so the writer's fold is judged against the hook's rule and
not against the writer's own helper.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.handlers.handler_release_notes_apply import apply_release_notes
from tests.unit.nodes.node_release_notes_apply_effect.test_handler_release_notes_apply import (
    _ENTRY_0_47_75,
    _readme_only_clone,
)

pytestmark = pytest.mark.integration

_BLOCK_INDENT = 10
_MAX_LINE_LENGTH = 100


def _yamlfmt_fold(text: str) -> list[str]:
    """Rows yamlfmt produces for a folded scalar: a row keeps taking words
    while the row already built is at most ``max_line_length - indent`` wide."""
    rows: list[str] = []
    for word in text.split(" "):
        if rows and len(rows[-1]) <= _MAX_LINE_LENGTH - _BLOCK_INDENT:
            rows[-1] += " " + word
        else:
            rows.append(word)
    return rows


def test_written_contract_item_is_a_fixed_point_of_the_yamlfmt_fold(
    tmp_path: Path,
) -> None:
    clone = _readme_only_clone(tmp_path)
    apply_release_notes(clone, _ENTRY_0_47_75, repo_name="omnibase_core")

    path = clone / "contracts" / "OMN-18595.yaml"
    raw_rows = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, row in enumerate(raw_rows) if "uv run --locked" in row) - 1
    assert raw_rows[start].strip() == "check_value: >-"
    block: list[str] = []
    for row in raw_rows[start + 1 :]:
        if not row.startswith(" " * _BLOCK_INDENT):
            break
        block.append(row[_BLOCK_INDENT:])
    assert len(block) > 1, "positive control: the long command is folded"

    command = " ".join(block)
    assert block == _yamlfmt_fold(command)

    contract = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = [
        check["check_value"]
        for item in contract["dod_evidence"]
        for check in item["checks"]
    ]
    assert command in values
