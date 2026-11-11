# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""No live list in omnibase_infra names a retired repository (OMN-23579).

The host reconcile unit refuses to run while any registry list, sync script or
source constant in a checkout names a retired repository. These files held such a
name as a live entry. Comments narrate the retirement and are not targets, so only
non-comment lines are read.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
RETIRED_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(onex[_-]change[_-]control|omnibase[_-]compat|ycbench[_-]onex)"
    r"(?![A-Za-z0-9_.-])"
)

LIVE_LIST_FILES = (
    "config/runner_fleet.yaml",
    "config/runner_routing_policy.yaml",
    "docker/runners/git-mirror-refresh.sh",
    "scripts/audit_orphan_required_contexts.py",
    "scripts/enforcement_parity_manifest.yaml",
    "scripts/omninode-ci-required-context-probe.py",
    "scripts/release_drift_monitor.py",
    "src/omnibase_infra/nodes/node_scope_extract_compute/handlers/handler_scope_extract.py",
    "src/omnibase_infra/services/agent_learning_extraction/consumer.py",
)


def _live_python_lines(source: str) -> list[int]:
    """Lines of a non-docstring string literal that names a retired repository."""
    tree = ast.parse(source)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            docstrings.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return sorted(
        {
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.lineno not in docstrings
            and RETIRED_RE.search(node.value)
        }
    )


def _live_lines(text: str, relative: str = "") -> list[int]:
    if relative.endswith(".py"):
        return _live_python_lines(text)
    return [
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if not line.lstrip().startswith("#") and RETIRED_RE.search(line)
    ]


def test_positive_control_the_matcher_sees_a_live_entry_and_skips_a_comment() -> None:
    assert _live_lines("    - omnibase_compat\n") == [1]
    assert _live_lines('REPOS = ("onex_change_control",)\n') == [1]
    assert _live_lines("# onex_change_control narrates the retirement\n") == []
    assert _live_lines("    - omnibase_core\n") == []
    assert _live_lines('REPOS = ("onex_change_control",)\n', "x.py") == [1]
    assert _live_lines('"""onex_change_control"""\nimport os\n', "x.py") == []


@pytest.mark.parametrize("relative", LIVE_LIST_FILES)
def test_live_list_names_no_retired_repository(relative: str) -> None:
    text = (ROOT / relative).read_text(encoding="utf-8")
    assert _live_lines(text, relative) == [], f"{relative} names a retired repository"


@pytest.mark.parametrize("relative", LIVE_LIST_FILES)
def test_live_list_keeps_its_active_repositories(relative: str) -> None:
    text = (ROOT / relative).read_text(encoding="utf-8")
    assert "omnibase_core" in text or "omnibase_infra" in text
