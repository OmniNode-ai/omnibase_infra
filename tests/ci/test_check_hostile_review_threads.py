# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/ci/check_hostile_review_threads.py (OMN-17492).

The thread gate is the DETERMINISTIC merge surface for hostile-review
findings — these tests pin its classification rule: only unresolved,
non-outdated threads whose first comment carries the marker block; resolved
threads, outdated threads, and other actors' threads never do.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "check_hostile_review_threads.py"
)
_spec = importlib.util.spec_from_file_location("check_hostile_review_threads", _SCRIPT)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules["check_hostile_review_threads"] = gate
_spec.loader.exec_module(gate)


def _thread(
    *,
    resolved: bool = False,
    outdated: bool = False,
    marker: bool = True,
    path: str = "src/foo.py",
) -> dict[str, Any]:
    body = (
        f"<!-- {gate.MARKER} fp=abc123def456 -->\nfinding" if marker else "human note"
    )
    return {
        "isResolved": resolved,
        "isOutdated": outdated,
        "path": path,
        "comments": {"nodes": [{"body": body, "url": "https://x/thread"}]},
    }


class TestClassifyThreads:
    def test_unresolved_marker_thread_blocks(self) -> None:
        blocking, outdated = gate.classify_threads([_thread()])
        assert len(blocking) == 1
        assert outdated == []

    def test_resolved_marker_thread_is_ignored(self) -> None:
        blocking, outdated = gate.classify_threads([_thread(resolved=True)])
        assert blocking == []
        assert outdated == []

    def test_outdated_unresolved_marker_thread_reports_but_does_not_block(
        self,
    ) -> None:
        blocking, outdated = gate.classify_threads([_thread(outdated=True)])
        assert blocking == []
        assert len(outdated) == 1

    def test_non_marker_threads_are_none_of_this_gates_business(self) -> None:
        blocking, outdated = gate.classify_threads(
            [_thread(marker=False), _thread(marker=False, outdated=True)]
        )
        assert blocking == []
        assert outdated == []

    def test_thread_with_no_comments_is_ignored(self) -> None:
        thread: dict[str, Any] = {
            "isResolved": False,
            "isOutdated": False,
            "path": "x",
            "comments": {"nodes": []},
        }
        blocking, outdated = gate.classify_threads([thread])
        assert blocking == []
        assert outdated == []

    def test_mixed_population(self) -> None:
        threads = [
            _thread(),
            _thread(resolved=True),
            _thread(outdated=True),
            _thread(marker=False),
        ]
        blocking, outdated = gate.classify_threads(threads)
        assert len(blocking) == 1
        assert len(outdated) == 1
