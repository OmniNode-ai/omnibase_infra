# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Replay of verdicts recorded from validator scripts that no longer exist (OMN-20568).

While a script and its replacement node coexisted, a parity test ran both over a
matrix of real git repositories and recorded the script's exit code, stdout and
stderr per test (``tests/fixtures/validator_parity/<rule>/golden.json``). After the
script is deleted the same test replays those recordings as the expected verdict,
so the matrix keeps guarding the node.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

__all__ = ["RecordedVerdicts", "normalise_git_error"]

_GIT_DETAIL = re.compile(r"((?:failed|exit) \(?\d+\)?): .*", re.DOTALL)


_COMMIT_SHA = re.compile(r"\b[0-9a-f]{40}\b")


def normalise_git_error(text: str) -> str:
    """Mask what varies run to run: commit shas, and git's own wording after the exit
    status, which varies by git version."""
    return _COMMIT_SHA.sub("<sha>", _GIT_DETAIL.sub(r"\1", text))


class RecordedVerdicts:
    """Per-test queue of recorded script verdicts, consumed in call order."""

    def __init__(self, golden_path: Path) -> None:
        self._data: dict[str, list[dict[str, Any]]] = json.loads(
            golden_path.read_text(encoding="utf-8")
        )
        self._queue: list[dict[str, Any]] = []

    def bind(self, nodeid: str) -> None:
        self._queue = list(self._data.get(nodeid, []))

    def take(self, kind: str, *, skip: tuple[str, ...] = ()) -> dict[str, Any]:
        """Pop the next record of ``kind``, dropping leading records of ``skip`` kinds.

        ``skip`` covers calls the removed script made internally before it returned,
        which the recording captured as their own entries.
        """
        while self._queue and self._queue[0]["kind"] in skip:
            self._queue.pop(0)
        record = self._queue.pop(0)
        assert record["kind"] == kind, f"recorded {record['kind']}, test asked {kind}"
        return record
