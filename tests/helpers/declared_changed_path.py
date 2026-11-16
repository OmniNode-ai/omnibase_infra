# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""node_lab_proof_judge_compute's answers for changed_path_live, as the node printed them (OMN-23715).

The pool driver asks the node's ``pool-run`` command and applies the answer (``prepr_runtime_pool.declared_changed_path``).
Each answer below is the JSON the node printed for the scenario named, recorded from its handlers over the example
overlay; the node's own tests (omnimarket ``tests/unit/nodes/node_lab_proof_judge_compute``) hold the rule, and these
tests hold what the driver does with the answer.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest import mock

LIVE_TERM_DOT = "[.]"

NODE_LIVE: dict[str, Any] = {
    "verdict": "PASS",
    "ok": True,
    "evidence": "judge read the changed path live",
    "notes": [],
}


def dead(node: str) -> dict[str, Any]:
    """A runtime node no log mentions, with no declaration to excuse it."""
    note = f"changed path not live, no runtime log mentions: {node}"
    return {
        "verdict": "FAIL",
        "ok": False,
        "evidence": f"no not-applicable declaration for OmniNode-ai/omnibase_infra; {note}",
        "notes": [note],
    }


def local(node: str) -> dict[str, Any]:
    """A node that runs in no runtime (no topics, no runtime profile)."""
    note = (
        "changed path not applicable, runs in no runtime "
        f"(no topics, no runtime profile): {node}"
    )
    return {
        "verdict": "PASS",
        "ok": True,
        "evidence": f"judge read the changed path live; {note}",
        "notes": [note],
    }


def declared(node: str, receipt_id: str) -> dict[str, Any]:
    """A node the overlay declares as running outside the runtime, proven by a substitute receipt."""
    return {
        "verdict": "NOT_APPLICABLE",
        "ok": True,
        "class_id": "runs_outside_runtime",
        "evidence": f"NOT-APPLICABLE class=runs_outside_runtime entries={node} substitute={receipt_id}",
        "notes": [],
    }


NO_CHECK: dict[str, Any] = {
    "verdict": "NO_CHECK",
    "ok": False,
    "evidence": "the probe printed no live-grep line, so there is no changed-path check",
    "notes": [],
}


@contextmanager
def node_says(pool: Any, answer: dict[str, Any] | None) -> Iterator[mock.Mock]:
    """Make the driver's command answer ``answer`` (None: the command is unset or failed)."""
    with mock.patch.object(pool, "declared_changed_path", return_value=answer) as asked:
        yield asked
