# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ARM64 Verify Runner Proof starts from no public pull request and no schedule.

RULING 2026-11-05T17:39:59Z: a public repository job on an internal runner is a
mistake. This workflow's jobs run on the lab verify pool, so a ``pull_request``
or ``schedule`` trigger places a public job there on every matching event. Only
the operator's own ``workflow_dispatch`` remains.

Ticket: OMN-22600
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "arm64-verify-runner-proof.yml"
)


def _triggers() -> dict[str, object]:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML reads the bare key ``on`` as the boolean True.
    triggers = document.get("on", document.get(True))
    assert isinstance(triggers, dict), "the workflow declares no trigger mapping"
    return triggers


def test_the_proof_declares_only_the_operators_dispatch() -> None:
    assert set(_triggers()) == {"workflow_dispatch"}


@pytest.mark.parametrize("trigger", ["pull_request", "schedule"])
def test_a_public_event_never_places_the_proof_on_the_lab_pool(trigger: str) -> None:
    assert trigger not in _triggers()


def test_the_positive_control_reads_a_trigger_that_exists() -> None:
    """The reader sees triggers: a document with a schedule is read as having one."""
    document = yaml.safe_load("on:\n  schedule:\n    - cron: '0 0 * * *'\n")
    assert "schedule" in document.get("on", document.get(True))
