# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17427 - a workflow whose job reads the pull request body re-runs when the body is edited.

`deploy-gate.yml` listed `[opened, synchronize, reopened, labeled, unlabeled, ready_for_review]`,
and its job reads the body live to find the cited ticket. A body fixed after the first run never
re-ran the job, so the red stayed on the head (omnibase_infra#5217 stayed red on a body that the
real validator accepted). These tests hold every workflow that reads the body to `edited`.

A workflow reads the body when its parsed content carries `github.event.pull_request.body`, a
`--json body` read, a `PR_BODY` variable, or when it calls one of the reusable workflows whose
job reads the body inside the remote repository (so the read is not visible in this one). A
workflow that fires only on `closed` is exempt: it judges the body the pull request merged with,
and an edit cannot change that. `ci.yml` does not read the body and keeps `edited` out on purpose
(OMN-16171: a body edit would restart its whole matrix); a control below pins both halves.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# The two omniclaude reusable workflows this repository calls whose job reads the body inside
# omniclaude: the deploy-gate composite action runs `gh pr view --json body`, and the skip-token
# scan reads the body for the bypass tokens.
BODY_READING_REUSABLES = ("deploy-gate-reusable.yml", "reject-deploy-gate-skip.yml")

_DIRECT_BODY_READ = re.compile(
    r"github\.event\.pull_request\.body|--json\s+body|\bPR_BODY\b"
)
_PULL_REQUEST_EVENTS = ("pull_request", "pull_request_target")
_DEFAULT_PULL_REQUEST_TYPES = frozenset({"opened", "synchronize", "reopened"})


def _triggers(document: dict[Any, Any]) -> dict[str, Any]:
    # PyYAML reads the bare key `on` as the boolean True.
    raw = document.get("on", document.get(True))
    if isinstance(raw, str):
        return {raw: None}
    if isinstance(raw, list):
        return {str(event): None for event in raw}
    return dict(raw or {})


def _reads_body(document: dict[Any, Any]) -> bool:
    # Serialising the parsed document drops comments, so prose that names the body does not count.
    text = json.dumps(document, default=str)
    return bool(_DIRECT_BODY_READ.search(text)) or any(
        reusable in text for reusable in BODY_READING_REUSABLES
    )


def _pull_request_types(document: dict[Any, Any]) -> dict[str, frozenset[str]]:
    types_by_event: dict[str, frozenset[str]] = {}
    triggers = _triggers(document)
    for event in _PULL_REQUEST_EVENTS:
        if event not in triggers:
            continue
        config = triggers[event] or {}
        declared = config.get("types") if isinstance(config, dict) else None
        types_by_event[event] = (
            frozenset(declared) if declared else _DEFAULT_PULL_REQUEST_TYPES
        )
    return types_by_event


def _violations(document: dict[Any, Any]) -> list[str]:
    """Return the pull request events of a body-reading workflow that do not fire on edit."""
    if not _reads_body(document):
        return []
    return [
        event
        for event, types in _pull_request_types(document).items()
        if types != frozenset({"closed"}) and "edited" not in types
    ]


def _load_all() -> dict[str, dict[Any, Any]]:
    return {
        path.name: dict(yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in sorted(WORKFLOWS.glob("*.yml"))
    }


def test_every_pr_body_reader_reruns_on_edit() -> None:
    offenders = {
        name: violations
        for name, document in _load_all().items()
        if (violations := _violations(document))
    }
    assert not offenders, (
        "a workflow reads the pull request body but its pull_request types omit `edited`, "
        f"so a body fixed after the first run never clears its red: {offenders}"
    )


def test_scan_finds_the_known_body_readers() -> None:
    # Without this the scan above could pass by finding nothing.
    readers = {name for name, doc in _load_all().items() if _reads_body(doc)}
    assert {
        "deploy-gate.yml",
        "ci-live-contact.yml",
        "non-dev-base-guard.yml",
        "call-reject-skip.yml",
    } <= readers


def test_deploy_gate_fires_on_edit() -> None:
    document = _load_all()["deploy-gate.yml"]
    assert "edited" in _pull_request_types(document)["pull_request"]


def test_ci_matrix_workflow_reads_no_body_and_keeps_edited_out() -> None:
    document = _load_all()["ci.yml"]
    assert not _reads_body(document)
    assert "edited" not in _pull_request_types(document)["pull_request"]


def _synthetic(body: str, on_block: str) -> dict[Any, Any]:
    return dict(yaml.safe_load(f"{on_block}\njobs:\n  j:\n    steps:\n{body}"))


_READS_BODY_STEP = (
    "      - run: echo\n        env:\n"
    "          PR_BODY: ${{ github.event.pull_request.body }}\n"
)


@pytest.mark.parametrize(
    ("body", "on_block", "expected"),
    [
        # Positive control: a body reader without `edited` is reported.
        (
            _READS_BODY_STEP,
            "on:\n  pull_request:\n    types: [opened, synchronize]",
            ["pull_request"],
        ),
        # The default types carry no `edited` either.
        (_READS_BODY_STEP, "on:\n  pull_request:", ["pull_request"]),
        (_READS_BODY_STEP, "on: pull_request", ["pull_request"]),
        (
            _READS_BODY_STEP,
            "on:\n  pull_request_target:\n    types: [opened]",
            ["pull_request_target"],
        ),
        # A reusable whose job reads the body in another repository counts.
        (
            "      - uses: O/omniclaude/.github/workflows/deploy-gate-reusable.yml@x\n",
            "on:\n  pull_request:\n    types: [opened]",
            ["pull_request"],
        ),
        # Negative controls: `edited` present, closed-only, no body read, no pull_request event.
        (
            _READS_BODY_STEP,
            "on:\n  pull_request:\n    types: [opened, edited]",
            [],
        ),
        (_READS_BODY_STEP, "on:\n  pull_request:\n    types: [closed]", []),
        ("      - run: echo hi\n", "on:\n  pull_request:\n    types: [opened]", []),
        (_READS_BODY_STEP, "on:\n  push:\n    branches: [dev]", []),
        # Prose in a comment is not a read.
        (
            "      - run: echo hi # PR_BODY\n",
            "on:\n  pull_request:\n    types: [opened]",
            [],
        ),
    ],
)
def test_checker_controls(body: str, on_block: str, expected: list[str]) -> None:
    assert _violations(_synthetic(body, on_block)) == expected
