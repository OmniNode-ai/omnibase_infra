# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replay (OMN-17427): a rejection alone cleared a hostile-reviewer finding.

Registered in tests/incident_replays/registry.yaml as
``omn17427-a-rejection-alone-cleared-a-finding``. The artifact is the two
``BLOCKING:`` lines of job 119316797666 (the Hostile Review Thread Gate on
omnibase_infra#5234), where the gate accepted a ``rejected:`` reply on each of
the PR's two finding threads as a disposition. The replay rebuilds those two
threads from the facts the lines carry and drives the real gate: its
``--pending`` scan, which hands rejections to the review bot, and its verdict.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_CI = _ROOT / "scripts" / "ci"
_ARTIFACT = (
    _ROOT
    / "tests"
    / "fixtures"
    / "omn17427"
    / "infra-pr5234-job-119316797666-thread-gate-log-lines.captured"
)
_SHA256 = "f51427b0f6236f1315f77934694bdd27c4f9d03ee72fce43e9ff3fba16133f4e"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _CI / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate = _load("check_hostile_review_threads")

_LINE_RE = re.compile(
    r"BLOCKING: (?P<path>\S+) \((?P<model>[a-z-]+)\) — (?P<reason>[^;]+);"
    r".*discussion_r(?P<id>\d+)$"
)
# The rejection's words are not in the log; the verdict under test does not
# depend on them.
_REJECTION = "rejected: placeholder, the job log does not carry the reply text"


def _incident_threads() -> list[dict[str, Any]]:
    raw = _ARTIFACT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _SHA256
    threads = []
    for index, line in enumerate(raw.decode("utf-8").splitlines()):
        match = _LINE_RE.search(line)
        assert match, line
        # What the shipped gate concluded about this thread.
        assert match["reason"] == "rejected with a reason"
        threads.append(
            {
                "id": f"T{index}",
                "isResolved": False,
                "isOutdated": False,
                "path": match["path"],
                "originalLine": None,
                "subjectType": "FILE",
                "comments": {
                    "nodes": [
                        {
                            "databaseId": int(match["id"]),
                            "body": (
                                f"<!-- {gate.MARKER} fp={index:012x} -->\n"
                                f"**[MINOR] hostile-reviewer ({match['model']})**\n\n"
                                "finding"
                            ),
                            "author": {"login": "github-actions"},
                        },
                        {
                            "databaseId": int(match["id"]) + 1,
                            "body": _REJECTION,
                            "author": {"login": "operator-account"},
                            "authorAssociation": "MEMBER",
                        },
                    ]
                },
            }
        )
    assert [t["comments"]["nodes"][0]["databaseId"] for t in threads] == [
        4338489373,
        4338489546,
    ]
    return threads


def _no_fix_check(sha: str, cited: object) -> tuple[bool, str]:
    raise AssertionError("a rejection runs no fix check")


def _verdict(thread: dict[str, Any], author: str) -> dict[str, Any]:
    rejection = thread["comments"]["nodes"][1]
    return {
        "body": (
            f"<!-- {gate.JUDGE_MARKER} rejection={rejection['databaseId']} "
            f"digest={gate.body_digest(rejection['body'])} verdict=accepted "
            "judge=rejection-judge identity=anthropic#id=claude-opus-5-5 -->\n"
            "**Review bot: rejection accepted**\n\n> the finding misreads the file\n"
        ),
        "author": {"login": author},
    }


def test_the_real_gate_keeps_both_rejected_findings_blocked() -> None:
    threads = _incident_threads()
    pending, notes = gate.pending_rejections(threads)
    # Both rejections await the review bot; neither clears by itself.
    assert [p.raised_by for p in pending] == ["correctness", "sonnet"]
    assert notes == {}
    blocking, cleared = gate.classify_threads(threads, _no_fix_check)
    assert cleared == []
    assert [v.judge_verdict for v in blocking] == ["unjudged", "unjudged"]


def test_the_same_findings_clear_once_the_judge_accepts() -> None:
    threads = _incident_threads()
    for thread in threads:
        thread["comments"]["nodes"].append(_verdict(thread, "github-actions"))
    blocking, cleared = gate.classify_threads(threads, _no_fix_check)
    assert blocking == []
    assert [v.judge_verdict for v in cleared] == ["accepted", "accepted"]
    assert gate.pending_rejections(threads)[0] == []

    forged = _incident_threads()
    for thread in forged:
        thread["comments"]["nodes"].append(_verdict(thread, "operator-account"))
    blocking, _ = gate.classify_threads(forged, _no_fix_check)
    assert len(blocking) == 2
