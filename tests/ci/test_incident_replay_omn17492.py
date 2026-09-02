# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Incident replays for the OMN-17492 hostile-review thread surface.

Two captured-bytes cases (registered in tests/incident_replays/registry.yaml):

1. ``cli-review-glm-infra-pr3121.result.json.captured`` — a VERBATIM
   ``ModelMultiReviewResult`` produced by the real
   ``omniintelligence.review_pairing.cli_review`` running the real
   ``glm-review`` model (z.ai GLM Coding Plan) against the real, merged
   omnibase_infra PR #3121 on 2026-09-01. It contains 10 findings. The
   shipped hostile-reviewer.yml verdict parser read ``merged_findings`` /
   ``total_input_findings`` — fields this model NEVER had — so on exactly
   this shape it reported "Total findings: 0" and discarded every finding.
   The replay drives the REAL poster over these bytes and requires a
   non-empty review payload (reject the zero-findings verdict).

2. ``omnibase-core-pr1604-review-threads.graphql.json.captured`` — the
   VERBATIM GraphQL ``reviewThreads`` connection for omnibase_core PR
   #1604, the OMN-16823 incident PR: ``check-unresolved-threads.sh``
   (CodeRabbit era) kept that PR merge-BLOCKED after CodeRabbit had
   conceded in-thread, because it parsed concession prose instead of
   resolution state. The replay drives the REAL thread gate over these
   bytes and requires accept: none of those threads are hostile-reviewer
   threads (marker scoping — other actors' threads are structurally out of
   scope), and resolution state is read from ``isResolved``, not prose.
   The paired discriminator (registry ``discriminator:``) proves the same
   gate still blocks on a real unresolved marker thread.

Re-fetch recipes are in each registry entry's ``why``; sha256 pins below
guarantee the tests run over the exact captured bytes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "omn17492"

_REVIEW_FIXTURE = _FIXTURES / "cli-review-glm-infra-pr3121.result.json.captured"
_REVIEW_SHA256 = "0339f941c500374269e88a5458a119d1692c2ad17710346c8bc9b4b00f987d82"

_THREADS_FIXTURE = (
    _FIXTURES / "omnibase-core-pr1604-review-threads.graphql.json.captured"
)
_THREADS_SHA256 = "d1f89dd5eed372dde8ccb67e5f9816283ddf8412f2cb27c8d83a956c8ed2e3d1"


def _load_module(name: str) -> object:
    spec = importlib.util.spec_from_file_location(
        name, _REPO_ROOT / "scripts" / "ci" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_real_poster_surfaces_the_findings_the_shipped_parser_discarded() -> None:
    """False-green replay: the shipped verdict parser said 0 on these bytes."""
    raw = _REVIEW_FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _REVIEW_SHA256, (
        "fixture bytes drifted — a replay over different bytes proves nothing"
    )
    review_result = json.loads(raw)

    # First, pin the incident: the SHIPPED parser's exact reads come back
    # empty on this real result — that emptiness IS the defect.
    assert review_result.get("merged_findings", []) == []
    assert review_result.get("total_input_findings", 0) == 0
    assert review_result["total_findings"] == 10

    poster = _load_module("post_hostile_review_threads")
    findings = poster.collect_findings(review_result)
    assert len(findings) == 10

    postable, suppressed_hints = poster.split_by_noise_policy(findings)
    # 2 hint-class findings suppressed by the noise policy; 8 postable.
    assert suppressed_hints == 2
    assert len(postable) == 8

    # Drive the real assembly with no diff anchors and no prior threads —
    # the weakest possible posting context. The verdict the buggy guard got
    # wrong: it produced NOTHING. The real guard must produce a review
    # payload carrying every postable finding.
    payload, stats = poster.build_review(
        postable,
        changed_files={},
        existing_fps=set(),
        suppressed_hints=suppressed_hints,
        models_succeeded=list(review_result["models_succeeded"]),
        models_failed=list(review_result["models_failed"]),
    )
    assert payload is not None, (
        "the poster produced no review for a 10-finding result — that is "
        "the OMN-17492 discarded-findings defect, replayed"
    )
    assert stats["body_findings"] == 8
    # The one MAJOR (warning) finding forces the request-changes posture.
    assert payload["event"] == "REQUEST_CHANGES"


def test_the_thread_gate_accepts_the_real_pr1604_threads_that_wedged_omn16823() -> None:
    """False-red replay: the CodeRabbit-era gate kept these threads blocking."""
    raw = _THREADS_FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _THREADS_SHA256, (
        "fixture bytes drifted — a replay over different bytes proves nothing"
    )
    data = json.loads(raw)
    threads = data["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]
    assert len(threads) == 3, "PR #1604 carried exactly 3 review threads"

    gate = _load_module("check_hostile_review_threads")
    blocking, outdated = gate.classify_threads(threads)

    # None of PR #1604's threads are hostile-reviewer threads: they are
    # structurally out of this gate's scope regardless of resolution state
    # or concession prose. blocking == [] is the verdict the OMN-16823-era
    # gate could not produce.
    assert blocking == []
    assert outdated == []
