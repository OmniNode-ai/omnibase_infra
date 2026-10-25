# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20910: recorded merge-only rereviews of omnibase_infra#5261.

The recording combines hostile-reviewer threads read through ``onex scm``
with the PR's commit history and GET /pulls/{n}/files-shaped diffs. All eight
heads after the last content commit carry identical PR files, yet repeated
reviews opened unanswered threads, including the already rejected empty
routing ladder finding under fresh fingerprints. A bot review's diff digest
must suppress those later rounds even when finding fingerprints change.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.unit

_POSTER = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "post_hostile_review_threads.py"
)
_spec = importlib.util.spec_from_file_location("post_hostile_review_threads", _POSTER)
assert _spec is not None and _spec.loader is not None
poster = importlib.util.module_from_spec(_spec)
sys.modules["post_hostile_review_threads"] = poster
_spec.loader.exec_module(poster)

_GATE = _POSTER.parent / "check_hostile_review_threads.py"
_gate_spec = importlib.util.spec_from_file_location(
    "check_hostile_review_threads", _GATE
)
assert _gate_spec is not None and _gate_spec.loader is not None
gate = importlib.util.module_from_spec(_gate_spec)
sys.modules["check_hostile_review_threads"] = gate
_gate_spec.loader.exec_module(gate)


@pytest.mark.live_contact(
    "tests/ci/fixtures/omn20910_infra_pr5261_merge_only_rereview.json"
)
def test_merge_only_heads_share_one_pr_diff_digest(
    recorded_response: dict[str, object],
) -> None:
    commits = cast("list[dict[str, Any]]", recorded_response["commits"])
    merge_only_heads = cast(
        "dict[str, dict[str, str]]", recorded_response["merge_only_heads"]
    )
    pr_files = cast("list[dict[str, Any]]", recorded_response["pr_files"])

    last_content_index = max(
        index for index, commit in enumerate(commits) if commit["parent_count"] == 1
    )
    last_content = commits[last_content_index]
    assert last_content["sha"] == "e2f258caf986defc1e438b860879f6755f83abf2"
    assert last_content["subject"].startswith("test(OMN-21279)")
    later_commits = commits[last_content_index + 1 :]
    assert len(later_commits) == len(merge_only_heads) == 8
    for commit in later_commits:
        assert commit["parent_count"] == 2
        assert commit["subject"].startswith(
            ("Merge branch 'dev' into", "Merge remote-tracking branch")
        )
        assert commit["sha"] in merge_only_heads
    assert {commit["sha"] for commit in later_commits} == set(merge_only_heads)

    recorded_files_sha256 = hashlib.sha256(
        json.dumps(pr_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert {head["pr_files_sha256"] for head in merge_only_heads.values()} == {
        recorded_files_sha256
    }


@pytest.mark.live_contact(
    "tests/ci/fixtures/omn20910_infra_pr5261_merge_only_rereview.json"
)
def test_rereview_rounds_on_one_diff_digest_opened_unanswered_threads(
    recorded_response: dict[str, object],
) -> None:
    threads = cast("list[dict[str, Any]]", recorded_response["threads"])
    commits = cast("list[dict[str, Any]]", recorded_response["commits"])
    merge_only_heads = cast(
        "dict[str, dict[str, str]]", recorded_response["merge_only_heads"]
    )
    first_merge = next(
        commit for commit in commits if commit["sha"] in merge_only_heads
    )
    # The recording uses fixed-width UTC timestamps, so lexical order is temporal order.
    rereviews = [
        thread
        for thread in threads
        if thread["comments"][0]["created_at"] > first_merge["committed_utc"]
    ]
    assert len(rereviews) == 28
    rounds: dict[str, list[str]] = {}
    for thread in rereviews:
        assert all(
            gate.parse_disposition(reply["body"]) is None
            for reply in thread["comments"][1:]
        )
        created_at = thread["comments"][0]["created_at"]
        head = max(
            (commit for commit in commits if commit["committed_utc"] <= created_at),
            key=lambda commit: commit["committed_utc"],
        )
        assert head["sha"] in merge_only_heads
        rounds.setdefault(head["sha"], []).append(thread["thread_id"])
    assert len(rounds) >= 5

    routing_threads = [
        thread
        for thread in rereviews
        if thread["path"] == "workspace/routing_tiers.yaml"
    ]
    assert len(routing_threads) >= 6
    fingerprints: set[str] = set()
    for thread in routing_threads:
        match = gate._FP_RE.search(thread["comments"][0]["body"])
        assert match is not None
        fingerprints.add(match.group(1))
    assert len(fingerprints) == len(routing_threads)
    assert any(
        (disposition := gate.parse_disposition(reply["body"])) is not None
        and disposition[0] == "rejected"
        for thread in threads
        if thread["path"] == "workspace/routing_tiers.yaml"
        and thread["comments"][0]["created_at"] < first_merge["committed_utc"]
        for reply in thread["comments"][1:]
    )


@pytest.mark.live_contact(
    "tests/ci/fixtures/omn20910_infra_pr5261_merge_only_rereview.json"
)
def test_diff_digest_suppresses_every_round_after_the_first(
    recorded_response: dict[str, object],
) -> None:
    pr_files = cast("list[dict[str, Any]]", recorded_response["pr_files"])
    merge_only_heads = cast(
        "dict[str, dict[str, str]]", recorded_response["merge_only_heads"]
    )
    digest = poster.pr_diff_digest(pr_files)
    findings = [
        {
            "severity": "warning",
            "file_path": "workspace/routing_tiers.yaml",
            "model": "a",
            "rule_id": "ai-reviewer:a:correctness",
            "normalized_message": "Empty routing ladder may silently misroute placements",
            "raw_message": "Empty routing ladder may silently misroute placements",
        }
    ]
    changed_files = {
        file_info["filename"]: poster.parse_right_side_lines(file_info["patch"])
        for file_info in pr_files
    }
    payload, _stats = poster.build_review(
        findings,
        changed_files,
        set(),
        suppressed_hints=0,
        models_succeeded=["a", "b"],
        models_failed=[],
        diff_digest=digest,
    )
    assert payload is not None
    assert f"<!-- {poster.DIFF_MARKER} digest={digest} -->" in payload["body"]

    reviews = [{"user": {"login": "github-actions[bot]"}, "body": payload["body"]}]
    reviewed = poster.reviewed_diff_digests(reviews)
    assert digest in reviewed
    assert len(merge_only_heads) == 8
    for sha in merge_only_heads:
        # Each head has these same recorded PR files. This is main()'s early-return
        # condition, before building or publishing another review or any threads.
        assert poster.pr_diff_digest(pr_files) in reviewed, sha

    assert (
        poster.reviewed_diff_digests(
            [{"user": {"login": "pr-author"}, "body": payload["body"]}]
        )
        == set()
    )
