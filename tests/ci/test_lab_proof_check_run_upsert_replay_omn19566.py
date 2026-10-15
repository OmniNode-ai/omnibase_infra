# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay of GitHub's own check-runs listing through the check-run poster (OMN-19566).

``upsert_check_run`` decides between PATCH and POST from the list-check-runs
answer for a head. The recording is that answer as GitHub returned it for the
heads of merged omnibase_infra pull requests, including a head that really
carried two runs of one name (a cancelled one and the rerun that replaced it).
Replayed as the transport's answer, the poster must read the real payload
shape, update the newest real run in place and name the older one.
"""

from __future__ import annotations

import io
import json
from typing import Any, cast

import pytest

from scripts.ci import lab_pass_receipt as receipts

pytestmark = pytest.mark.unit

REPO = "OmniNode-ai/omnibase_infra"
CHECK_NAME = "repo-evidence / dod-verify"


class _Answer(io.BytesIO):
    def __enter__(self) -> _Answer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _replaying(listing: list[dict[str, Any]], seen: list[tuple[str, str]]) -> Any:
    def urlopen(request: Any, timeout: float) -> _Answer:
        seen.append((request.get_method(), request.full_url))
        if request.get_method() == "GET":
            return _Answer(json.dumps({"check_runs": listing}).encode())
        return _Answer(
            json.dumps({"id": int(request.full_url.rsplit("/", 1)[1])}).encode()
        )

    return urlopen


def _head(recorded: dict[str, object], pr: int) -> dict[str, Any]:
    heads = cast("list[dict[str, Any]]", recorded["heads"])
    return next(h for h in heads if h["pr"] == pr)


@pytest.mark.live_contact("tests/ci/fixtures/omn20074_repo_evidence_check_runs.json")
def test_a_real_head_with_two_runs_of_one_name_updates_the_newest(
    recorded_response: dict[str, object],
) -> None:
    head = _head(recorded_response, 4715)
    listing = [r for r in head["check_runs"] if r["name"] == CHECK_NAME]
    ids = sorted((r["id"] for r in listing), reverse=True)
    # positive control: the recording really holds a duplicate on this head
    assert len(ids) == 2
    seen: list[tuple[str, str]] = []
    body = {"name": CHECK_NAME, "head_sha": head["head_sha"], "status": "completed"}
    posted = receipts.upsert_check_run(
        body, repo=REPO, token="t", urlopen=_replaying(listing, seen)
    )
    assert posted.action == "updated"
    assert posted.check_run["id"] == ids[0]
    assert posted.superseded_ids == (ids[1],)
    assert [method for method, _ in seen] == ["GET", "PATCH"]
    assert seen[1][1].endswith(f"/repos/{REPO}/check-runs/{ids[0]}")


@pytest.mark.live_contact("tests/ci/fixtures/omn20074_repo_evidence_check_runs.json")
def test_a_real_head_with_one_run_of_a_name_updates_that_run_alone(
    recorded_response: dict[str, object],
) -> None:
    head = _head(recorded_response, 4702)
    listing = [r for r in head["check_runs"] if r["name"] == CHECK_NAME]
    assert len(listing) == 1
    seen: list[tuple[str, str]] = []
    body = {"name": CHECK_NAME, "head_sha": head["head_sha"], "status": "completed"}
    posted = receipts.upsert_check_run(
        body, repo=REPO, token="t", urlopen=_replaying(listing, seen)
    )
    assert (posted.action, posted.superseded_ids) == ("updated", ())
    assert posted.check_run["id"] == listing[0]["id"]
