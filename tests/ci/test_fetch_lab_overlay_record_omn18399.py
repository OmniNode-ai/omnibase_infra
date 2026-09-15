# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18399 -- descendant tolerance for the onex-lab overlay record reader.

Under a busy ``dev`` branch the deploy agent keys its lab-overlay record by
whatever HEAD it resolved when it reached the job, not by the sha of the
command that triggered a given ``runtime-rebuild-trigger.yml`` run. On a busy
branch that is a LATER descendant, so `scripts/ci/fetch_lab_overlay_record.py`'s
exact-sha lookup (``GET /lab-overlay/{sha}``) never finds a record at all --
measured live on run 34976826412: 54 consecutive transport timeouts across the
full 1800s window.

This mirrors OMN-18388's fix for the compose-dev lane (a descendant of the
merge sha converges; an ancestor or unrelated revision stays stale), applied
here via a new ``/lab-overlay-latest`` agent endpoint and the same GitHub
compare-API ancestry resolution `check_dev_lane_staleness.py` already uses.

Hermetic: `resolve_relation` and `resolve_via_latest` are driven over fixtures
with `compare`/`fetch` stand-ins. No network, no `gh` subprocess, no deploy
agent.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable

import pytest

from scripts.ci import fetch_lab_overlay_record as reader

pytestmark = pytest.mark.unit

REPO = "OmniNode-ai/omnibase_infra"
REQUESTED_SHA = "1" * 40
DESCENDANT_SHA = "2" * 40
ANCESTOR_SHA = "3" * 40
UNRELATED_SHA = "4" * 40

VALID_CHECKS = [{"name": "lab_overlay_applied", "ok": True, "evidence": "applied"}]


class TestResolveRelation:
    """AC2's four cases, at the ancestry-resolution layer."""

    def test_identical_shas_never_call_compare(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(repo: str, base: str, head: str) -> str:
            raise AssertionError("must not call gh api for an identical sha")

        monkeypatch.setattr(reader, "_gh_compare_status", _boom)
        assert (
            reader.resolve_relation(REPO, REQUESTED_SHA, REQUESTED_SHA)
            == reader.RELATION_IDENTICAL
        )

    @pytest.mark.parametrize(
        ("compare_status", "expected_relation"),
        [
            ("ahead", reader.RELATION_DESCENDANT),
            ("behind", reader.RELATION_ANCESTOR),
            ("diverged", reader.RELATION_UNRELATED),
        ],
    )
    def test_relation_mapping(
        self,
        monkeypatch: pytest.MonkeyPatch,
        compare_status: str,
        expected_relation: str,
    ) -> None:
        monkeypatch.setattr(
            reader, "_gh_compare_status", lambda repo, base, head: compare_status
        )
        assert (
            reader.resolve_relation(REPO, REQUESTED_SHA, DESCENDANT_SHA)
            == expected_relation
        )


class TestResolveViaLatest:
    """AC1/AC2/AC3 -- the fallback function `poll()` calls once the exact-sha
    window is exhausted."""

    @staticmethod
    def _fetch_returning(
        sha: str, checks: list[dict[str, object]] | None = None
    ) -> Callable[[str, float], tuple[int, str]]:
        payload = {"sha": sha, "checks": checks if checks is not None else VALID_CHECKS}

        def _fetch(url: str, timeout: float) -> tuple[int, str]:
            return 200, json.dumps(payload)

        return _fetch

    def test_no_latest_record_returns_none(self) -> None:
        """404 (or any non-200) -- nothing to compare against, caller keeps its
        own timeout message."""

        def _fetch_404(url: str, timeout: float) -> tuple[int, str]:
            return 404, json.dumps({"error": "no lab-overlay record exists yet"})

        result = reader.resolve_via_latest(
            base_url="http://agent:8098",
            repo=REPO,
            sha=REQUESTED_SHA,
            request_timeout_seconds=5.0,
            fetch=_fetch_404,
        )
        assert result is None

    def test_descendant_is_accepted_and_evidence_names_both_shas(self) -> None:
        """AC1 + AC3: a descendant is accepted, the sha stays the REQUESTED
        one (asserted by the caller, `poll`, never rewriting it), and the
        evidence names both shas plus the word 'descendant'."""
        result = reader.resolve_via_latest(
            base_url="http://agent:8098",
            repo=REPO,
            sha=REQUESTED_SHA,
            request_timeout_seconds=5.0,
            fetch=self._fetch_returning(DESCENDANT_SHA),
            compare=lambda repo, requested, observed: reader.RELATION_DESCENDANT,
        )
        assert result is not None
        lead = result[0]
        assert lead["ok"] is True
        assert REQUESTED_SHA in lead["evidence"]
        assert DESCENDANT_SHA in lead["evidence"]
        assert "descendant" in lead["evidence"]
        # The original record's own checks travel with the accepting note.
        assert result[1:] == VALID_CHECKS

    def test_identical_is_accepted_too(self) -> None:
        result = reader.resolve_via_latest(
            base_url="http://agent:8098",
            repo=REPO,
            sha=REQUESTED_SHA,
            request_timeout_seconds=5.0,
            fetch=self._fetch_returning(REQUESTED_SHA),
            compare=lambda repo, requested, observed: reader.RELATION_IDENTICAL,
        )
        assert result is not None
        assert result[0]["ok"] is True

    def test_unrelated_latest_record_fails(self) -> None:
        result = reader.resolve_via_latest(
            base_url="http://agent:8098",
            repo=REPO,
            sha=REQUESTED_SHA,
            request_timeout_seconds=5.0,
            fetch=self._fetch_returning(UNRELATED_SHA),
            compare=lambda repo, requested, observed: reader.RELATION_UNRELATED,
        )
        assert result is not None
        assert len(result) == 1
        assert result[0]["ok"] is False
        assert "UNRELATED" in result[0]["evidence"]

    def test_ancestor_latest_record_fails_as_not_yet_applied(self) -> None:
        """AC2: an ancestor of the requested sha means the change has NOT been
        applied yet -- this must stay a FAIL, the same as the stale direction
        OMN-18388 kept for the compose lane."""
        result = reader.resolve_via_latest(
            base_url="http://agent:8098",
            repo=REPO,
            sha=REQUESTED_SHA,
            request_timeout_seconds=5.0,
            fetch=self._fetch_returning(ANCESTOR_SHA),
            compare=lambda repo, requested, observed: reader.RELATION_ANCESTOR,
        )
        assert result is not None
        assert len(result) == 1
        assert result[0]["ok"] is False
        assert "ANCESTOR" in result[0]["evidence"]
        assert "not applied yet" in result[0]["evidence"]

    def test_unresolvable_ancestry_fails_rather_than_crashing_or_passing(self) -> None:
        def _raising_compare(repo: str, requested: str, observed: str) -> str:
            raise RuntimeError("gh api compare failed (exit 1): 502 Bad Gateway")

        result = reader.resolve_via_latest(
            base_url="http://agent:8098",
            repo=REPO,
            sha=REQUESTED_SHA,
            request_timeout_seconds=5.0,
            fetch=self._fetch_returning(DESCENDANT_SHA),
            compare=_raising_compare,
        )
        assert result is not None
        assert result[0]["ok"] is False
        assert "could not be resolved" in result[0]["evidence"]

    def test_a_malformed_latest_record_returns_none_not_a_crash(self) -> None:
        def _fetch_malformed(url: str, timeout: float) -> tuple[int, str]:
            return 200, json.dumps({"sha": DESCENDANT_SHA, "checks": []})

        result = reader.resolve_via_latest(
            base_url="http://agent:8098",
            repo=REPO,
            sha=REQUESTED_SHA,
            request_timeout_seconds=5.0,
            fetch=_fetch_malformed,
        )
        assert result is None


class TestPollFallsBackOnTimeout:
    """`poll()`'s own integration of the fallback -- exercised at the seam,
    without a real HTTP server. `fetch_once` is a bare module-level name
    `poll()` calls internally, so it is monkeypatched on the module rather
    than injected."""

    def test_exact_match_never_invokes_the_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []

        def _fetch(url: str, timeout: float) -> tuple[int, str]:
            return 200, json.dumps({"sha": REQUESTED_SHA, "checks": VALID_CHECKS})

        def _resolve_latest(**kwargs: object) -> None:
            calls.append(kwargs)

        monkeypatch.setattr(reader, "fetch_once", _fetch)

        result = reader.poll(
            base_url="http://agent:8098",
            sha=REQUESTED_SHA,
            repo=REPO,
            wait_seconds=10,
            poll_interval_seconds=1,
            request_timeout_seconds=1.0,
            out=io.StringIO(),
            sleep=lambda _s: None,
            now=iter([0.0, 0.0]).__next__,
            resolve_latest=_resolve_latest,
        )
        assert result == VALID_CHECKS
        assert calls == []

    def test_timeout_falls_back_to_the_latest_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fetch_404(url: str, timeout: float) -> tuple[int, str]:
            return 404, json.dumps({"error": "no record", "sha": REQUESTED_SHA})

        accepting_result = [
            {"name": reader.RECORD_CHECK, "ok": True, "evidence": "descendant accepted"}
        ]

        def _resolve_latest(**kwargs: object) -> list[dict[str, object]]:
            assert kwargs["sha"] == REQUESTED_SHA
            assert kwargs["repo"] == REPO
            return accepting_result

        monkeypatch.setattr(reader, "fetch_once", _fetch_404)
        # A tiny deadline that is already exhausted on the first check.
        clock = iter([0.0, 100.0])

        result = reader.poll(
            base_url="http://agent:8098",
            sha=REQUESTED_SHA,
            repo=REPO,
            wait_seconds=1,
            poll_interval_seconds=1,
            request_timeout_seconds=1.0,
            out=io.StringIO(),
            sleep=lambda _s: None,
            now=clock.__next__,
            resolve_latest=_resolve_latest,
        )
        assert result == accepting_result

    def test_timeout_with_no_fallback_keeps_the_original_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fetch_404(url: str, timeout: float) -> tuple[int, str]:
            return 404, json.dumps({"error": "no record", "sha": REQUESTED_SHA})

        monkeypatch.setattr(reader, "fetch_once", _fetch_404)
        clock = iter([0.0, 100.0])

        result = reader.poll(
            base_url="http://agent:8098",
            sha=REQUESTED_SHA,
            repo=REPO,
            wait_seconds=1,
            poll_interval_seconds=1,
            request_timeout_seconds=1.0,
            out=io.StringIO(),
            sleep=lambda _s: None,
            now=clock.__next__,
            resolve_latest=lambda **_kwargs: None,
        )
        assert len(result) == 1
        assert result[0]["ok"] is False
        assert "never returned a record" in result[0]["evidence"]
