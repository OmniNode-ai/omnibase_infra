# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18233 — a closed cascade bump is ignorable only on PROVEN supersession.

The closer's cited-PR merge conjunct reads ``merged_at`` and nothing else, so a
pull request that closed WITHOUT merging blocks the flip forever: a closed pull
request never becomes merged. Cascade bump pull requests carry the RELEASING
ticket's id, so every ticket whose release opens downstream bumps inherits the
block permanently. OMN-18201 is the measured case — all four criteria evidenced,
flip refused because ``omnibase_infra#3446`` was closed in favour of ``#3448``.

The fix is NOT "ignore closed bumps". A closed pull request with no proven
replacement is abandoned work and must keep blocking. The predicate here is the
four-clause one the plan review demanded, and every clause is evidence:

  1. the closed pull request carries cascade provenance naming a source package
     and a required version;
  2. a MERGED pull request in the same repository changed that package's pin or
     lockfile entry;
  3. the delivered version is >= the required version, compared as a VERSION;
  4. the delivered version is readable from the repository's own default branch.

Nothing reads a title. Any clause that cannot be resolved HOLDS the candidate.

The green fixture replays the real pair verbatim, including its ordering trap:
``#3448`` merged at 2026-09-12T03:14:52Z, roughly three hours BEFORE ``#3446``
was closed at 05:56:58Z. A predicate that required the replacement to merge
after the close would refuse the one case it was built for.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.cascade_supersession import (
    parse_cascade_provenance,
    pinned_version_from_lockfile,
    pinned_version_from_pyproject,
    resolve_verified_supersession,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect._ac_binding_support import (
    BOUND_AC_DESCRIPTION,
    bound_ac_checks,
)

pytestmark = pytest.mark.unit

_OCC_REPO = "OmniNode-ai/onex_change_control"
_PRODUCT_REPO = "OmniNode-ai/omnibase_infra"
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)

# The body of `omnibase_infra#3446`, read live 2026-09-12 via
# `gh api repos/OmniNode-ai/omnibase_infra/pulls/3446 --jq .body`. The parser
# under test is pinned to the shape the generator actually emits, and OMN-18235
# asserts the generator keeps emitting it.
#
# Reproduced verbatim EXCEPT its two trailing evidence-trailer lines, which are
# omitted deliberately: those lines are parsed by substring from pull request
# bodies elsewhere in the fleet, and spelling them inside repository content is
# a documented way to make an unrelated gate fire on documentation about the
# gate. The parser under test reads only the provenance section, so their
# absence changes nothing it asserts.
PR_3446_BODY = """## Summary

Automated dependency bump triggered by a new release of `omnibase_core` (0.47.12).

- Updates `uv.lock` via `uv lock --upgrade-package omnibase_core`
- Updates `pyproject.toml` exact pin (`==X.Y.Z`) to match, when this repo pins that exactly
- No other source code changes

## Cascade provenance (OMN-16286)

This PR carries no dedicated testing of its own beyond this repo's own
CI running against the new lockfile. Its cited evidence below is the
upstream release's own Evidence-Ticket/Evidence-Source pair, which
already proved the released package version -- reused honestly rather
than fabricated or exempted:

- Source repo: `OmniNode-ai/omnibase_core`
- Released version: `0.47.12`

## Triggered by

Release workflow: https://github.com/OmniNode-ai/omnibase_core/actions/runs/34665021693

## Test plan

- [ ] CI passes (lockfile resolution is valid)
- [ ] Downstream tests pass with the new dependency version
"""

# An ordinary product pull request body. No provenance section at all.
PLAIN_PR_BODY = """## Summary

Fixes the disk guard so it halts on free space.

## Test plan

- [x] unit tests
"""


def _pyproject(version: str) -> str:
    return (
        "[project]\n"
        'name = "omnibase-infra"\n'
        "dependencies = [\n"
        f'    "omnibase-core=={version}",\n'
        '    "redis>=6.0.0,<9.0.0",\n'
        "]\n"
    )


def _lockfile(version: str) -> str:
    return (
        "version = 1\n\n"
        "[[package]]\n"
        'name = "redis"\n'
        'version = "6.1.0"\n\n'
        "[[package]]\n"
        'name = "omnibase-core"\n'
        f'version = "{version}"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
    )


def _contents(text: str) -> dict[str, object]:
    """The shape `gh api repos/<r>/contents/<p>?ref=<ref>` returns."""
    return {
        "encoding": "base64",
        "content": base64.b64encode(text.encode()).decode(),
    }


# --------------------------------------------------------------- gh double ---


class FakeGh:
    """A GitHub double keyed on the exact `gh api` paths the resolver issues.

    Deliberately strict: an unmapped path returns an ERROR, never an empty
    result. A resolver that silently treated "I could not read it" as "there is
    nothing there" would resolve clause 2 or 4 by absence, which is the reading
    this whole predicate exists to refuse.
    """

    def __init__(
        self,
        *,
        pulls: dict[str, dict[str, object]] | None = None,
        default_branch: str = "dev",
        contents: dict[tuple[str, str], str] | None = None,
        pin_commits: list[dict[str, object]] | None = None,
        commit_pulls: dict[str, list[dict[str, object]]] | None = None,
        files_by_pr: dict[int, list[str]] | None = None,
        companions: list[dict[str, object]] | None = None,
    ) -> None:
        self.pulls = pulls or {}
        self.default_branch = default_branch
        self.contents = contents or {}
        self.pin_commits = pin_commits if pin_commits is not None else []
        self.commit_pulls = commit_pulls or {}
        self.files_by_pr = files_by_pr or {}
        self.companions = companions or []
        self.paths: list[str] = []

    async def __call__(self, args: list[str], timeout: float):
        path = args[2]
        self.paths.append(path)
        if "/files" in path and "/pulls/" in path:
            number = int(path.split("/pulls/")[1].split("/files")[0])
            return [{"filename": f} for f in self.files_by_pr.get(number, [])], ""
        if "/contents/" in path:
            file_path = path.split("/contents/", 1)[1].split("?", 1)[0]
            ref = path.rsplit("ref=", 1)[1] if "ref=" in path else self.default_branch
            text = self.contents.get((file_path, ref))
            if text is None:
                return None, f"404 no such content: {file_path}@{ref}"
            return _contents(text), ""
        if "/commits/" in path and path.endswith("/pulls"):
            sha = path.split("/commits/", 1)[1].rsplit("/pulls", 1)[0]
            return self.commit_pulls.get(sha, []), ""
        if "/commits?" in path:
            return self.pin_commits, ""
        if "/pulls/" in path and "state=closed" not in path:
            key = path.split("repos/", 1)[1]
            if key in self.pulls:
                return self.pulls[key], ""
            return None, f"no such PR: {key}"
        if "state=closed" in path:
            page = int(path.rsplit("page=", 1)[1])
            return (self.companions, "") if page == 1 else ([], "")
        if path.startswith("repos/") and path.count("/") == 2:
            return {"default_branch": self.default_branch}, ""
        return None, f"unmapped path: {path}"


def _green_gh(**overrides: object) -> FakeGh:
    """The real `#3446` / `#3448` pair, replayed."""
    payload: dict[str, object] = {
        "pulls": {
            f"{_PRODUCT_REPO}/pulls/3446": {
                "number": 3446,
                "state": "closed",
                "merged_at": None,
                "closed_at": "2026-09-12T05:56:58Z",
                "body": PR_3446_BODY,
                "user": {"login": "github-actions[bot]", "type": "Bot"},
                "title": "chore(OMN-18201): bump omnibase_core to 0.47.12",
            },
            f"{_PRODUCT_REPO}/pulls/3448": {
                "number": 3448,
                "state": "closed",
                # The ordering trap: the replacement merged ~3h BEFORE the
                # superseded bump was closed.
                "merged_at": "2026-09-12T03:14:52Z",
                "merge_commit_sha": "49572505b86e",
                "base": {"sha": "fccef30d554f"},
                "body": "",
                "user": {"login": "jonahgabriel", "type": "User"},
                "title": "chore(OMN-18202): infra core pin 0.47.12",
            },
        },
        "default_branch": "dev",
        "contents": {
            ("pyproject.toml", "dev"): _pyproject("0.47.12"),
            ("pyproject.toml", "49572505b86e"): _pyproject("0.47.12"),
            ("pyproject.toml", "fccef30d554f"): _pyproject("0.47.11"),
        },
        "pin_commits": [
            {"sha": "adb77512d052"},
            {"sha": "49572505b86e"},
        ],
        "commit_pulls": {
            "adb77512d052": [{"number": 3451, "merged_at": "2026-09-12T04:35:00Z"}],
            "49572505b86e": [{"number": 3448, "merged_at": "2026-09-12T03:14:52Z"}],
        },
        "files_by_pr": {
            3451: ["docs/runbooks/thing.md", "pyproject.toml"],
            3448: ["pyproject.toml", "uv.lock"],
        },
    }
    payload.update(overrides)
    # PR 3451 touched pyproject but did NOT move the pin: same version on both
    # sides. Clause 2 must reject it and keep walking.
    contents = payload["contents"]
    assert isinstance(contents, dict)
    contents.setdefault(("pyproject.toml", "aaaa11112222"), _pyproject("0.47.12"))
    payload["pulls"] = dict(payload["pulls"])  # type: ignore[arg-type]
    pulls = payload["pulls"]
    assert isinstance(pulls, dict)
    pulls.setdefault(
        f"{_PRODUCT_REPO}/pulls/3451",
        {
            "number": 3451,
            "merged_at": "2026-09-12T04:35:00Z",
            "merge_commit_sha": "adb77512d052",
            "base": {"sha": "aaaa11112222"},
            "body": "",
        },
    )
    contents.setdefault(("pyproject.toml", "adb77512d052"), _pyproject("0.47.12"))
    return FakeGh(**payload)  # type: ignore[arg-type]


# ------------------------------------------------------------ pure parsing ---


class TestProvenanceParsing:
    def test_the_real_bump_body_parses_to_source_package_and_required_version(
        self,
    ) -> None:
        provenance = parse_cascade_provenance(PR_3446_BODY)

        assert provenance is not None
        assert provenance.source_repo == "OmniNode-ai/omnibase_core"
        assert provenance.required_version == "0.47.12"
        # The distribution is the PEP 503 normalisation of the repo name, which
        # is what a pyproject pin and a lockfile entry actually spell.
        assert provenance.distribution == "omnibase-core"

    def test_a_body_with_no_provenance_section_parses_to_nothing(self) -> None:
        assert parse_cascade_provenance(PLAIN_PR_BODY) is None

    def test_a_provenance_section_missing_the_version_parses_to_nothing(self) -> None:
        body = "## Cascade provenance (OMN-16286)\n\n- Source repo: `OmniNode-ai/omnibase_core`\n"

        assert parse_cascade_provenance(body) is None

    def test_a_provenance_section_missing_the_source_repo_parses_to_nothing(
        self,
    ) -> None:
        body = "## Cascade provenance (OMN-16286)\n\n- Released version: `0.47.12`\n"

        assert parse_cascade_provenance(body) is None

    def test_a_provenance_line_outside_the_section_is_not_read(self) -> None:
        """The fields bind to the heading, not to the whole body.

        Otherwise any prose that happens to spell the two field names turns an
        ordinary pull request into a cascade bump.
        """
        body = (
            "## Summary\n\n"
            "- Source repo: `OmniNode-ai/omnibase_core`\n"
            "- Released version: `0.47.12`\n\n"
            "## Test plan\n\n- [x] none\n"
        )

        assert parse_cascade_provenance(body) is None

    def test_an_unparseable_version_is_refused(self) -> None:
        body = (
            "## Cascade provenance (OMN-16286)\n\n"
            "- Source repo: `OmniNode-ai/omnibase_core`\n"
            "- Released version: `not-a-version`\n"
        )

        assert parse_cascade_provenance(body) is None


class TestPinReading:
    def test_the_pyproject_pin_is_read_for_either_spelling(self) -> None:
        assert pinned_version_from_pyproject(
            _pyproject("0.47.12"), "omnibase-core"
        ) == ("0.47.12")
        underscored = _pyproject("0.47.12").replace("omnibase-core", "omnibase_core")
        assert pinned_version_from_pyproject(underscored, "omnibase-core") == "0.47.12"

    def test_an_absent_package_reads_as_no_pin(self) -> None:
        assert (
            pinned_version_from_pyproject(_pyproject("0.47.12"), "omnimarket") is None
        )

    def test_the_lockfile_entry_is_read_when_pyproject_does_not_pin(self) -> None:
        assert pinned_version_from_lockfile(_lockfile("0.47.12"), "omnibase-core") == (
            "0.47.12"
        )
        # And it reads the right package's version, not the first one it sees.
        assert pinned_version_from_lockfile(_lockfile("0.47.12"), "redis") == "6.1.0"


# ------------------------------------------------------- the predicate ------


@pytest.mark.asyncio
class TestFourClausePredicate:
    async def test_the_real_pair_resolves_every_clause(self) -> None:
        gh = _green_gh()

        verdict = await resolve_verified_supersession(
            repo=_PRODUCT_REPO,
            closed_pr_number=3446,
            closed_pr_body=PR_3446_BODY,
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is True
        assert verdict.replacement_pr == 3448
        assert verdict.required_version == "0.47.12"
        assert verdict.delivered_version == "0.47.12"
        # The record names the replacement, because "ignored" with no named
        # replacement is indistinguishable from "not checked".
        assert "3448" in verdict.detail

    async def test_the_replacement_merging_before_the_close_still_resolves(
        self,
    ) -> None:
        """The one ordering the real pair actually has (OMN-18233 note)."""
        gh = _green_gh()
        pulls = dict(gh.pulls)
        replacement = dict(pulls[f"{_PRODUCT_REPO}/pulls/3448"])
        # Explicitly BEFORE the closed bump's close time.
        assert str(replacement["merged_at"]) < str(
            pulls[f"{_PRODUCT_REPO}/pulls/3446"]["closed_at"]
        )

        verdict = await resolve_verified_supersession(
            repo=_PRODUCT_REPO,
            closed_pr_number=3446,
            closed_pr_body=PR_3446_BODY,
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is True

    async def test_no_provenance_block_fails_clause_one(self) -> None:
        gh = _green_gh()

        verdict = await resolve_verified_supersession(
            repo=_PRODUCT_REPO,
            closed_pr_number=3446,
            closed_pr_body=PLAIN_PR_BODY,
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is False
        assert "provenance" in verdict.detail.lower()
        # Clause 1 fails on the body alone; nothing else is even asked.
        assert gh.paths == []

    async def test_no_merged_replacement_fails_clause_two_and_keeps_blocking(
        self,
    ) -> None:
        """The abandoned-work case the plan review named and refused to ignore."""
        gh = _green_gh(pin_commits=[], commit_pulls={})

        verdict = await resolve_verified_supersession(
            repo=_PRODUCT_REPO,
            closed_pr_number=3446,
            closed_pr_body=PR_3446_BODY,
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is False
        assert verdict.replacement_pr == 0

    async def test_a_merged_pr_that_did_not_move_the_pin_fails_clause_two(self) -> None:
        gh = _green_gh(
            pin_commits=[{"sha": "adb77512d052"}],
            commit_pulls={
                "adb77512d052": [{"number": 3451, "merged_at": "2026-09-12T04:35:00Z"}]
            },
        )

        verdict = await resolve_verified_supersession(
            repo=_PRODUCT_REPO,
            closed_pr_number=3446,
            closed_pr_body=PR_3446_BODY,
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is False

    async def test_an_unmerged_candidate_never_satisfies_clause_two(self) -> None:
        gh = _green_gh(
            commit_pulls={
                "49572505b86e": [{"number": 3448, "merged_at": None}],
            }
        )

        verdict = await resolve_verified_supersession(
            repo=_PRODUCT_REPO,
            closed_pr_number=3446,
            closed_pr_body=PR_3446_BODY,
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is False

    async def test_a_lower_delivered_version_fails_clause_three(self) -> None:
        gh = _green_gh(
            contents={
                ("pyproject.toml", "dev"): _pyproject("0.47.11"),
                ("pyproject.toml", "49572505b86e"): _pyproject("0.47.11"),
                ("pyproject.toml", "fccef30d554f"): _pyproject("0.47.10"),
            }
        )

        verdict = await resolve_verified_supersession(
            repo=_PRODUCT_REPO,
            closed_pr_number=3446,
            closed_pr_body=PR_3446_BODY,
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is False
        assert "0.47.11" in verdict.detail

    async def test_clause_three_compares_as_a_version_not_as_a_string(self) -> None:
        """`"0.47.9" > "0.47.12"` lexically; 0.47.12 > 0.47.9 as a version.

        A string comparison here would refuse the single most common real
        shape: a two-digit patch superseding a one-digit one.
        """
        body = PR_3446_BODY.replace(
            "Released version: `0.47.12`", "Released version: `0.47.9`"
        )
        gh = _green_gh()

        verdict = await resolve_verified_supersession(
            repo=_PRODUCT_REPO,
            closed_pr_number=3446,
            closed_pr_body=body,
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is True
        assert verdict.delivered_version == "0.47.12"
        assert verdict.required_version == "0.47.9"

    async def test_an_unreadable_default_branch_pin_fails_clause_four(self) -> None:
        gh = _green_gh(
            contents={
                ("pyproject.toml", "49572505b86e"): _pyproject("0.47.12"),
                ("pyproject.toml", "fccef30d554f"): _pyproject("0.47.11"),
            }
        )

        verdict = await resolve_verified_supersession(
            repo=_PRODUCT_REPO,
            closed_pr_number=3446,
            closed_pr_body=PR_3446_BODY,
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is False
        assert verdict.delivered_version == ""

    async def test_the_predicate_reads_no_title_anywhere(self) -> None:
        """Retitle everything in sight; the verdict must not move."""
        gh = _green_gh()
        for key, payload in gh.pulls.items():
            payload["title"] = "totally unrelated prose"
            gh.pulls[key] = payload

        verdict = await resolve_verified_supersession(
            repo=_PRODUCT_REPO,
            closed_pr_number=3446,
            closed_pr_body=PR_3446_BODY,
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is True

    async def test_the_lockfile_carries_the_pin_when_pyproject_does_not(self) -> None:
        gh = _green_gh(
            contents={
                ("uv.lock", "dev"): _lockfile("0.47.12"),
                ("uv.lock", "49572505b86e"): _lockfile("0.47.12"),
                ("uv.lock", "fccef30d554f"): _lockfile("0.47.11"),
            }
        )

        verdict = await resolve_verified_supersession(
            repo=_PRODUCT_REPO,
            closed_pr_number=3446,
            closed_pr_body=PR_3446_BODY,
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is True
        assert verdict.delivered_version == "0.47.12"


# ------------------------------------------------- wired into the closer ----


def _issue(identifier: str, description: str) -> dict[str, object]:
    return {
        "id": "issue-1",
        "identifier": identifier,
        "state": {"id": "s1", "name": "In Progress", "type": "started"},
        "labels": {"nodes": []},
        "team": {"id": "team-1"},
        "description": description,
        "children": {"nodes": []},
    }


class FakeLinear:
    def __init__(self, issues: dict[str, dict[str, object]]) -> None:
        self._issues = issues
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, object] | None:
        return self._issues.get(ticket_id)

    async def fetch_done_state_id(self, team_id: str) -> str | None:
        return "state-done"

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        return True

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return True

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        return tuple(body for target, body in self.comments if target == issue_id)

    async def fetch_issue_history(
        self, issue_id: str, page_size: int, max_pages: int
    ) -> tuple[list[dict[str, object]] | None, str]:
        # The bound readback (OMN-17658b) requires a NEW completed segment to
        # appear after the write. The history moves when the write happens,
        # not on the second read.
        if any(target == issue_id for target, _state in self.state_updates):
            return [
                {
                    "id": "entry-flip",
                    "createdAt": datetime.now(tz=UTC).isoformat(),
                    "actorId": "actor-1",
                    "botActor": None,
                    "fromState": {"type": "started"},
                    "toState": {"type": "completed"},
                }
            ], ""
        return [], ""


def _receipt() -> dict[str, object]:
    verdict: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "ticket_id": "OMN-18201",
        "status": "verified",
        "dry_run": False,
        "checks": bound_ac_checks(),
        "total_checks": 2,
        "verified_count": 2,
        "failed_count": 0,
        "skipped_count": 0,
        "superseded_count": 0,
        "non_probative_count": 0,
        "behavior_proving_count": 1,
        "error_message": None,
    }
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success",
        "correlation_id": str(uuid4()),
        "run_id": str(uuid4()),
        "exit_code": 0,
        "duration_ms": 1,
        "result": verdict,
        "result_model": _DOD_VERIFY_STATE_MODEL,
    }


def _dod_fake(receipt: dict[str, object]):
    async def run_dod(ticket_id: str, cwd: str, timeout: int):
        return receipt, 0, ""

    return run_dod


def _companion(number: int, ticket: str) -> dict[str, object]:
    recent = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({ticket}): OCC companion for {_PRODUCT_REPO}#3446",
        "updated_at": recent,
        "merged_at": recent,
    }


def _request() -> ModelEvidenceAutocloseSweepRequest:
    return ModelEvidenceAutocloseSweepRequest(
        correlation_id=uuid4(),
        occ_repo=_OCC_REPO,
        lookback_hours=24,
        apply=True,
    )


@pytest.mark.asyncio
class TestCloserIntegration:
    """AC1 on the whole closer, not only on the predicate in isolation."""

    async def test_omn_18201_shape_flips_once_supersession_is_proven(self) -> None:
        linear = FakeLinear(
            {
                "OMN-18201": _issue(
                    "OMN-18201",
                    BOUND_AC_DESCRIPTION + f"\n\nCascade bump: {_PRODUCT_REPO}#3446\n",
                )
            }
        )
        gh = _green_gh(
            companions=[_companion(9159, "OMN-18201")],
            files_by_pr={
                9159: ["contracts/OMN-18201.yaml"],
                3451: ["docs/runbooks/thing.md", "pyproject.toml"],
                3448: ["pyproject.toml", "uv.lock"],
            },
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,  # type: ignore[arg-type]
            autoclose_disabled=False,
            run_gh_command=gh,
            run_dod_verify_command=_dod_fake(_receipt()),
        )

        # Tick one arms the OMN-18056 re-draw; tick two, on the same verdict,
        # flips. Both must get past the cited-PR conjunct, so the supersession
        # predicate is exercised twice here rather than once.
        armed = await handler.handle(_request())
        assert [o.decision for o in armed.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_REDRAW_PENDING
        ]

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]
        assert result.tickets_flipped == 1

    async def test_a_closed_bump_with_no_proven_replacement_still_blocks(self) -> None:
        linear = FakeLinear(
            {
                "OMN-18201": _issue(
                    "OMN-18201",
                    BOUND_AC_DESCRIPTION + f"\n\nCascade bump: {_PRODUCT_REPO}#3446\n",
                )
            }
        )
        gh = _green_gh(
            companions=[_companion(9159, "OMN-18201")],
            files_by_pr={9159: ["contracts/OMN-18201.yaml"]},
            pin_commits=[],
            commit_pulls={},
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,  # type: ignore[arg-type]
            autoclose_disabled=False,
            run_gh_command=gh,
            run_dod_verify_command=_dod_fake(_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_REFERENCED_PR_UNMERGED
        ]
        assert result.tickets_flipped == 0
        assert linear.state_updates == []

    async def test_an_ordinary_open_product_pr_is_untouched_by_this_change(
        self,
    ) -> None:
        """The narrowing must not become a way past the original conjunct."""
        linear = FakeLinear(
            {
                "OMN-18201": _issue(
                    "OMN-18201",
                    BOUND_AC_DESCRIPTION + f"\n\nShipped in {_PRODUCT_REPO}#3499\n",
                )
            }
        )
        gh = _green_gh(companions=[_companion(9159, "OMN-18201")])
        gh.pulls[f"{_PRODUCT_REPO}/pulls/3499"] = {
            "number": 3499,
            "state": "open",
            "merged_at": None,
            "body": PR_3446_BODY,
        }
        gh.files_by_pr = {9159: ["contracts/OMN-18201.yaml"]}
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,  # type: ignore[arg-type]
            autoclose_disabled=False,
            run_gh_command=gh,
            run_dod_verify_command=_dod_fake(_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_REFERENCED_PR_UNMERGED
        ]
        assert "state=OPEN" in result.outcomes[0].reason
