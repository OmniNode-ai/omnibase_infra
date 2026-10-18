# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21146 — ``--offer`` resolves the companion without GitHub search.

The closer found an offered ticket's merged evidence companion with
``gh api search/issues``. The GitHub-search guard refuses that call on every
host, so an offer ended in ERROR_GITHUB_API on all of them (measured for
OMN-20113: dod_verify returned verified 3 of 3, and the closer still refused).

Four groups, each against REAL git repositories built in ``tmp_path`` and the
real resolution from the request's ``clone_root`` / ``pr_watcher_state_path``,
so a resolver that reads the wrong ref, the wrong key or the wrong file list
fails here and not live:

1. the companion is read from the canonical clone and the PR watcher's state;
2. the gateway node answers what the local sources did not;
3. the change-control copy absent: the product repository's contract is accepted;
4. negative controls: no companion and no product contract still refuses with a
   typed reason, and a decoy that names the ticket in passing is discarded.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect import (
    offer_companion_sources as sources,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_arm import (
    EnumEvidenceAutocloseArm,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_omn_16106_offer_selection import (
    _all_green,
    _RecordingLinear,
)

pytestmark = pytest.mark.unit

_OWNER = "OmniNode-ai"
_OCC_REPO = f"{_OWNER}/onex_change_control"
_TICKET = "OMN-20113"
_COMPANION_PR = 12455
_PRODUCT_TICKET = "OMN-20999"


def _git(repo: Path, *args: str, when: datetime | None = None) -> str:
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.test",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.test",
        "HOME": str(repo),
    }
    if when is not None:
        stamp = when.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        env["GIT_AUTHOR_DATE"] = stamp
        env["GIT_COMMITTER_DATE"] = stamp
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env({**os.environ, **env}),
    )
    return done.stdout


def _make_clone(root: Path, name: str) -> Path:
    """A repo on branch ``dev`` whose ``origin/HEAD`` points at ``origin/dev``."""
    repo = root / name
    repo.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q", "-b", "dev", str(repo)],
        check=True,
        env=scrub_git_location_env(),
    )
    _git(repo, "commit", "-q", "--allow-empty", "-m", "root")
    _git(repo, "update-ref", "refs/remotes/origin/dev", "HEAD")
    _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/dev")
    return repo


def _land(
    repo: Path, subject: str, files: list[str], when: datetime | None = None
) -> str:
    """One squash commit on dev touching ``files``, then moved onto origin/dev."""
    for path in files:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{subject}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", subject, when=when)
    sha = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "update-ref", "refs/remotes/origin/dev", sha)
    return sha


# Where this test's clones and watcher state live; set by the fixture below.
_SANDBOX: dict[str, str] = {}


def _request(**overrides: Any) -> ModelEvidenceAutocloseSweepRequest:
    defaults: dict[str, Any] = {
        "correlation_id": uuid4(),
        "evidence_repo": _OCC_REPO,
        "apply": True,
        "offer_tickets": (_TICKET,),
        "clone_root": _SANDBOX["clone_root"],
        # An absent watcher file is "no state to read", never a stale default.
        "pr_watcher_state_path": _SANDBOX["watcher"],
    }
    defaults.update(overrides)
    return ModelEvidenceAutocloseSweepRequest(**defaults)


class _Gh:
    """A ``gh api`` double that refuses what the guard refuses, and records."""

    def __init__(self, files: dict[int, list[str]] | None = None) -> None:
        self.paths: list[str] = []
        self._files = files or {}

    async def __call__(self, args: list[str], timeout: float) -> tuple[Any, str]:
        path = args[2]
        self.paths.append(path)
        if path.startswith("search/issues"):
            raise AssertionError("GitHub search was called: the guard refuses it")
        if "/files" in path:
            number = int(path.split("/pulls/", 1)[1].split("/", 1)[0])
            if number in self._files:
                return [{"filename": name} for name in self._files[number]], ""
        raise AssertionError(f"unexpected gh api path: {path}")


class _Gateway:
    """An ``onex scm`` double answering the shapes the gateway node returns."""

    def __init__(
        self,
        heads: list[dict[str, Any]] | None = None,
        views: dict[int, dict[str, Any]] | None = None,
        error: str = "",
    ) -> None:
        self.calls: list[list[str]] = []
        self._heads = heads or []
        self._views = views or {}
        self._error = error

    async def __call__(self, args: list[str], timeout: float) -> tuple[Any, str]:
        self.calls.append(args)
        if self._error:
            return None, self._error
        verb, target = args[2], args[3]
        if verb == "pr-heads":
            return {
                "outcome": "completed",
                "operation": "pr_heads",
                "result": {"operation": "pr_heads", "prs": self._heads},
            }, ""
        number = int(target.rsplit("#", 1)[1])
        return {
            "outcome": "completed",
            "operation": "pr_text",
            "result": self._views[number],
        }, ""


def _handler(
    linear: _RecordingLinear,
    gh: _Gh,
    gateway: _Gateway | None = None,
) -> tuple[HandlerEvidenceAutocloseSweep, list[str]]:
    verified: list[str] = []

    async def fake_dod_verify(
        ticket_id: str, cwd: str, timeout: float
    ) -> tuple[dict[str, Any], int, str]:
        verified.append(ticket_id)
        return _all_green(ticket_id), 0, ""

    handler = HandlerEvidenceAutocloseSweep(
        linear_client=cast("Any", linear),
        run_gh_command=gh,
        run_dod_verify_command=fake_dod_verify,
        run_gateway_command=gateway or _Gateway(),
    )
    return handler, verified


@pytest.fixture
def clone_root(tmp_path: Path) -> Path:
    root = tmp_path / "clones"
    root.mkdir()
    _SANDBOX["clone_root"] = str(root)
    _SANDBOX["watcher"] = str(tmp_path / "no-watcher.json")
    return root


def _write_watcher(
    path: Path, repo_name: str, number: int, title: str, merged_at: str
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "prs": {
                    f"{repo_name}#{number}": {
                        "facts": {
                            "number": number,
                            "state": "MERGED",
                            "title": title,
                            "merged_at": merged_at,
                        }
                    }
                },
            }
        )
    )


async def _two_ticks(
    handler: HandlerEvidenceAutocloseSweep, request: ModelEvidenceAutocloseSweepRequest
) -> Any:
    """The first eligible observation arms the re-draw; the flip is the second."""
    armed = await handler.handle(request)
    assert (
        armed.outcomes[0].decision
        is EnumEvidenceAutocloseDecision.SKIPPED_REDRAW_PENDING
    ), armed.outcomes[0]
    return await handler.handle(request)


# =====================================================================
# GROUP 1 — the companion is read from the canonical clone and the watcher
# =====================================================================


async def test_the_offered_companion_is_read_from_the_clone_without_a_github_read(
    clone_root: Path,
) -> None:
    """The blocker, executed against a real repository.

    The companion is a squash commit on the change-control repo's DEFAULT branch
    (``dev`` here, ``origin/HEAD`` says so — ``main`` would find nothing). Its
    number comes from the ``(#N)`` subject, its changed files from the commit,
    so neither the search nor the file listing touches GitHub.
    """
    occ = _make_clone(clone_root, "onex_change_control")
    _land(
        occ,
        f"evidence({_TICKET}): OCC batch window for {_OWNER}/omnibase_infra (#{_COMPANION_PR})",
        [f"contracts/{_TICKET}.yaml"],
        when=datetime.now(tz=UTC) - timedelta(days=12),
    )
    linear = _RecordingLinear()
    gh = _Gh()  # no files registered: any file listing read would raise
    handler, verified = _handler(linear, gh)

    result = await _two_ticks(handler, _request())

    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert outcome.ticket_id == _TICKET
    assert outcome.companion_pr_number == _COMPANION_PR
    assert outcome.enumeration_arm is EnumEvidenceAutocloseArm.OFFER
    assert verified == [_TICKET, _TICKET]
    assert gh.paths == [], "the offer path read GitHub: " + ", ".join(gh.paths)


async def test_the_watchers_merge_time_wins_and_the_clone_supplies_the_files(
    clone_root: Path, tmp_path: Path
) -> None:
    occ = _make_clone(clone_root, "onex_change_control")
    _land(
        occ,
        f"evidence({_TICKET}): OCC batch window (#{_COMPANION_PR})",
        [f"contracts/{_TICKET}.yaml"],
    )
    watcher = tmp_path / "watcher.json"
    _write_watcher(
        watcher,
        "onex_change_control",
        _COMPANION_PR,
        f"evidence({_TICKET}): OCC batch window (#{_COMPANION_PR})",
        "2026-10-03T00:42:57Z",
    )

    watcher_hits, watcher_answered = sources.read_watcher_candidates(
        watcher, _OCC_REPO, _TICKET
    )
    clone_hits, clone_answered = sources.read_clone_candidates(
        occ, _OCC_REPO, _TICKET, sources.run_git_real
    )
    merged = sources.merge_candidates(watcher_hits, clone_hits)

    assert watcher_answered and clone_answered
    assert len(merged) == 1
    assert merged[0]["number"] == _COMPANION_PR
    assert merged[0]["merged_at"] == "2026-10-03T00:42:57Z"
    assert merged[0]["files"] == [f"contracts/{_TICKET}.yaml"]


async def test_a_watcher_only_companion_resolves_and_lists_its_files_through_gh(
    clone_root: Path, tmp_path: Path
) -> None:
    """No clone: the watcher alone names the companion; its file list is a plain
    ``pulls/N/files`` read, never a search."""
    watcher = tmp_path / "watcher.json"
    _write_watcher(
        watcher,
        "onex_change_control",
        _COMPANION_PR,
        f"evidence({_TICKET}): OCC batch window",
        (datetime.now(tz=UTC) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    gh = _Gh(files={_COMPANION_PR: [f"contracts/{_TICKET}.yaml"]})
    handler, _verified = _handler(_RecordingLinear(), gh)

    result = await handler.handle(_request(pr_watcher_state_path=str(watcher)))

    assert result.outcomes[0].companion_pr_number == _COMPANION_PR
    assert not any(path.startswith("search/issues") for path in gh.paths)
    assert any("/files" in path for path in gh.paths)


async def test_the_gateway_is_not_asked_when_the_local_sources_answered(
    clone_root: Path,
) -> None:
    occ = _make_clone(clone_root, "onex_change_control")
    _land(
        occ,
        f"evidence({_TICKET}): OCC batch window (#{_COMPANION_PR})",
        [f"contracts/{_TICKET}.yaml"],
    )
    gateway = _Gateway()
    handler, _verified = _handler(_RecordingLinear(), _Gh(), gateway)

    result = await handler.handle(_request())

    assert result.outcomes[0].companion_pr_number == _COMPANION_PR
    assert gateway.calls == []


# =====================================================================
# GROUP 2 — the gateway answers what the local sources did not
# =====================================================================


async def test_the_gateway_resolves_a_ticket_named_companion_the_clone_lacks(
    clone_root: Path,
) -> None:
    _make_clone(clone_root, "onex_change_control")  # a clone that is behind
    merged_at = (datetime.now(tz=UTC) - timedelta(days=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    gateway = _Gateway(
        heads=[
            {
                "number": 13821,
                "state": "MERGED",
                "head_ref": "omn-20113-occ-companion",
                "merged_at": merged_at,
            },
            {
                "number": 13822,
                "state": "MERGED",
                "head_ref": "auto/window-omninode-ai-omnibase_infra-occ-autobind",
                "merged_at": merged_at,
            },
        ],
        views={
            13821: {
                "number": 13821,
                "title": f"evidence({_TICKET}): OCC companion",
                "state": "MERGED",
                "merged_at": merged_at,
            }
        },
    )
    gh = _Gh(files={13821: [f"contracts/{_TICKET}.yaml"]})
    handler, _verified = _handler(_RecordingLinear(), gh, gateway)

    result = await handler.handle(_request())

    assert result.outcomes[0].companion_pr_number == 13821
    verbs = [call[2] for call in gateway.calls]
    assert verbs == ["pr-heads", "pr-view"], (
        "only the PR whose branch names the ticket is read, and 13822 is not"
    )
    assert all("--source" in call and "bus" in call for call in gateway.calls)


async def test_a_gateway_that_cannot_answer_is_a_refusal_and_not_an_absence(
    clone_root: Path,
) -> None:
    _make_clone(clone_root, "onex_change_control")
    handler, verified = _handler(
        _RecordingLinear(), _Gh(), _Gateway(error="bus: no broker")
    )

    result = await handler.handle(_request())

    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.ERROR_GITHUB_API
    assert "bus: no broker" in outcome.reason
    assert verified == []


# =====================================================================
# GROUP 3 — the change-control copy is absent: the product contract stands
# =====================================================================


async def test_the_product_repo_contract_is_accepted_when_no_companion_exists(
    clone_root: Path,
) -> None:
    _make_clone(clone_root, "onex_change_control")
    product = _make_clone(clone_root, "omnibase_infra")
    sha = _land(
        product,
        f"feat({_PRODUCT_TICKET}): the thing (#77)",
        [f"contracts/{_PRODUCT_TICKET}.yaml"],
    )
    linear = _RecordingLinear()
    handler, verified = _handler(linear, _Gh())

    result = await _two_ticks(handler, _request(offer_tickets=(_PRODUCT_TICKET,)))

    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert outcome.ticket_id == _PRODUCT_TICKET
    assert outcome.companion_pr_number == 0
    assert sha in outcome.companion_pr_url
    assert f"{_OWNER}/omnibase_infra" in outcome.companion_pr_url
    assert outcome.enumeration_arm is EnumEvidenceAutocloseArm.OFFER
    assert verified == [_PRODUCT_TICKET, _PRODUCT_TICKET]


async def test_a_contract_on_two_product_repositories_is_not_chosen_between(
    clone_root: Path,
) -> None:
    _make_clone(clone_root, "onex_change_control")
    for name in ("omnibase_infra", "omnimarket"):
        _land(
            _make_clone(clone_root, name),
            f"feat({_PRODUCT_TICKET}): the thing (#1)",
            [f"contracts/{_PRODUCT_TICKET}.yaml"],
        )
    handler, verified = _handler(_RecordingLinear(), _Gh())

    result = await handler.handle(_request(offer_tickets=(_PRODUCT_TICKET,)))

    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.ERROR_GITHUB_API
    assert "omnibase_infra" in outcome.reason and "omnimarket" in outcome.reason
    assert verified == []


async def test_the_change_control_repos_own_contract_is_not_a_product_contract(
    clone_root: Path,
) -> None:
    """The change-control clone holds ``contracts/<ticket>.yaml`` for every
    ticket; it is the copy being retired, not a product repository's."""
    occ = _make_clone(clone_root, "onex_change_control")
    _land(
        occ,
        f"chore: seed ({_PRODUCT_TICKET}) (#5)",
        [f"contracts/{_PRODUCT_TICKET}.yaml"],
    )

    contract, error = sources.find_product_contract(
        clone_root, _OWNER, "onex_change_control", _PRODUCT_TICKET, sources.run_git_real
    )

    assert (contract, error) == (None, "")


# =====================================================================
# GROUP 4 — negative controls
# =====================================================================


async def test_no_companion_and_no_product_contract_refuses_with_a_typed_reason(
    clone_root: Path,
) -> None:
    """The control for every positive above: the same setup, nothing to find."""
    _make_clone(clone_root, "onex_change_control")
    _make_clone(clone_root, "omnibase_infra")
    gateway = _Gateway()  # answers: no merged PR on any ticket-named branch
    linear = _RecordingLinear()
    handler, verified = _handler(linear, _Gh(), gateway)

    result = await handler.handle(_request())

    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.SKIPPED_NO_OFFER_COMPANION
    assert outcome.ticket_id == _TICKET
    assert outcome.enumeration_arm is EnumEvidenceAutocloseArm.OFFER
    assert result.tickets_flipped == 0
    assert verified == []
    assert linear.reads == []
    assert [call[2] for call in gateway.calls] == ["pr-heads"], (
        "the gateway was asked once and found nothing"
    )


async def test_a_commit_that_only_mentions_the_ticket_is_not_its_companion(
    clone_root: Path,
) -> None:
    """``git log --grep`` is a substring match: ``OMN-20113`` matches a commit
    that binds ``OMN-20114`` and names ``OMN-20113`` in passing, and matches
    ``OMN-201130``. The same title binding the window arms use discards both."""
    occ = _make_clone(clone_root, "onex_change_control")
    _land(occ, f"evidence(OMN-20114): follow-up to {_TICKET} (#12456)", ["a.txt"])
    _land(occ, "evidence(OMN-201130): another ticket (#12457)", ["b.txt"])
    handler, verified = _handler(_RecordingLinear(), _Gh())

    result = await handler.handle(_request())

    assert [o.ticket_id for o in result.outcomes] == [_TICKET]
    assert (
        result.outcomes[0].decision
        is EnumEvidenceAutocloseDecision.SKIPPED_NO_OFFER_COMPANION
    )
    assert verified == []


def test_a_commit_with_no_pr_number_is_not_a_companion(clone_root: Path) -> None:
    occ = _make_clone(clone_root, "onex_change_control")
    _land(occ, f"evidence({_TICKET}): direct push", ["a.txt"])

    found, answered = sources.read_clone_candidates(
        occ, _OCC_REPO, _TICKET, sources.run_git_real
    )

    assert answered is True
    assert found == []


def test_a_missing_clone_and_a_missing_watcher_are_unanswered_not_empty(
    tmp_path: Path,
) -> None:
    assert sources.read_clone_candidates(
        None, _OCC_REPO, _TICKET, sources.run_git_real
    ) == ([], False)
    assert sources.read_watcher_candidates(
        tmp_path / "absent.json", _OCC_REPO, _TICKET
    ) == ([], False)


async def test_product_contract_is_read_before_an_unavailable_gateway(
    clone_root: Path,
) -> None:
    _make_clone(clone_root, "onex_change_control")
    product = _make_clone(clone_root, "omnibase_infra")
    _land(
        product, "chore: product evidence (#77)", [f"contracts/{_PRODUCT_TICKET}.yaml"]
    )
    gateway = _Gateway(error="bus: no broker")
    handler, verified = _handler(_RecordingLinear(), _Gh(), gateway)
    result = await _two_ticks(handler, _request(offer_tickets=(_PRODUCT_TICKET,)))
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert verified == [_PRODUCT_TICKET, _PRODUCT_TICKET]
    assert gateway.calls == []
