# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16106: the forward window reads a cut-over product repository's merged PRs.

The forward window enumerated merged change-control PRs only. A repository that
has cut over from change control keeps the ticket's contract in its own
``contracts/<ticket>.yaml`` and mints no companion, so a ticket whose work and
evidence merged there was never asked: not gapped, not refused, absent.

``product_repos`` names the repositories whose own merged PRs the forward window
also reads. A product PR binds through its changed files only, the same
``_extract_ticket_binding`` the change-control arm uses, and its files are read
from its own repository.
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_omn_17342_backfill_arm import (
    _FakeLinear,
    _iso,
    _verdict,
)

pytestmark = pytest.mark.unit

_OCC_REPO = "OmniNode-ai/onex_change_control"
_PRODUCT_REPO = "OmniNode-ai/omnimarket"


def _product_pr(
    number: int, title: str, hours_ago: float, *, bot: bool = False
) -> dict[str, object]:
    stamp = _iso(hours_ago)
    return {
        "number": number,
        "html_url": f"https://github.com/{_PRODUCT_REPO}/pull/{number}",
        "title": title,
        "updated_at": stamp,
        "merged_at": stamp,
        "user": {"login": "bot" if bot else "dev", "type": "Bot" if bot else "User"},
    }


class _Gh:
    """gh double serving one merged-PR page and file list per repository."""

    def __init__(
        self,
        pulls: dict[str, list[dict[str, object]]],
        files: dict[tuple[str, int], list[str]],
        failing: frozenset[str] = frozenset(),
    ) -> None:
        self.pulls = pulls
        self.files = files
        self.failing = failing
        self.paths: list[str] = []
        self.dod_verify: list[str] = []

    async def gh(self, args: list[str], timeout: float):
        path = args[2]
        self.paths.append(path)
        repo = "/".join(path.split("/")[1:3])
        if repo in self.failing:
            return None, f"HTTP 403 reading {repo}"
        if "/files" in path:
            number = int(path.rsplit("/pulls/", 1)[1].split("/", 1)[0])
            return [{"filename": name} for name in self.files[(repo, number)]], ""
        page = int(path.rsplit("page=", 1)[1])
        return (list(self.pulls.get(repo, [])), "") if page == 1 else ([], "")

    async def verify(self, ticket_id: str, cwd: str, timeout: float):
        self.dod_verify.append(ticket_id)
        return _verdict(behavior_proving_count=1, ticket=ticket_id), 0, ""

    def handler(self, linear: _FakeLinear) -> HandlerEvidenceAutocloseSweep:
        return HandlerEvidenceAutocloseSweep(
            linear_client=cast("Any", linear),
            autoclose_disabled=False,
            run_gh_command=self.gh,
            run_dod_verify_command=self.verify,
        )


def _request(**overrides: object) -> ModelEvidenceAutocloseSweepRequest:
    defaults: dict[str, object] = {
        "correlation_id": uuid4(),
        "occ_repo": _OCC_REPO,
        "lookback_hours": 6,
        "backfill_lookback_hours": 0,
        "apply": False,
    }
    defaults.update(overrides)
    return ModelEvidenceAutocloseSweepRequest(**defaults)


def _merged_with_contract() -> _Gh:
    return _Gh(
        pulls={
            _OCC_REPO: [],
            _PRODUCT_REPO: [
                _product_pr(3993, "feat(OMN-21146): run dod verification", 1)
            ],
        },
        files={
            (_PRODUCT_REPO, 3993): [
                "contracts/OMN-21146.yaml",
                "src/omnimarket/nodes/node_dod_verify/contract.yaml",
            ]
        },
    )


async def test_a_product_pr_carrying_its_ticket_contract_reaches_the_verifier() -> None:
    gh = _merged_with_contract()
    linear = _FakeLinear()

    result = await gh.handler(linear).handle(_request(product_repos=(_PRODUCT_REPO,)))

    assert gh.dod_verify == ["OMN-21146"]
    outcome = next(o for o in result.outcomes if o.ticket_id == "OMN-21146")
    assert outcome.companion_pr_url == f"https://github.com/{_PRODUCT_REPO}/pull/3993"
    assert outcome.enumeration_arm == "forward"
    assert any(
        p.startswith(f"repos/{_PRODUCT_REPO}/pulls/3993/files") for p in gh.paths
    )
    assert not any(
        p.startswith(f"repos/{_OCC_REPO}/pulls/3993/files") for p in gh.paths
    )
    assert result.companions_scanned == 1


async def test_without_product_repos_the_window_reads_change_control_only() -> None:
    """Control: the same merged product PR is absent when no repository is named."""
    gh = _merged_with_contract()
    linear = _FakeLinear()

    result = await gh.handler(linear).handle(_request())

    assert gh.dod_verify == []
    assert result.outcomes == ()
    assert not any(_PRODUCT_REPO in path for path in gh.paths)


async def test_a_product_pr_with_no_contract_file_is_never_verified() -> None:
    gh = _Gh(
        pulls={
            _OCC_REPO: [],
            _PRODUCT_REPO: [_product_pr(4000, "fix(OMN-21000): a fix", 1)],
        },
        files={(_PRODUCT_REPO, 4000): ["src/omnimarket/x.py"]},
    )
    linear = _FakeLinear()

    result = await gh.handler(linear).handle(_request(product_repos=(_PRODUCT_REPO,)))

    assert gh.dod_verify == []
    assert [o.decision for o in result.outcomes] == [
        EnumEvidenceAutocloseDecision.SKIPPED_NO_BINDING
    ]


async def test_a_product_pr_carrying_two_contracts_is_refused() -> None:
    gh = _Gh(
        pulls={
            _OCC_REPO: [],
            _PRODUCT_REPO: [_product_pr(4001, "fix(OMN-21001): two", 1)],
        },
        files={
            (_PRODUCT_REPO, 4001): [
                "contracts/OMN-21001.yaml",
                "contracts/OMN-21002.yaml",
            ]
        },
    )
    linear = _FakeLinear()

    result = await gh.handler(linear).handle(_request(product_repos=(_PRODUCT_REPO,)))

    assert gh.dod_verify == []
    assert [o.decision for o in result.outcomes] == [
        EnumEvidenceAutocloseDecision.SKIPPED_AMBIGUOUS_BINDING
    ]


async def test_a_recurring_bot_product_pr_is_dropped_before_any_read() -> None:
    gh = _Gh(
        pulls={
            _OCC_REPO: [],
            _PRODUCT_REPO: [
                _product_pr(
                    4002,
                    "chore(pins): advance omnimarket contract pin to 1.2.3",
                    1,
                    bot=True,
                )
            ],
        },
        files={(_PRODUCT_REPO, 4002): ["contracts/OMN-21003.yaml"]},
    )
    linear = _FakeLinear()

    result = await gh.handler(linear).handle(_request(product_repos=(_PRODUCT_REPO,)))

    assert gh.dod_verify == []
    assert result.outcomes == ()
    assert not any("/pulls/4002/files" in p for p in gh.paths)


async def test_an_unreadable_product_repo_is_an_error_and_change_control_still_runs() -> (
    None
):
    gh = _Gh(
        pulls={
            _OCC_REPO: [
                {
                    "number": 8300,
                    "html_url": f"https://github.com/{_OCC_REPO}/pull/8300",
                    "title": "evidence(OMN-17872): OCC companion",
                    "updated_at": _iso(1),
                    "merged_at": _iso(1),
                }
            ],
        },
        files={(_OCC_REPO, 8300): ["contracts/OMN-17872.yaml"]},
        failing=frozenset({_PRODUCT_REPO}),
    )
    linear = _FakeLinear()

    result = await gh.handler(linear).handle(_request(product_repos=(_PRODUCT_REPO,)))

    assert gh.dod_verify == ["OMN-17872"]
    errors = [
        o
        for o in result.outcomes
        if o.decision is EnumEvidenceAutocloseDecision.ERROR_GITHUB_API
    ]
    assert len(errors) == 1
    assert _PRODUCT_REPO in errors[0].reason
