# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The pre-push ticket-contract leg refuses what repo-evidence / dod-verify refuses (OMN-20870).

The hosted ``repo-evidence / dod-verify`` context refuses a pull request whose
cited ticket has no ``contracts/<ticket>.yaml`` at the head, and its verifier
refuses a prose check value. ``scripts/ci/check_deploy_scope_dod.py`` (the
existing ``prepush-deploy-scope-dod`` hook) now refuses both before the push.

The presence tests drive a real git repository, because the leg reads the
pushed HEAD commit and not the working tree. The prerequisite tests call the released
verifier at the version the repo-evidence caller pins, the same
``contract_prerequisite`` onex scm pr-create runs (OMN-21642).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from packaging.version import Version

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
    strip_git_location_env,
)
from scripts.ci.check_deploy_scope_dod import (
    FAIL_CONTRACT_REFUSED,
    FAIL_NO_CONTRACT,
    NOTICE_NO_TICKET_CITED,
    PASS_CONTRACTS,
    PREREQUISITE_VERIFIER_VERSION,
    cited_tickets,
    classify_ticket_contracts,
    pinned_verifier_judge,
    read_head_contract,
    resolve_branch_subjects,
    verifier_prerequisite_argv,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
_CALLER = REPO_ROOT / ".github" / "workflows" / "call-repo-evidence-gate.yml"

_PROSE = "deploy: restart the runtime on the lab and confirm it is healthy"
_COMMAND = "uv run pytest tests/ci/test_ticket_contract_prepush_omn20870.py -q"


@pytest.fixture(autouse=True)
def _no_inherited_git_location() -> Iterator[None]:
    """The leg's own git calls must reach tmp_path, not the invoking worktree."""
    saved = dict(os.environ)
    strip_git_location_env(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "base")
    _git(repo, "config", "user.email", "lane@example.invalid")
    _git(repo, "config", "user.name", "lane")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "chore: base")
    return repo


def _contract(check_value: str) -> str:
    return yaml.safe_dump(
        {
            "ticket_id": "OMN-99001",
            "dod_evidence": [
                {
                    "id": "dod-ac2",
                    "checks": [
                        {"check_type": "test_passes", "check_value": check_value}
                    ],
                }
            ],
        }
    )


def _never_judge(contracts: Mapping[str, str]) -> dict[str, str | None]:
    raise AssertionError(f"the prerequisite must not run here: {sorted(contracts)}")


def _accept_all(contracts: Mapping[str, str]) -> dict[str, str | None]:
    return dict.fromkeys(contracts)


@pytest.fixture(scope="module")
def released_judge(tmp_path_factory: pytest.TempPathFactory):
    """The released verifier's guard, at the pin the pre-push leg uses."""
    return pinned_verifier_judge(
        PREREQUISITE_VERIFIER_VERSION, tmp_path_factory.mktemp("judge-cwd")
    )


def test_a_cited_ticket_with_no_head_contract_is_refused_and_the_file_named(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "lane")
    (repo / "change.txt").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "change.txt")
    _git(repo, "commit", "-q", "-m", "feat(OMN-99001): a change with no contract")

    tickets = cited_tickets(resolve_branch_subjects(repo, "base"))
    decision = classify_ticket_contracts(
        tickets, lambda t: read_head_contract(repo, t), _never_judge
    )

    assert tickets == ["OMN-99001"]
    assert decision.outcome == FAIL_NO_CONTRACT
    assert decision.exit_code == 1
    assert decision.findings == (
        "this push cites OMN-99001 but carries no contracts/OMN-99001.yaml",
    )


@pytest.mark.live_contact("tests/ci/fixtures/ticket_contract_refusal_omn20870.json")
def test_the_recorded_hosted_refusal_is_refused_locally_naming_the_same_file(
    tmp_path: Path, recorded_response: dict[str, object]
) -> None:
    """omnibase_infra#4772 went red on repo-evidence / dod-verify; this push is refused first."""
    response = cast("dict[str, Any]", recorded_response["response"])
    hosted_error = str(response["error_line"])
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "lane")
    (repo / "census.json").write_text("{}\n", encoding="utf-8")
    _git(repo, "add", "census.json")
    _git(repo, "commit", "-q", "-m", str(response["pr_title"]))

    tickets = cited_tickets(resolve_branch_subjects(repo, "base"))
    decision = classify_ticket_contracts(
        tickets, lambda t: read_head_contract(repo, t), _never_judge
    )

    assert hosted_error.endswith(
        "pull request cites OMN-18606 but carries no contracts/OMN-18606.yaml"
    )
    assert tickets == ["OMN-18606"]
    assert decision.outcome == FAIL_NO_CONTRACT
    assert decision.findings == (
        "this push cites OMN-18606 but carries no contracts/OMN-18606.yaml",
    )


def test_an_uncommitted_contract_does_not_satisfy_the_pushed_head(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    (repo / "contracts").mkdir()
    (repo / "contracts" / "OMN-99001.yaml").write_text(
        _contract(_COMMAND), encoding="utf-8"
    )

    decision = classify_ticket_contracts(
        ["OMN-99001"], lambda t: read_head_contract(repo, t), _never_judge
    )

    assert decision.outcome == FAIL_NO_CONTRACT


def test_a_committed_contract_is_read_from_head(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (repo / "contracts").mkdir()
    (repo / "contracts" / "OMN-99001.yaml").write_text(
        _contract(_COMMAND), encoding="utf-8"
    )
    _git(repo, "add", "contracts")
    _git(repo, "commit", "-q", "-m", "feat(OMN-99001): carry the contract")

    decision = classify_ticket_contracts(
        ["OMN-99001"], lambda t: read_head_contract(repo, t), _accept_all
    )

    assert decision.outcome == PASS_CONTRACTS
    assert decision.exit_code == 0


def test_subjects_merged_in_from_the_base_branch_are_not_cited(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    _git(repo, "checkout", "-q", "-b", "lane")
    (repo / "lane.txt").write_text("lane\n", encoding="utf-8")
    _git(repo, "add", "lane.txt")
    _git(repo, "commit", "-q", "-m", "feat(OMN-99001): the lane's change")
    _git(repo, "checkout", "-q", "base")
    (repo / "other.txt").write_text("other\n", encoding="utf-8")
    _git(repo, "add", "other.txt")
    _git(repo, "commit", "-q", "-m", "fix(OMN-99002): someone else's change")
    _git(repo, "checkout", "-q", "lane")
    _git(repo, "merge", "-q", "--no-edit", "--no-ff", "base")
    _git(repo, "branch", "-q", "-f", "base", "base~1")

    assert cited_tickets(resolve_branch_subjects(repo, "base")) == ["OMN-99001"]


def test_no_cited_ticket_is_a_notice_not_a_refusal() -> None:
    decision = classify_ticket_contracts([], _never_read, _never_judge)

    assert decision.outcome == NOTICE_NO_TICKET_CITED
    assert decision.exit_code == 0


def _never_read(ticket: str) -> str | None:
    raise AssertionError(f"no contract should be read: {ticket}")


#: The first omnimarket release carrying ``contract_prerequisite`` (OMN-21642), and
#: the first whose node_dod_verify reads the live Linear ticket (OMN-20858).
_PREREQUISITE_AND_LINEAR_READ_RELEASE = Version("0.4.312")


def _caller_verifier_version() -> Version:
    caller = yaml.safe_load(_CALLER.read_text(encoding="utf-8"))
    return Version(str(caller["jobs"]["repo-evidence"]["with"]["verifier-version"]))


def test_the_prerequisite_is_the_released_verifier_at_its_own_pin() -> None:
    argv = verifier_prerequisite_argv(PREREQUISITE_VERIFIER_VERSION)

    assert f"omnimarket=={PREREQUISITE_VERIFIER_VERSION}" in argv
    assert Version(PREREQUISITE_VERIFIER_VERSION) >= (
        _PREREQUISITE_AND_LINEAR_READ_RELEASE
    )
    assert any(
        "from omnimarket.nodes.node_dod_verify.services.contract_prerequisite "
        "import contract_prerequisite" in arg
        for arg in argv
    )


def test_a_prose_check_value_is_refused_and_the_check_named(
    tmp_path: Path, released_judge
) -> None:
    decision = classify_ticket_contracts(
        ["OMN-99001"], lambda _t: _contract(_PROSE), released_judge
    )

    assert decision.outcome == FAIL_CONTRACT_REFUSED
    assert decision.exit_code == 1
    assert len(decision.findings) == 1
    assert decision.findings[0].startswith(
        "contracts/OMN-99001.yaml: INVALID_CHECK_VALUE_NOT_A_COMMAND: "
    )


def test_a_command_shaped_contract_passes_the_released_guard(released_judge) -> None:
    decision = classify_ticket_contracts(
        ["OMN-99001"], lambda _t: _contract(_COMMAND), released_judge
    )

    assert decision.outcome == PASS_CONTRACTS
    assert decision.findings == ()


def test_the_hosted_caller_pins_a_verifier_that_reads_no_linear_ticket() -> None:
    """OMN-20413: the caller passes no secrets, so a verifier that reads the live
    Linear ticket gives every contract with a requirements acceptance id
    status=unresolved ticket_unavailable, whatever the pull request changed."""
    assert _caller_verifier_version() < _PREREQUISITE_AND_LINEAR_READ_RELEASE
