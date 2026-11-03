# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21928: the dev lane may build a declared open pull request's branch, for its own repository only.

``first_party_sources.yaml`` names a build context for it (``open_pr_context``) and
the lanes that may use it (``open_pr_lanes``). The context accepts the TARGET package
(the one installed from the build context) only when a declaration names the target
repository and the exact commit that was staged. A vendored sibling needs a dev-merge
tag in every context. The stability, judge, collaborator and prod paths are not
in ``open_pr_lanes`` and keep refusing a branch build. The image records the target as
a branch build naming the pull request and its head, never as a dev merge.

The throwaway environments and the policy-editing helpers are the ones of
``test_first_party_slot_context_omn21928.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_SHARED_PATH = Path(__file__).with_name("test_first_party_slot_context_omn21928.py")
_spec = importlib.util.spec_from_file_location(
    "first_party_slot_context_fixtures_omn21928", _SHARED_PATH
)
assert _spec and _spec.loader
shared = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = shared
_spec.loader.exec_module(shared)

fps = shared.fps
CONTRACT = shared.CONTRACT
DEV = shared.DEV
COMMITS = shared.COMMITS
DISTRIBUTIONS = shared.DISTRIBUTIONS
OPEN_PR = CONTRACT.open_pr_context
PR_HEAD = COMMITS["omnibase_infra"]
DECLARED = f"omnibase_infra#5564@{PR_HEAD}"
#: Every lane but dev: stability, judge, collaborator and prod keep refusing.
REFUSING_LANES = ("stability", "judge", "collaborator", "prod")


def _dev_lane_target(branch: str = "HEAD") -> dict[str, Any]:
    """What the dev lane's workspace staging records for the infra clone at the PR head."""
    return {
        "vcs_ref": PR_HEAD,
        "vcs_dirty": False,
        "vcs_branch": branch,
        "dev_merge_tag": "",
    }


def _audit(
    tmp_path: Path, provenance: dict[str, Any], context: str, declared: str
) -> dict[str, Any]:
    sources = fps.audit_environment(
        CONTRACT,
        provenance,
        search_paths=[str(shared._site(tmp_path))],
        required=tuple(DISTRIBUTIONS.values()),
        context=context,
        declared_pr=fps.parse_declared_pr(declared),
    )
    return {source.repo: source for source in sources}


@pytest.mark.parametrize(("branch", "kind"), [("HEAD", "sha"), ("omn-18087", "branch")])
def test_dev_lane_accepts_declared_open_pr_branch_for_its_own_repository(
    tmp_path: Path, branch: str, kind: str
) -> None:
    """The dev lane's declared open PR builds; the image says branch build, PR and head."""
    context, declared = fps.build_context_for_lane("dev", DECLARED)
    assert (context, declared) == (OPEN_PR, DECLARED)
    provenance = shared._provenance(omnibase_infra=_dev_lane_target(branch))
    found = _audit(tmp_path, provenance, context, declared)
    target = found["omnibase_infra"]
    assert target.kind == kind  # a branch build, never a dev merge
    assert target.ok
    assert target.accepted_in == OPEN_PR
    assert target.declared_pr == DECLARED
    assert target.tag == ""
    assert fps.violations(found.values()) == []
    assert f"[declared open pull request {DECLARED}]" in fps.describe(target)


def test_dev_lane_open_pr_provenance_names_the_pr_and_head(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    staging = tmp_path / "sibling-vcs-provenance.json"
    staging.write_text(
        json.dumps(shared._provenance(omnibase_infra=_dev_lane_target())),
        encoding="utf-8",
    )
    manifest = tmp_path / "build-provenance.json"
    manifest.write_text("{}", encoding="utf-8")
    exit_code = fps.run_audit(
        staging,
        require=tuple(DISTRIBUTIONS.values()),
        search_paths=[str(shared._site(tmp_path))],
        record_into=manifest,
        context=OPEN_PR,
        declared_pr=DECLARED,
    )
    assert exit_code == 0
    assert f"[declared open pull request {DECLARED}]" in capsys.readouterr().out
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    assert recorded["first_party_build_context"] == OPEN_PR
    assert recorded["first_party_declared_pr"] == DECLARED
    (infra,) = [
        row
        for row in recorded["first_party_sources"]
        if row["distribution"] == "omnibase-infra"
    ]
    assert infra["kind"] == "sha"
    assert infra["tag"] == ""
    assert infra["commit"] == PR_HEAD
    assert infra["declared_pr"] == DECLARED


def test_open_pr_context_negative_control_without_a_declaration_the_branch_fails(
    tmp_path: Path,
) -> None:
    provenance = shared._provenance(omnibase_infra=_dev_lane_target("omn-18087"))
    for context in (DEV, OPEN_PR):
        found = _audit(tmp_path / context, provenance, context, "")
        assert not found["omnibase_infra"].ok, context
        assert found["omnibase_infra"].accepted_in == ""
    assert "needs a declared open pull request" in found["omnibase_infra"].detail
    # The dev lane's default context refuses it even with a declaration in hand.
    found = _audit(tmp_path / "dev-declared", provenance, DEV, DECLARED)
    assert not found["omnibase_infra"].ok
    assert found["omnibase_infra"].declared_pr == ""


def test_open_pr_declaration_must_name_the_staged_head_and_repository(
    tmp_path: Path,
) -> None:
    provenance = shared._provenance(omnibase_infra=_dev_lane_target("omn-18087"))
    wrong = (
        f"omnibase_infra#5564@{'f' * 40}",  # a different commit than staged
        f"omnibase_core#5564@{PR_HEAD}",  # a different repository
    )
    for index, declared in enumerate(wrong):
        found = _audit(tmp_path / str(index), provenance, OPEN_PR, declared)
        assert not found["omnibase_infra"].ok, declared
        assert "does not name omnibase_infra" in found["omnibase_infra"].detail


def test_sibling_branch_still_refused_on_the_dev_lane_with_a_declared_open_pr(
    tmp_path: Path,
) -> None:
    for repo in ("omnibase_core", "omnibase_spi", "omnimarket"):
        from_branch = {
            "vcs_ref": COMMITS[repo],
            "vcs_dirty": False,
            "vcs_branch": "dev",
            "dev_merge_tag": "",
        }
        provenance = shared._provenance(
            **{repo: from_branch, "omnibase_infra": _dev_lane_target()}
        )
        found = _audit(tmp_path / repo, provenance, OPEN_PR, DECLARED)
        assert found["omnibase_infra"].ok, repo
        assert not found[repo].ok, repo
        assert found[repo].accepted_in == ""
        # A declaration naming the sibling at its own commit accepts nothing either.
        sibling_declared = f"{repo}#1@{COMMITS[repo]}"
        found = _audit(
            tmp_path / f"{repo}-named", provenance, OPEN_PR, sibling_declared
        )
        assert not found[repo].ok, repo
        assert not found["omnibase_infra"].ok, repo


@pytest.mark.parametrize("lane", REFUSING_LANES)
def test_stability_judge_collaborator_and_prod_paths_refuse_a_branch_build(
    lane: str,
) -> None:
    with pytest.raises(ValueError, match="cannot build a declared open pull request"):
        fps.build_context_for_lane(lane, DECLARED)
    # With no declaration the lane audits in the default context, which accepts no branch.
    assert fps.build_context_for_lane(lane) == (DEV, "")
    assert CONTRACT.context(DEV).target_accepts == ()


def test_the_lanes_that_may_build_an_open_pr_come_from_the_data(tmp_path: Path) -> None:
    assert CONTRACT.open_pr_lanes == ("dev",)

    widened = shared._write_policy(
        tmp_path / "widened",
        lambda data: data.update(open_pr_lanes=["dev", "stability"]),
    )
    assert fps.build_context_for_lane("stability", DECLARED, contract_path=widened) == (
        OPEN_PR,
        DECLARED,
    )

    closed = shared._write_policy(
        tmp_path / "closed", lambda data: data.update(open_pr_lanes=[])
    )
    with pytest.raises(ValueError, match="cannot build a declared open pull request"):
        fps.build_context_for_lane("dev", DECLARED, contract_path=closed)


def test_the_open_pr_context_must_require_a_declaration(tmp_path: Path) -> None:
    path = shared._write_policy(
        tmp_path,
        lambda data: shared._context(data, "dev-lane-open-pr").pop(
            "requires_declared_pr"
        ),
    )
    with pytest.raises(ValueError, match="requires_declared_pr"):
        fps.load_contract(path)


@pytest.mark.parametrize(
    "text", ["omnibase_infra#0@" + "a" * 40, "omnibase_infra#5@abc", "5564@" + "a" * 40]
)
def test_a_malformed_declaration_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="is not <repo>#<number>@<40-hex head sha>"):
        fps.parse_declared_pr(text)
    assert fps.parse_declared_pr("") is None


def test_the_image_audit_passes_the_declaration_build_arg() -> None:
    dockerfile = (shared.REPO_ROOT / "docker/Dockerfile.runtime").read_text(
        encoding="utf-8"
    )
    assert 'ARG FIRST_PARTY_DECLARED_PR=""' in dockerfile
    assert 'declared_pr=os.environ.get("FIRST_PARTY_DECLARED_PR", "")' in dockerfile
