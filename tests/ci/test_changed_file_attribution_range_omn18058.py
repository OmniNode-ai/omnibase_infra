# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Changed-file ATTRIBUTION ranges are merge-base anchored, never two-dot (OMN-18058).

A gate that asks "what did THIS branch change?" must diff against the merge base
with ``origin/dev``. The two-dot form ``git diff A B`` describes the difference
between two *trees*, so on a branch whose base is stale it additionally reports
every file a PEER landed on ``dev`` since the branch point -- presented as this
branch's own work. For a deploy-scope gate that inverts the message: the staler
your base, the more of somebody else's deploy surface you are asked to write DoD
evidence for, and the refusal names a ticket you never cited.

Every test here carries its own POSITIVE CONTROL: the same fixture, read through
the two-dot form, DOES surface the peer's file. Without that control a green
assertion is indistinguishable from a fixture that never reproduced the class.

Sites pinned by this module:

* ``scripts/ci/check_deploy_scope_dod.py`` -- ``compute_changed_files``
  (the OMN-14681 pre-push deploy-scope DoD parity gate).
* ``scripts/hooks/prepush_smart_tests.sh`` -- the OMN-13973 governed selector's
  ``BASE_SHA`` range (structural ratchet; the shell hook is not importable).
* ``scripts/validate-kafka-schema-handshake.py`` -- ``_changed_module_prefixes``.
* ``scripts/check_release_identity.py`` -- ``_collect_changed_files``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from scripts.ci.check_deploy_scope_dod import (
    FAIL_NO_EVIDENCE,
    FAIL_NO_TICKET,
    DeployScopeHookError,
    classify_deploy_scope,
    compute_changed_files,
    load_canonical_validator,
    resolve_omni_home,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Matches the canonical ``src/omnibase_infra/runtime/**/*.py`` deploy-scope
#: pattern, so a peer landing this file is deploy-scoped by construction.
_PEER_DEPLOY_SCOPED = "src/omnibase_infra/runtime/peer_landed_by_someone_else.py"
#: Matches no runtime pattern -- the pushing branch's own, innocuous change.
_BRANCH_UNRELATED = "docs/branch_note.md"
#: Deploy-scoped, but authored by the pushing branch itself.
_BRANCH_DEPLOY_SCOPED = "src/omnibase_infra/runtime/branch_owned_surface.py"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _write(repo: Path, rel: str, text: str) -> None:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _build_stale_base_repo(
    tmp_path: Path,
    *,
    default_branch: str = "dev",
    branch_files: tuple[str, ...] = (_BRANCH_UNRELATED,),
    branch_commits: bool = True,
) -> Path:
    """A branch cut from ``<default_branch>``, after which a PEER advanced it.

    Layout after this returns (``work`` is checked out on ``feature``)::

        o base            <- merge base / branch point
        |\\
        | o feature: branch_files
        o peer: src/omnibase_infra/runtime/peer_landed_by_someone_else.py
                          <- origin/<default_branch>

    So the branch is BEHIND the base ref by exactly one peer commit that touched
    a deploy-scoped path -- the measured OMN-18058 shape.
    """
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    _git(work, "config", "user.email", "omn18058@example.invalid")
    _git(work, "config", "user.name", "omn18058 fixture")
    _git(work, "config", "commit.gpgsign", "false")
    _git(work, "checkout", "-q", "-b", default_branch)

    _write(work, "docs/base.md", "base\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "base")
    _git(work, "remote", "add", "origin", str(origin))
    _git(work, "push", "-q", "origin", default_branch)

    _git(work, "checkout", "-q", "-b", "feature")
    if branch_commits:
        for rel in branch_files:
            _write(work, rel, "branch content\n")
        _git(work, "add", "-A")
        _git(
            work, "commit", "-q", "-m", "feat(OMN-18058): work this branch actually did"
        )

    _git(work, "checkout", "-q", default_branch)
    _write(work, _PEER_DEPLOY_SCOPED, "peer content\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "fix(OMN-99999): a peer's deploy-scoped landing")
    _git(work, "push", "-q", "origin", default_branch)

    _git(work, "checkout", "-q", "feature")
    _git(work, "fetch", "-q", "origin", default_branch)
    return work


def _two_dot(repo: Path, base_ref: str) -> list[str]:
    """The buggy form, run on the SAME fixture -- the positive control."""
    out = _git(repo, "diff", "--name-only", base_ref, "HEAD")
    return [line for line in out.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def validator() -> ModuleType:
    try:
        return load_canonical_validator(resolve_omni_home(REPO_ROOT))
    except DeployScopeHookError as exc:
        pytest.skip(f"canonical omniclaude validator unusable here: {exc}")


@pytest.fixture(autouse=True)
def _report_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPLOY_GATE_FALSIFIABILITY", "report")


@pytest.mark.unit
def test_positive_control_the_two_dot_form_does_attribute_the_peers_file(
    tmp_path: Path,
) -> None:
    """The fixture really does reproduce the attribution class.

    This is the control that makes every green assertion below meaningful: read
    through the two-dot form, the peer's deploy-scoped file IS reported as part
    of this branch's changes.
    """
    repo = _build_stale_base_repo(tmp_path)
    assert _PEER_DEPLOY_SCOPED in _two_dot(repo, "origin/dev")


@pytest.mark.unit
def test_deploy_scope_gate_does_not_attribute_a_peers_deploy_scoped_file(
    tmp_path: Path,
) -> None:
    """OMN-18058: the pre-push deploy-scope gate sees only the branch's own work."""
    repo = _build_stale_base_repo(tmp_path)
    changed = compute_changed_files(repo, "origin/dev")
    assert changed == [_BRANCH_UNRELATED]
    assert _PEER_DEPLOY_SCOPED not in changed


@pytest.mark.unit
def test_deploy_scope_gate_passes_when_only_a_peer_touched_deploy_scope(
    tmp_path: Path, validator: ModuleType
) -> None:
    """End-to-end: no refusal is raised on a peer's deploy-scoped landing."""
    repo = _build_stale_base_repo(tmp_path)
    decision = classify_deploy_scope(
        validator,
        compute_changed_files(repo, "origin/dev"),
        pr_body="a body that cites no ticket at all",
        contracts_dir=repo / "no-such-contracts",
    )
    assert decision.exit_code == 0
    assert decision.runtime_hits == ()


@pytest.mark.unit
def test_positive_control_the_gate_still_refuses_the_branchs_own_deploy_scope(
    tmp_path: Path, validator: ModuleType
) -> None:
    """The narrowing does NOT disarm the gate.

    Same stale-base fixture, but the branch itself authors a deploy-scoped file
    and cites no ticket: the refusal must still fire, and must name the branch's
    file rather than the peer's.
    """
    repo = _build_stale_base_repo(
        tmp_path, branch_files=(_BRANCH_UNRELATED, _BRANCH_DEPLOY_SCOPED)
    )
    changed = compute_changed_files(repo, "origin/dev")
    assert _BRANCH_DEPLOY_SCOPED in changed
    assert _PEER_DEPLOY_SCOPED not in changed

    decision = classify_deploy_scope(
        validator,
        changed,
        pr_body="a body that cites no ticket at all",
        contracts_dir=repo / "no-such-contracts",
    )
    assert decision.outcome in (FAIL_NO_TICKET, FAIL_NO_EVIDENCE)
    assert decision.exit_code == 1
    assert decision.runtime_hits == (_BRANCH_DEPLOY_SCOPED,)


@pytest.mark.unit
def test_kafka_schema_handshake_range_is_merge_base_anchored() -> None:
    """Structural ratchet over ``--changed-only`` narrowing.

    ``scripts/validate-kafka-schema-handshake.py`` cannot be imported here: its
    module body ``sys.exit(2)``s when the omniintelligence/omnimemory siblings are
    absent, which is the normal state of this repo's own venv. Stated limit, not a
    silent one -- the range shape is pinned by source instead of by execution, and
    the class it excludes is proven live by the positive control above.
    """
    source = (REPO_ROOT / "scripts/validate-kafka-schema-handshake.py").read_text(
        encoding="utf-8"
    )
    assert '["git", "merge-base", "origin/main", "HEAD"]' in source
    assert '["git", "diff", "--name-only", merge_base]' in source
    forbidden = '["git", "diff", "--name-only", "origin/main"]'
    assert forbidden not in source
    # Positive control: the forbidden literal is exactly what the assertion reads.
    assert forbidden in source.replace("merge_base]", '"origin/main"]')


@pytest.mark.unit
def test_release_identity_empty_branch_diff_does_not_fall_back_to_a_peers_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The empty-three-dot fallback must stay merge-base anchored.

    When the branch has no commits of its own, the branch changed nothing. The old
    two-dot fallback reported the PEER's ``src/`` landings instead, which arms the
    release-identity version gate against a branch that touched no packaged source
    at all.
    """
    identity = _load_module(
        "_omn18058_release_identity", REPO_ROOT / "scripts/check_release_identity.py"
    )
    repo = _build_stale_base_repo(tmp_path, branch_commits=False)
    # Positive control: the three-dot set really is empty here, so the fallback
    # branch under test is the one that executes, and the two-dot form on this
    # same fixture DOES surface the peer's file.
    assert _git(repo, "diff", "--name-only", "origin/dev...HEAD").strip() == ""
    assert _PEER_DEPLOY_SCOPED in _two_dot(repo, "origin/dev")

    # ``_git`` runs with ``cwd=_REPO_ROOT``; point it at the fixture so the test
    # exercises the fixture's history rather than this checkout's.
    monkeypatch.setattr(identity, "_REPO_ROOT", repo)
    assert identity._collect_changed_files("origin/dev", []) == ()

    # And the fallback still sees this branch's own UNCOMMITTED edit.
    (repo / "src" / "omnibase_infra").mkdir(parents=True, exist_ok=True)
    (repo / "docs" / "base.md").write_text("locally edited\n", encoding="utf-8")
    assert identity._collect_changed_files("origin/dev", []) == ("docs/base.md",)


@pytest.mark.unit
def test_governed_selector_range_is_merge_base_anchored() -> None:
    """Structural ratchet over the OMN-13973 selector (shell, not importable).

    The selector resolves ``BASE_SHA`` from ``git merge-base`` and diffs against
    that sha. A refactor back to ``git diff --name-only "${BASE_REF}" HEAD``
    would silently re-introduce the OMN-18058 class here; this pins it.
    """
    source = (REPO_ROOT / "scripts/hooks/prepush_smart_tests.sh").read_text(
        encoding="utf-8"
    )
    assert 'BASE_SHA="$(git merge-base "${BASE_REF}" HEAD' in source
    assert 'git diff --name-only "${BASE_SHA}" HEAD' in source
    # Positive control: the forbidden form is what this assertion excludes.
    forbidden = 'git diff --name-only "${BASE_REF}" HEAD'
    assert forbidden not in source
    assert forbidden in source.replace("BASE_SHA", "BASE_REF")


def _load_module(name: str, path: Path) -> ModuleType:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
