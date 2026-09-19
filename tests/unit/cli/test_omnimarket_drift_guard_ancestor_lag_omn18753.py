# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18814: a KNOWN-ANCESTOR lag proceeds and is stamped; everything else
still refuses.

Why this exists, measured rather than asserted. On 2026-09-19 the plugin CLI
venv sat in a refusing state for about 545 of 740 elapsed minutes -- roughly
74% of the day -- across two windows, 8h31m and 34m, each closed only when a
lane hand-ran a repair. omnimarket dev took 159 commits in 7 days, about 22.7
a day, and the guard compares against the canonical clone's checked-out head,
so EVERY commit opens a window. Refusal was the normal state, not the
exception, and the thing being refused is the product.

The distinction this file pins is the one that makes relaxing it safe. An
installed commit that is a strict ANCESTOR of the clone head is a staleness
fact: the bytes are real, reviewed, and merged, they are simply not the tip.
An installed commit standing in no descendant relationship to the head, or one
the clone cannot resolve at all, is a PROVENANCE failure -- nobody can say what
those bytes are -- and it must keep refusing exactly as before.

Both directions are pinned here because both failure modes are silent. Refuse
the ancestor case and delegation stays unusable while the gate looks healthy;
admit the unknown case and the guard keeps reporting green over bytes it cannot
account for.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import MISSING
from datetime import datetime
from pathlib import Path

import pytest

from omnibase_infra.cli import omnimarket_drift_guard as guard

pytestmark = pytest.mark.unit


def _git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` with a hermetic identity and return stdout."""
    env: Mapping[str, str] = {
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "HOME": str(cwd),
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
        env=dict(env),
        timeout=30,
    ).stdout.strip()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An ``$OMNI_HOME`` whose ``omnimarket`` clone has a real history.

    Real git, not a stub: the property under test IS an ancestry question, and
    a stubbed answer would pin the test's own opinion of ancestry rather than
    git's.
    """
    clone = tmp_path / "omnimarket"
    clone.mkdir()
    _git(clone, "init", "-q", "-b", "dev")
    for n in range(4):
        (clone / "f.txt").write_text(f"{n}\n", encoding="utf-8")
        _git(clone, "add", "f.txt")
        _git(clone, "commit", "-q", "-m", f"c{n}")
    return tmp_path


@pytest.fixture
def head(workspace: Path) -> str:
    return _git(workspace / "omnimarket", "rev-parse", "HEAD")


def _ancestor(workspace: Path, back: int) -> str:
    return _git(workspace / "omnimarket", "rev-parse", f"HEAD~{back}")


# ---------------------------------------------------------------------------
# AC1 / AC2 — a known ancestor proceeds, and says so on the receipt
# ---------------------------------------------------------------------------


def test_a_known_ancestor_lag_proceeds_instead_of_refusing(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, head: str
) -> None:
    """AC1. The bytes are merged and reviewed; they are simply not the tip."""
    monkeypatch.setattr(
        guard, "installed_omnimarket_commit", lambda: _ancestor(workspace, 2)
    )

    stamp = guard.check_omnimarket_drift(omni_home=str(workspace))

    assert stamp is not None, (
        "an ancestor lag returned None, which is the 'clone comparison passed "
        "exactly' signal -- the lag has to be reported, not swallowed"
    )
    assert not isinstance(stamp, guard.ModelOffRegistryCheck), (
        "an ancestor lag on a registry machine must not be reported as the "
        "off-registry verdict; that branch is OMN-17255's and means no clone"
    )


def test_the_lag_is_stamped_with_typed_fields_and_no_defaults(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, head: str
) -> None:
    """AC2. A stamp whose fields can be omitted is a stamp that can lie by
    omission: a receipt with no ``commits_behind`` reads to a later auditor
    exactly like a run that was at the tip."""
    installed = _ancestor(workspace, 2)
    monkeypatch.setattr(guard, "installed_omnimarket_commit", lambda: installed)

    stamp = guard.check_omnimarket_drift(omni_home=str(workspace))
    assert isinstance(stamp, guard.ModelOmnimarketLagStamp)

    assert stamp.installed_commit == installed
    assert stamp.clone_head == head
    assert stamp.commits_behind == 2
    assert isinstance(stamp.stamped_at, datetime)
    assert stamp.stamped_at.tzinfo is not None, "a naive timestamp is unanchored"

    fields = guard.ModelOmnimarketLagStamp.__dataclass_fields__
    for name in ("installed_commit", "clone_head", "commits_behind", "stamped_at"):
        assert name in fields, f"{name} is not a declared field of the stamp"
        assert fields[name].default is MISSING, (
            f"{name} carries a default, so a caller that never resolved it "
            f"produces a receipt indistinguishable from a measured one"
        )
        assert fields[name].default_factory is MISSING, (
            f"{name} carries a default_factory, same failure as a default"
        )


def test_the_stamp_reaches_the_receipt_and_names_the_lag(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, head: str
) -> None:
    """AC2. The receipt outlives the terminal; the stderr line does not."""
    installed = _ancestor(workspace, 1)
    monkeypatch.setattr(guard, "installed_omnimarket_commit", lambda: installed)

    stamp = guard.check_omnimarket_drift(omni_home=str(workspace))
    assert isinstance(stamp, guard.ModelOmnimarketLagStamp)

    receipt = stamp.as_receipt_fields()
    assert receipt["installed_commit"] == installed
    assert receipt["clone_head"] == head
    assert receipt["commits_behind"] == 1
    assert isinstance(receipt["stamped_at"], str), (
        "the receipt is serialized to JSON; a datetime that is not rendered "
        "here fails at write time instead of here"
    )
    assert "ancestor-lag" in stamp.line


# ---------------------------------------------------------------------------
# AC3 / AC4 — everything that is not a known ancestor still refuses
# ---------------------------------------------------------------------------


def test_a_non_descendant_commit_still_refuses(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, head: str
) -> None:
    """AC3. A commit on a divergent line is not a staleness fact.

    Built as a real second root so it is genuinely unrelated to the head
    rather than merely older -- the case a version comparison would wave
    through and an ancestry check must not.
    """
    clone = workspace / "omnimarket"
    _git(clone, "checkout", "-q", "--orphan", "sideline")
    (clone / "g.txt").write_text("side\n", encoding="utf-8")
    _git(clone, "add", "g.txt")
    _git(clone, "commit", "-q", "-m", "divergent")
    divergent = _git(clone, "rev-parse", "HEAD")
    _git(clone, "checkout", "-q", "dev")

    monkeypatch.setattr(guard, "installed_omnimarket_commit", lambda: divergent)

    with pytest.raises(guard.OmnimarketDriftError):
        guard.check_omnimarket_drift(omni_home=str(workspace))


def test_an_unknown_commit_still_refuses(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, head: str
) -> None:
    """AC4. A commit the clone cannot resolve is a provenance failure.

    This is the one that must not be waved through on a probe error: git
    exits non-zero both for 'not an ancestor' and for 'no such object', and
    reading either as 'ancestor' would admit bytes nobody can account for.
    """
    monkeypatch.setattr(guard, "installed_omnimarket_commit", lambda: "b" * 40)

    with pytest.raises(guard.OmnimarketDriftError):
        guard.check_omnimarket_drift(omni_home=str(workspace))


def test_an_unusable_ancestry_probe_refuses_rather_than_assuming_ancestor(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, head: str
) -> None:
    """AC4. Fails CLOSED. An ancestry question that could not be asked is not
    an ancestry question answered yes."""
    monkeypatch.setattr(
        guard, "installed_omnimarket_commit", lambda: _ancestor(workspace, 2)
    )

    # Scoped to the ANCESTRY probe only. Breaking every git call instead
    # would make the clone head unresolvable, which routes to the
    # off-registry branch and would prove a different thing entirely.
    real_run = subprocess.run

    def _fail_ancestry(args: object, *rest: object, **kwargs: object) -> object:
        argv = list(args) if isinstance(args, (list, tuple)) else [args]
        if any(a in ("merge-base", "rev-list") for a in argv):
            raise OSError("git unavailable")
        return real_run(args, *rest, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(guard.subprocess, "run", _fail_ancestry)

    with pytest.raises(guard.OmnimarketDriftError):
        guard.check_omnimarket_drift(omni_home=str(workspace))


def test_a_machine_with_no_clone_is_untouched_by_the_ancestor_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """AC6. The off-registry branch is OMN-17255's and reports without
    blocking; the ancestor path must not reach it."""
    monkeypatch.setattr(guard, "installed_omnimarket_commit", lambda: "b" * 40)

    result = guard.check_omnimarket_drift(omni_home=str(tmp_path))

    assert isinstance(result, guard.ModelOffRegistryCheck), (
        "a machine with no canonical clone must still take the off-registry "
        "branch; the ancestor path has no clone to resolve ancestry against"
    )


# ---------------------------------------------------------------------------
# AC5 — the manual override is untouched
# ---------------------------------------------------------------------------


def test_the_manual_override_is_not_consulted_on_the_ancestor_path(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, head: str
) -> None:
    """AC5. This change replaces a hand judgement with a recorded automatic
    one; it does not widen the hand judgement. An ancestor lag proceeds on its
    own merits, so passing the opt-out changes nothing about the verdict."""
    monkeypatch.setattr(
        guard, "installed_omnimarket_commit", lambda: _ancestor(workspace, 1)
    )

    without = guard.check_omnimarket_drift(omni_home=str(workspace))
    with_flag = guard.check_omnimarket_drift(omni_home=str(workspace), allow_drift=True)

    assert isinstance(without, guard.ModelOmnimarketLagStamp)
    assert isinstance(with_flag, guard.ModelOmnimarketLagStamp)
    assert without.installed_commit == with_flag.installed_commit
    assert without.commits_behind == with_flag.commits_behind


def test_the_reconciler_is_not_burned_on_a_lag_that_needs_no_repair(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, head: str
) -> None:
    """An ancestor lag proceeds, so installing packages mid-dispatch to close
    it would spend a minute of a human's latency for a run that was already
    going to succeed. The reconciler that DOES close it is the tick
    (OMN-18815), off the hot path."""
    monkeypatch.setattr(
        guard, "installed_omnimarket_commit", lambda: _ancestor(workspace, 1)
    )
    calls = {"n": 0}

    def _reconcile() -> object:
        calls["n"] += 1
        raise AssertionError("the reconciler must not run for an ancestor lag")

    guard.check_omnimarket_drift(omni_home=str(workspace), reconcile=_reconcile)

    assert calls["n"] == 0
