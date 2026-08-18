# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Hermetic tests for the OMN-16096 mirror-based lineage resolver.

The merge-required ``dep-provenance-lineage-gate`` CI job (``ci.yml``) used to
resolve dep-provenance lineage exclusively through the live GitHub REST API
(``resolve_src_tree_sha``), coupling merge eligibility to github.com egress
weather -- two canary attempts on PR #2758 burned on
``transport error: The read operation timed out``.
``resolve_src_tree_sha_hermetic`` closes that gap: it tries the OMN-16053
host-level git mirror first (``resolve_src_tree_sha_via_mirror``, which shells
out to local ``git`` against ``git://<host>:<port>/<repo>.git``) and falls
back to the live REST resolver only when the mirror itself cannot serve the
ref.

Every test here is hermetic (no network, no real git subprocess): ``git``
subprocess calls are replaced with a fake ``subprocess.run`` so the module's
own decision logic -- success, each failure shape, and the fallback order --
is exercised without ever touching a real mirror or github.com. This mirrors
the sibling hermetic file's philosophy
(``tests/scripts/test_check_dep_provenance_lineage_omn15604.py``: fake
resolver, no network) one level down, at the resolver's own internals.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_dep_provenance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_dep_provenance", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return _load_module()


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    """Minimal stand-in for subprocess.CompletedProcess -- only the three
    attributes resolve_src_tree_sha_via_mirror reads."""
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# resolve_src_tree_sha_via_mirror -- the pure mirror probe.
# ---------------------------------------------------------------------------


def test_mirror_resolver_returns_tree_sha_on_success(mod, monkeypatch) -> None:
    tree_sha = "a74566fd92b0ca9bb86919df5e7f804cc4307793"
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1:3] == ["init", "--quiet"] or "init" in cmd:
            return _completed(0)
        if "fetch" in cmd:
            return _completed(0)
        if "rev-parse" in cmd:
            return _completed(0, stdout=f"{tree_sha}\n")
        raise AssertionError(f"unexpected git invocation: {cmd}")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    sha, detail = mod.resolve_src_tree_sha_via_mirror("omnibase_core", "abc123")

    assert sha == tree_sha
    assert detail == "ok (mirror)"
    # init, fetch, rev-parse -- exactly three git invocations, no more.
    assert len(calls) == 3


def test_mirror_resolver_reports_init_failure(mod, monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        if "init" in cmd:
            return _completed(1, stderr="permission denied")
        raise AssertionError(f"should not reach: {cmd}")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    sha, detail = mod.resolve_src_tree_sha_via_mirror("omnibase_core", "abc123")

    assert sha is None
    assert "git init error" in detail
    assert "permission denied" in detail


def test_mirror_resolver_reports_fetch_failure(mod, monkeypatch) -> None:
    """The measured real-world failure mode (OMN-16063 C2b docstring):
    `fatal: remote error: upload-pack: not our ref <sha>` when the mirror
    does not carry the requested commit."""

    def fake_run(cmd, **kwargs):
        if "init" in cmd:
            return _completed(0)
        if "fetch" in cmd:
            return _completed(
                1, stderr="fatal: remote error: upload-pack: not our ref abc123"
            )
        raise AssertionError(f"should not reach rev-parse: {cmd}")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    sha, detail = mod.resolve_src_tree_sha_via_mirror("omnibase_core", "abc123")

    assert sha is None
    assert "could not fetch" in detail
    assert "not our ref" in detail


def test_mirror_resolver_reports_missing_src_tree(mod, monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        if "init" in cmd:
            return _completed(0)
        if "fetch" in cmd:
            return _completed(0)
        if "rev-parse" in cmd:
            return _completed(128, stderr="fatal: Not a valid object name")
        raise AssertionError(f"unexpected git invocation: {cmd}")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    sha, detail = mod.resolve_src_tree_sha_via_mirror("omnibase_core", "abc123")

    assert sha is None
    assert "no top-level 'src' tree" in detail


def test_mirror_resolver_handles_timeout(mod, monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout", 15.0))

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    sha, detail = mod.resolve_src_tree_sha_via_mirror("omnibase_core", "abc123")

    assert sha is None
    assert "timed out" in detail


def test_mirror_resolver_handles_missing_git_binary(mod, monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        raise OSError("git: command not found")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    sha, detail = mod.resolve_src_tree_sha_via_mirror("omnibase_core", "abc123")

    assert sha is None
    assert "mirror resolution failed" in detail


# ---------------------------------------------------------------------------
# resolve_src_tree_sha_hermetic -- mirror-first, live-fallback composition.
# ---------------------------------------------------------------------------


def test_hermetic_resolver_prefers_mirror_and_never_calls_live(
    mod, monkeypatch
) -> None:
    live_called = False

    def fake_mirror(repo, ref, **kwargs):
        return "mirror-tree-sha", "ok (mirror)"

    def fake_live(repo, ref):
        nonlocal live_called
        live_called = True
        return "live-tree-sha", "ok"

    monkeypatch.setattr(mod, "resolve_src_tree_sha_via_mirror", fake_mirror)
    monkeypatch.setattr(mod, "resolve_src_tree_sha", fake_live)

    sha, detail = mod.resolve_src_tree_sha_hermetic("omnibase_core", "abc123")

    assert sha == "mirror-tree-sha"
    assert detail == "ok (mirror)"
    assert live_called is False


def test_hermetic_resolver_falls_back_to_live_when_mirror_misses(
    mod, monkeypatch
) -> None:
    def fake_mirror(repo, ref, **kwargs):
        return None, "could not fetch 'abc123' from mirror: not our ref"

    def fake_live(repo, ref):
        return "live-tree-sha", "ok"

    monkeypatch.setattr(mod, "resolve_src_tree_sha_via_mirror", fake_mirror)
    monkeypatch.setattr(mod, "resolve_src_tree_sha", fake_live)

    sha, detail = mod.resolve_src_tree_sha_hermetic("omnibase_core", "abc123")

    assert sha == "live-tree-sha"
    assert detail == "ok"


def test_hermetic_resolver_reports_both_failures_when_neither_resolves(
    mod, monkeypatch
) -> None:
    def fake_mirror(repo, ref, **kwargs):
        return None, "mirror unreachable"

    def fake_live(repo, ref):
        return None, "transport error: timed out"

    monkeypatch.setattr(mod, "resolve_src_tree_sha_via_mirror", fake_mirror)
    monkeypatch.setattr(mod, "resolve_src_tree_sha", fake_live)

    sha, detail = mod.resolve_src_tree_sha_hermetic("omnibase_core", "abc123")

    assert sha is None
    assert "mirror unreachable" in detail
    assert "transport error: timed out" in detail


def test_hermetic_resolver_wired_as_check_lineage_default(mod) -> None:
    """The CLI's --check-lineage path must use the hermetic resolver, not the
    pure live one, on the merge-required gate (OMN-16096) -- a regression
    here silently reintroduces the live-API merge-path dependency this
    ticket closes."""
    import inspect

    source = inspect.getsource(mod.main)
    assert "resolve_src_tree_sha_hermetic" in source
