# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Hermetic tests for the dep-provenance content-lineage check (OMN-15604).

Root cause this closes: ``find_violations`` (OMN-13873) forbids a git source
for a first-party dep, but a line carrying a well-formed
``# raw-override-ok: <ticket>`` token is exempt from that rule
*unconditionally and forever* -- the token is validated only for shape, never
against whether the pinned rev's content matches the PyPI version declared
alongside it. Live incident this reproduces: ``omnibase_infra@dev`` declared
``omnibase-core==0.46.8`` while pinning git rev ``3d51b047`` (escaped via
``# raw-override-ok: OMN-15414``) whose ``src/`` tree measurably DIFFERED from
released tag ``v0.46.8``.

This module is hermetic (no network): every case injects a fake ``resolve``
callable into ``find_lineage_violations`` so the pure decision logic
(declared-version parsing, source/version cross-referencing, escape-token
independence, message shape) is exercised without hitting GitHub. The LIVE
half -- proving the checker against the real ``3d51b047`` vs ``v0.46.8`` pin
via the real GitHub REST API -- lives in
``tests/integration/ci/test_dep_provenance_lineage_live_omn15604.py``, for the
same reason the OMN-15538 pin-reachability gate splits the same way: the
pre-push selector always ignores ``tests/integration``, so a transient network
failure there can never make a branch locally unpushable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_dep_provenance.py"

# The two real tree SHAs from the live incident (OMN-15604 ticket evidence,
# independently reproduced against the canonical omnibase_core clone during
# this lane): `git rev-parse 3d51b047:src` and `git rev-parse v0.46.8:src`.
_PINNED_REV = "3d51b047a43ee412a7521502619d35c216dc7811"
_PINNED_SRC_TREE = "a74566fd92b0ca9bb86919df5e7f804cc4307793"
_RELEASED_TAG = "v0.46.8"
_RELEASED_SRC_TREE = "008efdba12b39cf04d90d17468523daa281fe4fd"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_dep_provenance", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return _load_module()


def _write_pyproject(
    tmp_path: Path,
    *,
    dependencies: str = '"omnibase-core==0.46.8",',
    sources_block: str,
) -> Path:
    content = (
        "[project]\n"
        'name = "omnibase-infra"\n'
        'version = "0.0.0"\n'
        "dependencies = [\n"
        f"    {dependencies}\n"
        "]\n"
        "\n"
        f"{sources_block}"
        "\n"
        "[tool.ruff]\n"
        'target-version = "py312"\n'
    )
    path = tmp_path / "pyproject.toml"
    path.write_text(content)
    return path


def _fixed_resolver(mapping: dict[tuple[str, str], tuple[str | None, str]]):
    """Build a fake `resolve` callable from a fixed {(repo, ref): (sha, detail)} map."""

    def _resolve(repo: str, ref: str) -> tuple[str | None, str]:
        try:
            return mapping[(repo, ref)]
        except KeyError:
            return None, f"unexpected probe: repo={repo!r} ref={ref!r}"

    return _resolve


# ---------------------------------------------------------------------------
# RED: reproduce the exact live incident (3d51b047 vs v0.46.8) with a fake
# resolver returning the REAL tree SHAs measured against the live pin --
# proving RED against exists-but-wrong, not a synthetic value.
# ---------------------------------------------------------------------------


def test_red_reproduces_the_live_incident_pin(mod, tmp_path: Path) -> None:
    block = (
        "[tool.uv.sources]\n"
        'omnibase-core = { git = "https://github.com/OmniNode-ai/omnibase_core.git", '
        f'rev = "{_PINNED_REV}" }}  # raw-override-ok: OMN-15414\n'
    )
    path = _write_pyproject(tmp_path, sources_block=block)
    resolver = _fixed_resolver(
        {
            ("omnibase_core", _PINNED_REV): (_PINNED_SRC_TREE, "ok"),
            ("omnibase_core", _RELEASED_TAG): (_RELEASED_SRC_TREE, "ok"),
        }
    )

    violations = mod.find_lineage_violations(path.read_text(), resolve=resolver)

    assert len(violations) == 1
    assert "omnibase-core" in violations[0]
    assert _PINNED_SRC_TREE in violations[0]
    assert _RELEASED_SRC_TREE in violations[0]
    assert "differs from" in violations[0]


def test_red_applies_even_with_a_valid_escape_token(mod, tmp_path: Path) -> None:
    """The whole point of OMN-15604: find_violations exempts an escaped line;
    find_lineage_violations must NOT -- the token only exempts the
    "forbid git source" rule, never the content-matches-version rule."""
    block = (
        "[tool.uv.sources]\n"
        'omnibase-core = { git = "https://github.com/OmniNode-ai/omnibase_core.git", '
        f'rev = "{_PINNED_REV}" }}  # raw-override-ok: OMN-15414\n'
    )
    path = _write_pyproject(tmp_path, sources_block=block)
    text = path.read_text()
    resolver = _fixed_resolver(
        {
            ("omnibase_core", _PINNED_REV): (_PINNED_SRC_TREE, "ok"),
            ("omnibase_core", _RELEASED_TAG): (_RELEASED_SRC_TREE, "ok"),
        }
    )

    # find_violations is fooled by the token (this is #2 of the two facts the
    # ticket says are "stored side by side and never compared").
    assert mod.find_violations(text) == []
    # find_lineage_violations is not.
    assert len(mod.find_lineage_violations(text, resolve=resolver)) == 1


# ---------------------------------------------------------------------------
# GREEN: pinned rev's src/ tree matches the released tag's src/ tree.
# ---------------------------------------------------------------------------


def test_green_when_pinned_tree_matches_released_tree(mod, tmp_path: Path) -> None:
    block = (
        "[tool.uv.sources]\n"
        'omnibase-core = { git = "https://github.com/OmniNode-ai/omnibase_core.git", '
        'rev = "105f7ce0a8f4b31f6f01fc94e9b43e75984f166a" }\n'
    )
    path = _write_pyproject(tmp_path, sources_block=block)
    resolver = _fixed_resolver(
        {
            ("omnibase_core", "105f7ce0a8f4b31f6f01fc94e9b43e75984f166a"): (
                _RELEASED_SRC_TREE,
                "ok",
            ),
            ("omnibase_core", _RELEASED_TAG): (_RELEASED_SRC_TREE, "ok"),
        }
    )

    assert mod.find_lineage_violations(path.read_text(), resolve=resolver) == []


def test_green_when_no_uv_sources_override_present(mod, tmp_path: Path) -> None:
    """Steady state after AC1 (the git override deleted): nothing to compare,
    zero resolver calls -- lineage check is a cheap no-op."""
    calls: list[tuple[str, str]] = []

    def _resolve(repo: str, ref: str) -> tuple[str | None, str]:
        calls.append((repo, ref))
        return None, "should not be called"

    block = "[tool.uv.sources]\n"
    path = _write_pyproject(tmp_path, sources_block=block)

    assert mod.find_lineage_violations(path.read_text(), resolve=_resolve) == []
    assert calls == []


def test_green_when_git_override_has_no_declared_version(mod, tmp_path: Path) -> None:
    """A git override for a package with NO `pkg==X.Y.Z` constraint has
    nothing to compare against -- that shape is find_violations' problem
    (forbidden override), not a lineage-mismatch problem."""

    def _resolve(repo: str, ref: str) -> tuple[str | None, str]:
        raise AssertionError("resolver must not be called with no declared version")

    path = _write_pyproject(
        tmp_path,
        dependencies='"omnibase-core>=0.46.1,<0.47.0",',
        sources_block=(
            "[tool.uv.sources]\n"
            'omnibase-core = { git = "https://github.com/OmniNode-ai/omnibase_core.git", '
            'rev = "deadbeef" }\n'
        ),
    )
    assert mod.find_lineage_violations(path.read_text(), resolve=_resolve) == []


# ---------------------------------------------------------------------------
# UNDETERMINED: network/resolution failure fails closed by default.
# ---------------------------------------------------------------------------


def test_undetermined_when_resolver_cannot_resolve(mod, tmp_path: Path) -> None:
    block = (
        "[tool.uv.sources]\n"
        'omnibase-core = { git = "https://github.com/OmniNode-ai/omnibase_core.git", '
        f'rev = "{_PINNED_REV}" }}\n'
    )
    path = _write_pyproject(tmp_path, sources_block=block)

    def _resolve(repo: str, ref: str) -> tuple[str | None, str]:
        return None, "transport error: simulated"

    violations = mod.find_lineage_violations(path.read_text(), resolve=_resolve)
    assert len(violations) == 1
    assert "UNDETERMINED lineage" in violations[0]


# ---------------------------------------------------------------------------
# _declared_versions / _repo_from_git_url unit coverage
# ---------------------------------------------------------------------------


def test_declared_versions_prefers_override_dependencies(mod) -> None:
    parsed = {
        "project": {"dependencies": ["omnibase-core==0.46.7"]},
        "tool": {"uv": {"override-dependencies": ["omnibase-core==0.46.8"]}},
    }
    assert mod._declared_versions(parsed) == {"omnibase-core": "0.46.8"}


def test_declared_versions_ignores_range_constraints(mod) -> None:
    parsed = {"project": {"dependencies": ["omnibase-spi>=0.23.0,<0.24.0"]}}
    assert mod._declared_versions(parsed) == {}


def test_repo_from_git_url_extracts_bare_repo_name(mod) -> None:
    assert (
        mod._repo_from_git_url("https://github.com/OmniNode-ai/omnibase_core.git")
        == "omnibase_core"
    )
    assert (
        mod._repo_from_git_url("https://github.com/other-org/omnibase_core.git") is None
    )


# ---------------------------------------------------------------------------
# CLI wiring: --check-lineage is opt-in and off by default.
# ---------------------------------------------------------------------------


def test_check_lineage_flag_is_off_by_default(
    mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default no-flag invocation never imports/calls the network resolver --
    pre-commit's zero-arg invocation stays offline (unchanged from OMN-13873)."""
    block = (
        "[tool.uv.sources]\n"
        'omnibase-core = { git = "https://github.com/OmniNode-ai/omnibase_core.git", '
        f'rev = "{_PINNED_REV}" }}  # raw-override-ok: OMN-15414\n'
    )
    path = _write_pyproject(tmp_path, sources_block=block)

    def _boom(repo: str, ref: str) -> tuple[str | None, str]:
        raise AssertionError("network resolver must not run without --check-lineage")

    monkeypatch.setattr(mod, "resolve_src_tree_sha", _boom)
    # No --check-lineage: find_violations sees the escape token and exits 0,
    # and find_lineage_violations is never invoked at all.
    assert mod.main(["--pyproject", str(path)]) == 0


def test_allow_undetermined_lineage_refused_under_ci(
    mod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CI", "true")
    path = _write_pyproject(tmp_path, sources_block="[tool.uv.sources]\n")
    assert (
        mod.main(
            [
                "--pyproject",
                str(path),
                "--check-lineage",
                "--allow-undetermined-lineage",
            ]
        )
        == 2
    )
