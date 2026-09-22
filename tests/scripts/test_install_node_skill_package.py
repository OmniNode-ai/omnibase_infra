# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Guard tests for the node-skill-package install script (OMN-13829, OMN-14060).

These are hermetic: they never hit the network or mutate a venv. Every exec
test pins ``OMNIMARKET_REF`` explicitly so the dynamic-resolution path (which
does a live ``git ls-remote``, OMN-14060) is never exercised here — that path
was verified manually against the live repo (see the OMN-14060 PR body) rather
than baked into this suite, to keep it network-independent. The tests below
validate the committed script's invariants (dynamic-by-default ref resolution
with a pinning override, --no-deps composition of the market provider layer,
an --execute gate, and portability) and exercise the dry-run path, which
prints the plan without installing anything.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "install-node-skill-package.sh"


def _script_text() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


# OMN-18675: the script reads the co-installed pin versions out of the ref's own
# pyproject.toml, so an exec test must name a ref it can actually resolve. A
# throwaway local clone keeps that hermetic — a CI runner has no $OMNI_HOME
# registry, and depending on one would make this suite pass only on a developer
# machine.
_FIXTURE_PYPROJECT = """\
[project]
name = "omnimarket"
version = "0.0.0"
dependencies = [
    "omnibase-compat==0.5.7",
    "omninode-memory==0.18.0",
]
"""


def _make_fixture_registry(root: Path) -> tuple[Path, str]:
    """Build a throwaway $OMNI_HOME/omnimarket clone; return (omni_home, sha)."""
    omni_home = root / "omni_home"
    clone = omni_home / "omnimarket"
    clone.mkdir(parents=True)

    def git(*argv: str) -> None:
        # GIT_DIR/GIT_WORK_TREE from a hook environment override cwd= and would
        # retarget the real worktree (OMN-14891).
        subprocess.run(
            argv,
            cwd=clone,
            check=True,
            capture_output=True,
            timeout=30,
            env=scrub_git_location_env(os.environ),
        )

    git("git", "init", "--quiet", "-b", "dev")
    git("git", "config", "user.email", "test@example.com")
    git("git", "config", "user.name", "Test")
    (clone / "pyproject.toml").write_text(_FIXTURE_PYPROJECT, encoding="utf-8")
    git("git", "add", "pyproject.toml")
    git("git", "commit", "--quiet", "-m", "fixture")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=clone,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()
    return omni_home, sha


def _make_worktree_fixture_registry(root: Path) -> tuple[Path, str]:
    """Build a linked worktree at the canonical registry location."""
    omni_home = root / "omni_home"
    clone = omni_home / "omnimarket"
    source = root / "market-source"
    omni_home.mkdir()
    source.mkdir()

    def git(cwd: Path, *args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=scrub_git_location_env(os.environ),
        )
        return result.stdout.strip()

    git(source, "init", "--quiet", "-b", "dev")
    git(source, "config", "user.email", "test@example.com")
    git(source, "config", "user.name", "Test")
    (source / "pyproject.toml").write_text(_FIXTURE_PYPROJECT, encoding="utf-8")
    git(source, "add", "pyproject.toml")
    git(source, "commit", "--quiet", "-m", "fixture")
    sha = git(source, "rev-parse", "HEAD")
    git(source, "worktree", "add", "--quiet", "-b", "registry", str(clone))
    return omni_home, sha


def _run_pinned_dry_run(
    omni_home: Path, sha: str, tmp_path: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SCRIPT), sys.executable],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={
            **os.environ,
            "PATH": f"{_fake_uv_on_path(tmp_path)}{os.pathsep}{os.environ['PATH']}",
            "OMNIMARKET_REF": sha,
            "OMNI_HOME": str(omni_home),
        },
    )


def _fake_uv_on_path(root: Path) -> Path:
    """A `uv` that prints an empty change plan, so no network or venv is touched.

    The plan path now runs a real `uv pip install --dry-run` (OMN-18675), which
    would otherwise try to fetch the fixture sha from github. Stubbing uv keeps
    this suite offline while still driving the script's own plan plumbing.
    """
    bin_dir = root / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "uv"
    fake.write_text(
        "#!/usr/bin/env bash\necho 'Audited 3 packages in 1ms'\nexit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return bin_dir


def test_script_exists_and_executable() -> None:
    assert _SCRIPT.is_file(), f"missing install script: {_SCRIPT}"
    mode = _SCRIPT.stat().st_mode
    assert mode & 0o111, "install script must be executable"


def test_script_has_spdx_header() -> None:
    head = _script_text().splitlines()[:4]
    joined = "\n".join(head)
    assert "SPDX-License-Identifier: MIT" in joined
    assert "SPDX-FileCopyrightText" in joined


def test_resolves_ref_dynamically_from_live_dev_head() -> None:
    # OMN-14060: a hand-edited SHA literal goes stale the moment omnimarket@dev
    # advances past it (the OMN-13829 recurrence mechanism). The default path
    # must resolve from the live `dev` branch, never a baked-in literal.
    text = _script_text()
    assert "git ls-remote" in text
    assert '"$OMNIMARKET_GIT" dev' in text or 'OMNIMARKET_GIT}" dev' in text


def test_omnimarket_ref_override_still_takes_precedence() -> None:
    # An operator-set OMNIMARKET_REF (pinned/offline use) must win outright —
    # the dynamic resolution branch must never run when it's set.
    text = _script_text()
    assert 'if [[ -n "${OMNIMARKET_REF:-}" ]]; then' in text


def test_has_offline_fallback_to_local_canonical_clone() -> None:
    # When `git ls-remote` is unreachable (offline) and no explicit override is
    # given, fall back to the already-checked-out local clone at
    # $OMNI_HOME/omnimarket rather than hard-failing outright.
    text = _script_text()
    assert "OMNI_HOME" in text
    assert "rev-parse --show-toplevel" in text
    assert "rev-parse HEAD" in text


def test_fails_fast_when_ref_cannot_be_resolved() -> None:
    # No silent fallback to a stale default (CLAUDE.md rule #8) — if neither
    # live resolution nor the local-clone fallback succeeds, exit non-zero
    # with an actionable message.
    text = _script_text()
    assert "could not resolve an omnimarket ref" in text
    assert "exit 1" in text


def test_no_deps_used_for_market_provider_layer() -> None:
    text = _script_text()
    # omnimarket sits ABOVE the infra layer; it must be composed --no-deps so its
    # metadata never re-resolves (or downgrades) the infra layer beneath it.
    assert "--no-deps" in text
    assert "OmniNode-ai/omnimarket.git" in text
    # OMN-18675: the NAMES are fixed here, the VERSIONS come from the ref being
    # installed. A version literal beside either name is the recurrence
    # mechanism that downgraded a shared venv and broke the onex CLI host-wide.
    assert "OMNI_INTERNAL_NO_DEPS_PKGS=(omnibase-compat omninode-memory)" in text
    assert 'COMPAT_PIN="omnibase-compat==' not in text
    assert 'MEMORY_PIN="omninode-memory==' not in text


def test_verifies_merge_sweep_and_session_nodes() -> None:
    text = _script_text()
    # DoD: after install, the nodes behind `onex skill merge_sweep` and
    # `onex skill session` must resolve. Step 3 asserts both entry points.
    assert "node_pr_lifecycle_orchestrator" in text
    assert "node_session_orchestrator" in text


def test_framed_as_canonical_not_interim() -> None:
    # Operator-confirmed direction (OMN-13829): this co-install is the CANONICAL
    # mechanism, not a stopgap. Guard against interim/retire framing regressing.
    lowered = _script_text().lower()
    assert "canonical co-install" in lowered
    for banned in ("retire this script", "interim", "workaround", "stopgap"):
        assert banned not in lowered, f"interim-framing token present: {banned!r}"


def test_has_execute_gate() -> None:
    text = _script_text()
    assert "--execute" in text, "script must gate mutation behind --execute"
    assert "DRY RUN" in text, "script must default to a dry-run plan"


def test_no_hardcoded_absolute_machine_paths() -> None:
    # CLAUDE.md rule #6: no /Users/ or /Volumes/ absolute paths in source.
    text = _script_text()
    for token in ("/Users/", "/Volumes/"):
        assert token not in text, f"hardcoded machine path {token!r} present"


def test_dry_run_prints_plan_and_does_not_require_execute() -> None:
    # Dry run against a non-existent python still prints nothing that installs;
    # it fails fast on interpreter resolution rather than touching any venv.
    result = subprocess.run(
        ["bash", str(_SCRIPT), "/nonexistent/python"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "not executable" in (result.stdout + result.stderr)


def test_dry_run_with_current_interpreter_prints_plan(tmp_path: Path) -> None:
    # Using the running interpreter (guaranteed executable) exercises the plan
    # print path; without --execute it must not install anything. OMNIMARKET_REF
    # is pinned to a throwaway fixture clone so this never triggers the live
    # `git ls-remote` resolution path (OMN-14060) and still names a ref whose
    # pyproject the OMN-18675 pin resolution can read.
    omni_home, pinned_test_ref = _make_fixture_registry(tmp_path)
    result = subprocess.run(
        ["bash", str(_SCRIPT), sys.executable],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={
            **os.environ,
            "PATH": f"{_fake_uv_on_path(tmp_path)}{os.pathsep}{os.environ['PATH']}",
            "OMNIMARKET_REF": pinned_test_ref,
            "OMNI_HOME": str(omni_home),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "node-skill-package install plan" in result.stdout
    assert "DRY RUN" in result.stdout
    assert pinned_test_ref in result.stdout
    assert "OMNIMARKET_REF override (pinned/offline use)" in result.stdout
    # The pins printed are the fixture ref's, not any literal in the script.
    assert "omnibase-compat==0.5.7" in result.stdout
    assert "omninode-memory==0.18.0" in result.stdout


def test_dry_run_accepts_canonical_omnimarket_worktree(tmp_path: Path) -> None:
    omni_home, sha = _make_worktree_fixture_registry(tmp_path)
    assert (omni_home / "omnimarket" / ".git").is_file()

    result = _run_pinned_dry_run(omni_home, sha, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert sha in result.stdout
    assert "DRY RUN" in result.stdout


def test_dry_run_rejects_nested_subdirectory_as_canonical_root(tmp_path: Path) -> None:
    repo = tmp_path / "outer-repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            capture_output=True,
            timeout=30,
            env=scrub_git_location_env(os.environ),
        )

    git("init", "--quiet", "-b", "dev")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    nested_market = repo / "omnimarket"
    nested_market.mkdir()
    (nested_market / "pyproject.toml").write_text(_FIXTURE_PYPROJECT, encoding="utf-8")
    git("add", "omnimarket/pyproject.toml")
    git("commit", "--quiet", "-m", "nested fixture")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()

    result = _run_pinned_dry_run(repo, sha, tmp_path)

    assert result.returncode != 0
    assert "repo registry containing an omnimarket clone" in (
        result.stdout + result.stderr
    )
    assert "node-skill-package install plan" not in result.stdout


def test_dry_run_rejects_nonrepository_with_git_named_file(tmp_path: Path) -> None:
    omni_home = tmp_path / "omni_home"
    clone = omni_home / "omnimarket"
    clone.mkdir(parents=True)
    (clone / ".git").write_text("not git metadata", encoding="utf-8")

    result = _run_pinned_dry_run(omni_home, "a" * 40, tmp_path)

    assert result.returncode != 0
    assert "repo registry containing an omnimarket clone" in (
        result.stdout + result.stderr
    )
    assert "node-skill-package install plan" not in result.stdout


def test_dry_run_without_override_does_not_touch_network_before_python_check() -> None:
    # Reordered (OMN-14060): interpreter resolution runs BEFORE ref resolution,
    # so a bad interpreter path fails fast without ever doing a `git ls-remote`.
    # No OMNIMARKET_REF is set here on purpose -- this proves the ordering.
    env = {k: v for k, v in os.environ.items() if k != "OMNIMARKET_REF"}
    result = subprocess.run(
        ["bash", str(_SCRIPT), "/nonexistent/python"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=env,
    )
    assert result.returncode != 0
    assert "not executable" in (result.stdout + result.stderr)
    assert "Resolving omnimarket ref" not in (result.stdout + result.stderr)
