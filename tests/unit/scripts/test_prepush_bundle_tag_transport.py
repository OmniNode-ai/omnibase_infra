# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The pre-push remote leg must transplant the TAG STATE, not just HEAD (OMN-17240).

``scripts/hooks/prepush_dispatch.sh`` ships the tree to a lab host as a git
bundle and clones it there. It built that bundle with::

    git bundle create "$bundle" HEAD

which packs only the commits reachable from ``HEAD`` and **no ref under
``refs/tags/``**. Every remote clone, on every host, on every push, therefore had
zero tags -- and ``scripts/check_release_identity.py`` derives "the latest
published version" from ``git tag --list``. With no tags it took its "no published
tag yet" branch and printed a message the suite does not expect, so three tests
went red remotely while passing locally at the identical SHA (first observed on
h101 at OMN-17139's ``47d7da183``).

Three separate facts are pinned here, because the obvious one-line fix is wrong:

1. **The defect itself.** A ``HEAD``-only bundle clones to a tree with zero tags
   even when the source repo has many.

2. **``--tags`` alone is not the fix, and is worse on a shallow source.**
   ``git bundle create f HEAD --tags`` run against a shallow repository exits 0
   and writes a bundle whose header lists every tag ref, but cloning that bundle
   dies with "remote did not send all necessary objects" because the tags'
   ancestry lies beyond the shallow graft. The canonical ``omnibase_infra`` clone
   *was* shallow when this was diagnosed, so the one-liner would have converted a
   false red into a hard transport failure on every push.

3. **The shipped transport.** ``prepush_bundle_tree`` proves the source can bundle
   tag ancestry (unshallowing once when it cannot), bundles ``HEAD`` and the tags,
   and then proves the written bundle actually carries tag refs before it is
   shipped. Every unprovable step returns non-zero -- "no evidence" -- which sends
   the caller back to its existing precedence. Nothing here can make the gate
   accept less work.

The shell is extract-and-executed (the pattern already used for this hook's other
pure functions) so these assertions run THE code that ships, never a Python
re-implementation that could pass while the shipped transport is broken.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LIB = REPO_ROOT / "scripts" / "hooks" / "prepush_dispatch.sh"

pytestmark = pytest.mark.unit

_GIT_ID = ("-c", "user.email=t@t", "-c", "user.name=t")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *_GIT_ID, *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _git_rc(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *_GIT_ID, *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _tagged_origin(tmp_path: Path, *, tags: int = 6, name: str = "origin") -> Path:
    """A real repo with TAGS release tags spread across TAGS+2 commits.

    Tags are deliberately placed on OLD commits, so a clone shallow enough to
    exclude them reproduces the canonical clone's shape at diagnosis time.
    """
    repo = tmp_path / name
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "dev", ".")
    for i in range(tags + 2):
        (repo / "f.txt").write_text(f"rev {i}\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", f"c{i}")
        if i < tags:
            _git(repo, "tag", f"v0.{i}.0")
    return repo


def _clone(src: Path, dst: Path, *extra: str) -> Path:
    subprocess.run(
        ["git", "clone", "-q", *extra, str(src), str(dst)],
        capture_output=True,
        text=True,
        check=True,
    )
    return dst


def _shallow_clone(src: Path, dst: Path) -> Path:
    """A genuinely shallow clone.

    ``--no-local`` is required: over the local transport git hardlinks the whole
    object store and silently ignores ``--depth``, so without it the fixture is
    not shallow and the case under test never occurs.
    """
    return _clone(src, dst, "--depth", "1", "--no-tags", "--no-local")


def _tags(repo: Path) -> list[str]:
    return [t for t in _git(repo, "tag", "--list").splitlines() if t.strip()]


def _run_lib(body: str) -> subprocess.CompletedProcess[str]:
    """Run BODY with the real dispatch library sourced and ``log`` stubbed."""
    script = f"""
set -uo pipefail
REPO_ROOT=/nonexistent
log() {{ printf '[t] %s\\n' "$1" >&2; }}
die() {{ printf 'DIE: %s\\n' "$1" >&2; exit 1; }}
. {LIB}
{body}
"""
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        stdin=subprocess.DEVNULL,
        env={
            **os.environ,
            "PREPUSH_LOAD_OVERRIDE_MAP": "",
            "PREPUSH_SLOT_OVERRIDE_MAP": "",
        },
    )


# =============================================================================
# 1. The defect: a HEAD-only bundle carries no tags
# =============================================================================


def test_head_only_bundle_reproduces_the_tagless_remote_tree(tmp_path: Path) -> None:
    """The pre-fix transport: 6 tags at the source, 0 tags on the remote host."""
    src = _tagged_origin(tmp_path)
    bundle = tmp_path / "head-only.bundle"
    _git(src, "bundle", "create", str(bundle), "HEAD")

    landed = _clone(bundle, tmp_path / "landed")

    assert len(_tags(src)) == 6
    assert _tags(landed) == [], (
        "a HEAD-only bundle must be shown to strand the tag state -- this is the "
        "OMN-17240 defect being fixed"
    )


# =============================================================================
# 2. `--tags` alone is not the fix: it breaks outright on a shallow source
# =============================================================================


def test_naive_tags_bundle_from_a_shallow_source_writes_an_unclonable_bundle(
    tmp_path: Path,
) -> None:
    """`git bundle create ... HEAD --tags` exits 0 on a shallow repo, then fails to clone.

    This is why the one-line ``--tags`` fix proposed on the ticket is refused:
    the canonical clone was shallow, so it would have replaced a false red with a
    hard transport failure on every push.
    """
    origin = _tagged_origin(tmp_path)
    shallow = _shallow_clone(origin, tmp_path / "shallow")
    assert _git(shallow, "rev-parse", "--is-shallow-repository").strip() == "true"

    # Bring a tag across at depth 1, so the tag's commit is present but its
    # parents sit beyond a graft -- the exact shape the canonical clone was in.
    _git(shallow, "fetch", "-q", "--depth", "1", "origin", "tag", "v0.3.0")
    assert _tags(shallow) == ["v0.3.0"]

    bundle = tmp_path / "naive.bundle"
    created = _git_rc(shallow, "bundle", "create", str(bundle), "HEAD", "--tags")
    assert created.returncode == 0, "bundle create reports success even when truncated"

    cloned = subprocess.run(
        ["git", "clone", "-q", str(bundle), str(tmp_path / "naive-landed")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cloned.returncode != 0, (
        "a --tags bundle built from a shallow source must be shown to be unclonable"
    )
    assert "necessary objects" in cloned.stderr or "Could not read" in cloned.stderr


# =============================================================================
# 3. The shipped transport
# =============================================================================


def test_bundle_tree_transports_every_tag_from_a_full_source(tmp_path: Path) -> None:
    origin = _tagged_origin(tmp_path)
    full = _clone(origin, tmp_path / "full")
    bundle = tmp_path / "tree.bundle"

    run = _run_lib(f'prepush_bundle_tree "{full}" "{bundle}"; echo "RC=$?"')
    assert "RC=0" in run.stdout, run.stderr
    assert bundle.is_file()

    landed = _clone(bundle, tmp_path / "landed")
    assert (
        sorted(_tags(landed)) == sorted(_tags(full)) == [f"v0.{i}.0" for i in range(6)]
    )
    assert (
        _git(landed, "rev-parse", "HEAD").strip()
        == _git(full, "rev-parse", "HEAD").strip()
    )


def test_bundle_tree_unshallows_a_shallow_source_and_then_carries_the_tags(
    tmp_path: Path,
) -> None:
    origin = _tagged_origin(tmp_path)
    shallow = _shallow_clone(origin, tmp_path / "shallow")
    assert _git(shallow, "rev-parse", "--is-shallow-repository").strip() == "true"
    assert _tags(shallow) == []

    bundle = tmp_path / "tree.bundle"
    run = _run_lib(f'prepush_bundle_tree "{shallow}" "{bundle}"; echo "RC=$?"')
    assert "RC=0" in run.stdout, run.stderr
    assert _git(shallow, "rev-parse", "--is-shallow-repository").strip() == "false"

    landed = _clone(bundle, tmp_path / "landed")
    assert sorted(_tags(landed)) == [f"v0.{i}.0" for i in range(6)]


def test_bundle_tree_refuses_when_a_shallow_source_cannot_be_unshallowed(
    tmp_path: Path,
) -> None:
    """No reachable origin -> refuse the leg rather than ship a tag-less tree."""
    origin = _tagged_origin(tmp_path)
    shallow = _shallow_clone(origin, tmp_path / "shallow")
    _git(shallow, "remote", "set-url", "origin", str(tmp_path / "no-such-origin"))

    bundle = tmp_path / "tree.bundle"
    run = _run_lib(f'prepush_bundle_tree "{shallow}" "{bundle}"; echo "RC=$?"')
    assert "RC=1" in run.stdout, run.stderr
    assert "OMN-17240" in run.stderr
    assert not bundle.exists(), "a refused transport must not leave a bundle to ship"


def test_bundle_tree_refuses_a_bundle_that_lost_the_tag_refs(tmp_path: Path) -> None:
    """The post-write proof: tags at the source but none in the bundle -> refuse.

    Exercised by making ``git bundle create`` write a HEAD-only bundle (the
    pre-fix behaviour) through a shim earlier on PATH, so the guard is proven
    against the exact regression it exists to catch rather than a mock.
    """
    origin = _tagged_origin(tmp_path)
    full = _clone(origin, tmp_path / "full")

    shimdir = tmp_path / "shim"
    shimdir.mkdir()
    real_git = subprocess.run(
        ["bash", "-c", "command -v git"], capture_output=True, text=True, check=True
    ).stdout.strip()
    (shimdir / "git").write_text(
        "#!/usr/bin/env bash\n"
        "# Drops a trailing --tags from `git bundle create`, reproducing the\n"
        "# pre-OMN-17240 HEAD-only transport.\n"
        'args=(); for a in "$@"; do [ "$a" = "--tags" ] || args+=("$a"); done\n'
        f'exec {real_git} "${{args[@]}}"\n',
        encoding="utf-8",
    )
    (shimdir / "git").chmod(0o755)

    bundle = tmp_path / "tree.bundle"
    run = _run_lib(
        f'export PATH="{shimdir}:$PATH"\n'
        f'prepush_bundle_tree "{full}" "{bundle}"; echo "RC=$?"'
    )
    assert "RC=1" in run.stdout, run.stderr
    assert "OMN-17240" in run.stderr
    assert not bundle.exists()


def test_bundle_tree_succeeds_on_a_tagless_repo_without_a_remote(
    tmp_path: Path,
) -> None:
    """A repo that genuinely has no tags is not a transport failure."""
    repo = tmp_path / "fresh"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "dev", ".")
    (repo / "f.txt").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "c0")

    bundle = tmp_path / "tree.bundle"
    run = _run_lib(f'prepush_bundle_tree "{repo}" "{bundle}"; echo "RC=$?"')
    assert "RC=0" in run.stdout, run.stderr
    assert bundle.is_file()


# =============================================================================
# 4. The call site actually uses it
# =============================================================================


def test_remote_leg_calls_the_tag_carrying_transport() -> None:
    text = LIB.read_text(encoding="utf-8")
    assert 'prepush_bundle_tree "$REPO_ROOT" "$bundle"' in text, (
        "prepush_remote_run must build its bundle through prepush_bundle_tree"
    )
    assert 'bundle create "$bundle" HEAD > /dev/null' not in text, (
        "the HEAD-only bundle call must be gone from the remote leg (OMN-17240)"
    )
