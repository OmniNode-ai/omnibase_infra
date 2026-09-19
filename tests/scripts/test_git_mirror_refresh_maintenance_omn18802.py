# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The git mirrors must maintain themselves, not defer to an absent runbook.

WHY THIS FILE EXISTS (OMN-18802). `docker/runners/git-mirror-refresh.sh` sets
`gc.auto 0` on every mirror it clones, with the comment "`git gc` is run
explicitly by the runbook". The runbook it names --
`docs/runbooks/c2-git-mirror-egress-rollout.md`, which is also the
`Documentation=` target of `omninode-git-mirror-daemon.service` -- has never
existed in this repository. Nothing ever repacked the mirrors, and for four
months that sentence read as coverage.

Compounding it, git's default `fetch.unpackLimit` of 100 UNPACKS any fetch
bringing fewer than 100 objects into individual loose files. This timer fires
every ~2 minutes and almost every pass brings a handful, so the mirrors
accumulated loose objects indefinitely and packed essentially nothing.

Measured on `192.168.86.201` on 2026-09-19, the `onex_change_control` mirror:

    153,451 loose objects / 4.52 GiB    against 385 MiB of packed content
    389 packs / 11,338 refs

Serving a clone out of that took 47-264s where `omnimarket` (226 MiB, 12
packs) took 5s. That is what spent the runner job-started hook's whole
wall clock and cancelled both OCC publisher jobs at their 5-minute budget --
the other half of OMN-18802, pinned by
test_runner_job_started_hook_budget_omn18802.py.

WHAT IS PINNED HERE. That maintenance is a MECHANISM in the script rather
than a reference to a document: the unpack limits that stop the accumulation,
and a threshold-gated repack that runs under the lock the script already
holds. The end-to-end test calls the script's own `maintain_mirror_packs`
against a real fragmented mirror and asserts it comes back consolidated with
every ref intact; its positive control asserts a tidy mirror is left alone,
so a green can never be "the function repacks everything unconditionally".
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REFRESH_SCRIPT = REPO_ROOT / "docker" / "runners" / "git-mirror-refresh.sh"

pytestmark = [pytest.mark.unit]


_MAIN_BODY_MARKER = '\nmkdir -p "${MIRROR_ROOT}"\n'


def _script_text() -> str:
    return REFRESH_SCRIPT.read_text()


def _functions_only_script() -> str:
    """Everything before the refresh script's main body starts executing.

    Sourcing this lets a test call `maintain_mirror_packs` directly without
    the main loop trying to clone the nine production repos from github.com.
    """
    text = _script_text()
    idx = text.index(_MAIN_BODY_MARKER)
    assert idx > 0, (
        f"expected marker 'mkdir -p \"${{MIRROR_ROOT}}\"' in {REFRESH_SCRIPT} "
        "-- the function-defs/main-body boundary moved"
    )
    return text[:idx]


# Authorship for fixture commits. Merged into a freshly scrubbed environment at
# every call site rather than cached, because the guard (OMN-14891) reads the
# `env=` expression statically: it has to SEE `scrub_git_location_env` there,
# and a local variable holding the result carries no such evidence.
_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def _git(*args: str, cwd: Path | None = None) -> str:
    # OMN-18434: git exports GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE into
    # every hook environment and those OVERRIDE both `cwd=` and `git -C`. A
    # fixture that shells out to git under a pre-push hook would mutate the
    # REAL invoking worktree rather than tmp_path.
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        env=scrub_git_location_env(os.environ) | _GIT_IDENTITY,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def test_serving_config_sets_the_unpack_limits() -> None:
    """Every refresh must write a pack, not loose objects.

    This is the setting that stops the accumulation at source. Without it a
    repack is a bailing-out exercise against a leak that never closes.
    """
    text = _script_text()
    start = text.index("apply_mirror_serving_config() {")
    body = text[start : text.index("\n}\n", start)]
    for key in ("fetch.unpackLimit", "transfer.unpackLimit"):
        assert f"config {key} 1" in body, (
            f"`apply_mirror_serving_config` does not set {key} to 1 in "
            f"{REFRESH_SCRIPT}. git's default of 100 unpacks every small "
            "fetch into loose objects, which is how the onex_change_control "
            "mirror reached 153,451 loose objects / 4.52 GiB (OMN-18802)."
        )


def test_serving_config_is_applied_on_every_pass_not_only_at_clone() -> None:
    """A mirror cloned before this existed must pick the settings up.

    The call outside the `if [[ ! -d ... ]]` clone branch is the one that
    makes this true; without it the fix reaches only mirrors created later.
    """
    text = _script_text()
    loop = text[text.index('for repo in "${MIRROR_REPOS[@]}"') :]
    after_clone = loop[loop.index("        continue\n    fi\n") :]
    assert 'apply_mirror_serving_config "${mirror_dir}"' in after_clone, (
        "`apply_mirror_serving_config` is not called on the already-cloned "
        "path; existing mirrors would never pick up the unpack limits."
    )


def test_maintenance_exists_as_code_rather_than_a_runbook_reference() -> None:
    """The repack is in the script.

    `gc.auto 0` plus "the runbook does it" plus no runbook is how this
    surface went four months with no maintenance at all.
    """
    text = _script_text()
    assert "maintain_mirror_packs() {" in text, (
        f"no `maintain_mirror_packs` function in {REFRESH_SCRIPT}. Mirror "
        "maintenance must be a mechanism here, not a deferral to "
        "docs/runbooks/c2-git-mirror-egress-rollout.md, which does not exist."
    )
    assert 'maintain_mirror_packs "${mirror_dir}" "${repo}"' in text, (
        "`maintain_mirror_packs` is defined but never called."
    )


def test_maintenance_runs_under_the_existing_flock() -> None:
    """Serialization is the whole point of this script; maintenance inherits it.

    A repack racing a refresh, or two repacks racing each other, is exactly
    the IO spike the original `gc.auto 0` comment was worried about.
    """
    text = _script_text()
    lock_idx = text.index("if ! flock -n 9; then")
    call_idx = text.index('maintain_mirror_packs "${mirror_dir}" "${repo}"')
    assert lock_idx < call_idx, (
        "`maintain_mirror_packs` is called before the flock is acquired; it "
        "must run inside the lock this script already holds."
    )


def test_maintenance_is_threshold_gated_and_niced() -> None:
    """The steady state is a no-op, and the exceptional case yields to CI."""
    text = _script_text()
    body = text[
        text.index("maintain_mirror_packs() {") : text.index(
            "\n}\n", text.index("maintain_mirror_packs() {")
        )
    ]
    assert "MAINTENANCE_PACK_THRESHOLD" in body, (
        "the repack is unconditional; it must be gated on a pack-count "
        "threshold so a healthy mirror costs nothing."
    )
    assert "nice -n 19" in body and "ionice -c3" in body, (
        "the repack must run behind live CI for the host's CPU and IO."
    )
    assert '"${niced[@]}" git -C "${mirror_dir}" repack' in body, (
        "the repack must go through the resolved nicing wrapper, not call "
        "`git repack` directly -- otherwise it competes with live CI jobs "
        "for the one device that carries the whole fleet."
    )
    assert "command -v ionice" in body, (
        "`ionice` must be resolved rather than assumed: a missing binary "
        "would otherwise surface as a repack FAILURE, which reads as a "
        "corrupt mirror rather than a missing tool."
    )


def test_maintenance_moves_objects_and_never_expires_them() -> None:
    """`repack -a -d` + `prune-packed`, never `gc --prune`.

    The mirrors are served live by git-daemon. Moving reachable objects from
    loose files into a pack is safe for a concurrent `upload-pack`; expiring
    unreachable objects underneath one is not. This test is what stops a
    later "simplification" to `git gc`.
    """
    body_start = _script_text().index("maintain_mirror_packs() {")
    body = _script_text()[body_start : _script_text().index("\n}\n", body_start)]
    assert "repack -a -d" in body and "prune-packed" in body, (
        "expected `git repack -a -d` plus `git prune-packed` in "
        "`maintain_mirror_packs`."
    )
    assert "gc --prune" not in body and "git gc" not in body, (
        "`maintain_mirror_packs` must not run `git gc`: these mirrors are "
        "served live, and expiring objects under a concurrent `upload-pack` "
        "is the one way this maintenance could break a job. `repack -a -d` "
        "only moves reachable objects into a pack."
    )


# ---------------------------------------------------------------------------
# End to end against the real script
# ---------------------------------------------------------------------------


# Deliberately NOT skipped where `ionice` is absent. `maintain_mirror_packs`
# resolves the nicing tools and degrades without them, so this runs on macOS
# too -- and it has to. An earlier revision skipped here, which meant the one
# test that exercises the fixture end to end never ran locally and a broken
# fixture (4 packs against a threshold of 12) was first seen in CI.
def test_end_to_end_a_fragmented_mirror_is_consolidated(tmp_path: Path) -> None:
    """Drive a real mirror past the threshold and run the real function.

    The structural tests above all read the file. This one does not care how
    the script is spelled: it builds a mirror with many packs, calls the
    script's own `maintain_mirror_packs` against it, and asserts the mirror
    comes back consolidated and still holding every ref it had.

    Only the function-definition prefix is sourced, so the test never reaches
    the script's main loop and never tries to clone the nine production repos
    from github.com. Same split the sibling hook suite uses.
    """
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git("init", "--quiet", "--initial-branch=dev", ".", cwd=upstream)
    (upstream / "f.txt").write_text("0\n")
    _git("add", "-A", cwd=upstream)
    _git("commit", "--quiet", "-m", "c0", cwd=upstream)

    mirror_root = tmp_path / "mirrors"
    mirror_root.mkdir()
    repo = "onex_change_control"
    mirror = mirror_root / f"{repo}.git"
    _git("clone", "--quiet", "--mirror", str(upstream), str(mirror))
    _git("-C", str(mirror), "config", "remote.origin.fetch", "+refs/*:refs/*")

    # Fragment it the way the production mirror fragmented: many small
    # fetches, each landing its own pack.
    #
    # `fetch.unpackLimit 1` is what makes this deterministic, and it is the
    # same setting the fix adds in production: without it git's default of
    # 100 unpacks each small fetch into loose objects and the pack count
    # barely moves. An earlier revision of this fixture called `git repack`
    # once per iteration instead, and CI produced 4 packs against a threshold
    # of 12 -- repack CONSOLIDATES, so it was undoing the fragmentation it
    # was there to create.
    _git("-C", str(mirror), "config", "fetch.unpackLimit", "1")
    _git("-C", str(mirror), "config", "gc.auto", "0")
    for i in range(1, 20):
        (upstream / "f.txt").write_text(f"{i}\n")
        _git("add", "-A", cwd=upstream)
        _git("commit", "--quiet", "-m", f"c{i}", cwd=upstream)
        _git(
            "-C", str(mirror), "fetch", "--quiet", "--prune", "origin", "+refs/*:refs/*"
        )

    packs_before = len(list((mirror / "objects" / "pack").glob("*.pack")))
    refs_before = _git(
        "-C", str(mirror), "for-each-ref", "--format=%(refname) %(objectname)"
    )
    assert packs_before >= 12, (
        f"fixture did not fragment the mirror enough ({packs_before} packs); "
        "the threshold branch would not be exercised"
    )

    driver = tmp_path / "driver.sh"
    driver.write_text(_functions_only_script() + '\nmaintain_mirror_packs "$1" "$2"\n')
    result = subprocess.run(
        ["bash", str(driver), str(mirror), repo],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    packs_after = len(list((mirror / "objects" / "pack").glob("*.pack")))
    refs_after = _git(
        "-C", str(mirror), "for-each-ref", "--format=%(refname) %(objectname)"
    )

    assert result.returncode == 0, result.stderr
    assert packs_after < packs_before, (
        f"the mirror still has {packs_after} packs (was {packs_before}); the "
        "threshold-gated repack did not run.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert refs_after == refs_before, (
        "maintenance changed the mirror's refs. `repack -a -d` plus "
        "`prune-packed` must only move objects, never drop history."
    )
    assert "repacked" in result.stdout, (
        f"expected a repack line in the output:\n{result.stdout}"
    )
    # The mirror must still be a valid, fully connected object store.
    subprocess.run(
        ["git", "-C", str(mirror), "fsck", "--connectivity-only"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        env=scrub_git_location_env(os.environ),
    )


def test_positive_control_a_tidy_mirror_is_left_alone(tmp_path: Path) -> None:
    """Below the threshold the maintenance is a no-op.

    Without this the test above passes on a function that repacks
    unconditionally, which is the IO spike the original `gc.auto 0` comment
    was right to avoid.
    """
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git("init", "--quiet", "--initial-branch=dev", ".", cwd=upstream)
    (upstream / "f.txt").write_text("0\n")
    _git("add", "-A", cwd=upstream)
    _git("commit", "--quiet", "-m", "c0", cwd=upstream)

    mirror = tmp_path / "onex_change_control.git"
    _git("clone", "--quiet", "--mirror", str(upstream), str(mirror))
    packs_before = len(list((mirror / "objects" / "pack").glob("*.pack")))

    driver = tmp_path / "driver.sh"
    driver.write_text(_functions_only_script() + '\nmaintain_mirror_packs "$1" "$2"\n')
    result = subprocess.run(
        ["bash", str(driver), str(mirror), "onex_change_control"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "repacking" not in result.stdout, (
        f"a mirror below the pack threshold was repacked anyway:\n{result.stdout}"
    )
    assert len(list((mirror / "objects" / "pack").glob("*.pack"))) == packs_before
