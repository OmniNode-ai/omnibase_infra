# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17975: the autoclose sweep must hand dod_verify a clone whose freshness is establishable.

`EvidenceCollector._compute_product_clone_resolution` (omnimarket) establishes a
product clone's freshness by reading ``git rev-parse --abbrev-ref
--symbolic-full-name @{upstream}``. On a DETACHED HEAD that command fails with
``fatal: HEAD does not point to a branch``, the resolution is ``UNKNOWN``, and
every behaviour check whose ``cwd`` names that tree is refused UNEXECUTED as
``PRODUCT_CLONE_NOT_FRESH``.

Every clone the evidence-autoclose sweep provided was detached — the reused
`actions/checkout` trees are pinned to a resolved SHA by construction — so the
one behaviour-proving check on every candidate was refused, and
``behavior_proving_count`` was 0 for the whole corpus.

Measured on the runner, dispatch dry-run 33997616996, 2026-09-05T23:05:36Z::

    [skipped] dod-occ-diff-derived-behavior-proof: PRODUCT_CLONE_NOT_FRESH:
    the check's cwd names /home/runner/work/omnibase_infra/omnimarket, whose
    freshness is unknown (HEAD a138c1e4c79a..., upstream <none>, behind None;
    no remote-tracking upstream is configured for HEAD ...). The command was
    NOT executed.

These tests exercise the real script against real git repositories. The first
one is the RED case: it asserts the defect exists in a detached clone, so a
regression that reintroduces it is caught by the same file that proves the fix.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "normalize_product_clone.sh"
)

pytestmark = pytest.mark.unit


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _upstream_resolves(clone: Path) -> bool:
    """The exact predicate the collector uses to decide freshness is knowable."""
    proc = _git(
        clone, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
    )
    return proc.returncode == 0 and "/" in proc.stdout.strip()


def _behind_count(clone: Path) -> int:
    upstream = _git(
        clone, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
    ).stdout.strip()
    out = _git(clone, "rev-list", "--count", f"HEAD..{upstream}").stdout.strip()
    return int(out)


@pytest.fixture
def origin(tmp_path: Path) -> Path:
    """A real upstream repository with a `dev` branch and two commits."""
    remote = tmp_path / "origin.git"
    work = tmp_path / "seed"
    work.mkdir()
    _git(work, "init", "--quiet", "--initial-branch", "dev")
    _git(work, "config", "user.email", "test@example.invalid")
    _git(work, "config", "user.name", "Test")
    (work / "a.txt").write_text("one\n", encoding="utf-8")
    _git(work, "add", "a.txt")
    _git(work, "commit", "--quiet", "-m", "one")
    (work / "b.txt").write_text("two\n", encoding="utf-8")
    _git(work, "add", "b.txt")
    _git(work, "commit", "--quiet", "-m", "two")
    _git(work, "clone", "--quiet", "--bare", str(work), str(remote))
    return remote


def _detached_clone(origin: Path, dest: Path) -> Path:
    """Reproduce the shape the sweep produced: a checkout pinned to a SHA."""
    subprocess.run(
        ["git", "clone", "--quiet", str(origin), str(dest)],
        check=True,
        capture_output=True,
    )
    head = _git(dest, "rev-parse", "HEAD").stdout.strip()
    _git(dest, "checkout", "--quiet", "--detach", head)
    return dest


def test_detached_clone_has_no_establishable_freshness(
    origin: Path, tmp_path: Path
) -> None:
    """RED case: this is the defect, asserted directly rather than described.

    A clone in this shape is what the sweep handed dod_verify on every run, and
    it is why every behaviour check was refused unexecuted.
    """
    clone = _detached_clone(origin, tmp_path / "detached")

    assert not _upstream_resolves(clone), (
        "a detached clone must not resolve @{upstream} — if this passes, the "
        "fixture no longer reproduces the shape the sweep produced and the "
        "GREEN case below proves nothing."
    )
    proc = _git(
        clone, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
    )
    assert "HEAD does not point to a branch" in proc.stderr


def test_normalise_makes_freshness_establishable_and_current(
    origin: Path, tmp_path: Path
) -> None:
    """GREEN case: after normalisation the collector's predicate is satisfiable."""
    clone = _detached_clone(origin, tmp_path / "detached")

    proc = subprocess.run(
        [str(SCRIPT), str(clone), str(origin), "dev"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert _upstream_resolves(clone), proc.stdout + proc.stderr
    assert _behind_count(clone) == 0
    assert _git(clone, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "dev"
    assert "FRESH" in proc.stdout


def test_normalise_fast_forwards_a_clone_that_is_behind(
    origin: Path, tmp_path: Path
) -> None:
    """A clone pinned to an older SHA is brought to the upstream tip, not just labelled."""
    clone = _detached_clone(origin, tmp_path / "behind")
    first = _git(clone, "rev-list", "--max-parents=0", "HEAD").stdout.strip()
    _git(clone, "checkout", "--quiet", "--detach", first)
    assert _git(clone, "rev-parse", "HEAD").stdout.strip() == first

    proc = subprocess.run(
        [str(SCRIPT), str(clone), str(origin), "dev"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert _behind_count(clone) == 0
    assert _git(clone, "rev-parse", "HEAD").stdout.strip() != first


def test_no_move_leaves_a_behind_tree_untouched_and_refuses(
    origin: Path, tmp_path: Path
) -> None:
    """AC2. The job's own checkout must never be moved out from under the venv.

    Rewinding or fast-forwarding ``github.workspace`` mid-job would change the
    source the gate venv was synced against, and the purity assertion that runs
    after the sweep would be measuring a different tree than the one it
    installed. So ``--no-move`` refuses instead, the tree is left exactly as it
    was, and the resulting PRODUCT_CLONE_NOT_FRESH refusal stands honestly.
    """
    clone = _detached_clone(origin, tmp_path / "gate")
    first = _git(clone, "rev-list", "--max-parents=0", "HEAD").stdout.strip()
    _git(clone, "checkout", "--quiet", "--detach", first)

    proc = subprocess.run(
        [str(SCRIPT), "--no-move", str(clone), str(origin), "dev"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 3, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert _git(clone, "rev-parse", "HEAD").stdout.strip() == first
    assert "BEHIND" in proc.stdout


def test_no_move_still_names_a_tree_already_at_the_tip(
    origin: Path, tmp_path: Path
) -> None:
    """``--no-move`` is a refusal to MOVE, not a refusal to establish freshness."""
    clone = _detached_clone(origin, tmp_path / "gate-current")
    head = _git(clone, "rev-parse", "HEAD").stdout.strip()

    proc = subprocess.run(
        [str(SCRIPT), "--no-move", str(clone), str(origin), "dev"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert _git(clone, "rev-parse", "HEAD").stdout.strip() == head
    assert _upstream_resolves(clone)
    assert _behind_count(clone) == 0
    assert "FRESH" in proc.stdout


def test_a_directory_that_is_not_a_repository_is_named_not_crashed(
    tmp_path: Path,
) -> None:
    """Fail-closed with a named reason; never a traceback the job log swallows."""
    plain = tmp_path / "not-a-repo"
    plain.mkdir()

    proc = subprocess.run(
        [str(SCRIPT), str(plain), "https://example.invalid/x.git", "dev"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 1
    assert "NOT_A_GIT_REPOSITORY" in (proc.stdout + proc.stderr)


# --------------------------------------------------------------------------
# Wiring. The script above is only worth anything if the sweep actually calls
# it: a detection tool that is not wired as the gate is advisory and gets
# ignored (CLAUDE.md rule 5). These assert the workflow's own text, which is
# weaker evidence than the git fixtures above and is why both exist.
# --------------------------------------------------------------------------

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "evidence-autoclose-sweep.yml"
)


def test_every_materialised_dest_is_normalised_before_the_sweep() -> None:
    """Both destination classes call the normaliser, and both use --no-move."""
    text = WORKFLOW.read_text(encoding="utf-8")

    calls = text.count("scripts/ci/normalize_product_clone.sh")
    assert calls >= 2, (
        "expected the normaliser to be called for BOTH the gate checkout "
        "(ALREADY_PRESENT_GATE_CHECKOUT) and every materialised dest; found "
        f"{calls} reference(s)"
    )
    assert text.count('normalize_product_clone.sh" --no-move') == calls, (
        "every call must pass --no-move: the OMN-14060 drift guard requires "
        "${OMNI_HOME}/omnimarket HEAD to equal the commit co-installed into the "
        "dispatch venv, and the gate venv is synced against github.workspace. "
        "Moving either tree would hard-fail the job on a mid-run `dev` advance."
    )


def test_the_sweep_names_its_own_per_check_ceiling() -> None:
    """AC4. OMN-17795 made the bound overridable; nothing ever set the override."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["evidence-autoclose-sweep"]["steps"]
    sweep = next(s for s in steps if s.get("name") == "Run evidence autoclose sweep")

    ceiling = sweep["env"].get("DOD_VERIFY_CHECK_TIMEOUT_S")
    assert ceiling is not None, (
        "the sweep must name its per-check ceiling rather than inheriting the "
        "30s default authored for a developer laptop"
    )
    assert int(ceiling) > 30
    # A raise past this needs `timeout-minutes` raised in the same change: a job
    # killed at its own timeout reports nothing at all, which is a strictly worse
    # signal than a named CHECK_BUDGET_EXCEEDED.
    job_timeout = workflow["jobs"]["evidence-autoclose-sweep"]["timeout-minutes"]
    assert int(ceiling) * 50 <= job_timeout * 60 * 3, (
        f"a {ceiling}s per-check ceiling against the {job_timeout}-minute job "
        "timeout and a 50-companion enumeration cap is not bounded; raise the "
        "job timeout in the same change or lower the ceiling"
    )


def test_the_default_ceiling_in_the_collector_is_not_moved() -> None:
    """AC4, negative half. The override is per-job; the default is org-wide.

    OMN-17795 left `_DEFAULT_CHECK_TIMEOUT_S = 30` alone on purpose — raising it
    would shift the timing envelope of every existing verdict on every host at
    once. This ticket names a ceiling for THIS job and must not reach into that.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "_DEFAULT_CHECK_TIMEOUT_S" not in text
    for laundering_flag in (
        "DOD_VERIFY_ALLOW_STALE_PRODUCT_CLONE",
        "DOD_VERIFY_ALLOW_STALE_OCC_REF",
    ):
        assert f"{laundering_flag}:" not in text, (
            f"{laundering_flag} marks every verdict it touches un-attributable "
            "to a verified-fresh tree. The fix is to make the tree fresh, never "
            "to tell the verifier to stop checking."
        )
