# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17975 incident replay: the refusal that made the evidence closer flip nothing.

The captured artifact is the VERBATIM stdout of the evidence-autoclose sweep's own
diagnostic step, run 33997616996 (dispatch dry-run, `diagnose_tickets=OMN-17907`),
2026-09-05T23:05:36Z. It is not a reconstruction: it is what the runner printed.

What it records is a `false_red` — the merged product work was real, its behaviour
proof was correctly minted and correctly diff-derived, and the verifier refused it
UNEXECUTED anyway, because the tree the sweep handed it had a detached HEAD:

    [skipped] dod-occ-diff-derived-behavior-proof: PRODUCT_CLONE_NOT_FRESH: ...
    upstream <none> ... no remote-tracking upstream is configured for HEAD, so
    freshness cannot be established (fatal: HEAD does not point to a branch).
    The command was NOT executed.

That refusal is CORRECT — a verdict from a tree whose contents cannot be placed
relative to the work under adjudication is not evidence about it. The defect is
the tree, and `scripts/ci/normalize_product_clone.sh` is what removes it.

The replay does three things in order, and the order is the point:

1. Reads the incident's own numbers out of the capture rather than restating them.
2. Reconstructs the tree state the capture describes and proves, against LIVE git,
   that it reproduces the exact error string the capture recorded. Without this
   step the fixture would only be a quotation; with it, the shape is verified to
   be the failing one.
3. Drives the REAL script over that reconstructed tree and requires the freshness
   predicate `EvidenceCollector` reads to become satisfiable.

The `discriminator` (registry-required for a `false_red`) lives in
`test_autoclose_product_clone_freshness_omn17975.py` and proves the same script
still refuses a tree it cannot make fresh — an accept-only proof cannot tell a
working normaliser from one that is stuck open.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "normalize_product_clone.sh"
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn17975"
    / "diagnose-omn17907-run33997616996.log.captured"
)

pytestmark = pytest.mark.unit


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


@pytest.fixture(scope="module")
def captured() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_the_capture_records_a_behaviour_proof_refused_unexecuted(
    captured: str,
) -> None:
    """The incident, asserted from the bytes rather than described in prose."""
    assert "dod-occ-diff-derived-behavior-proof" in captured
    assert "PRODUCT_CLONE_NOT_FRESH" in captured
    assert "no remote-tracking upstream is configured for HEAD" in captured
    assert "fatal: HEAD does not point to a branch" in captured
    assert "The command was NOT executed" in captured

    # The whole reason this refusal matters: it lands on the one behaviour-proving
    # check the candidate has, so the flip predicate's OMN-15911 conjunct can never
    # be satisfied no matter how much real work merged.
    verdict = re.search(
        r"status=(\w+) total=(\d+) verified=(\d+) failed=(\d+) skipped=(\d+)"
        r" superseded=(\d+) non_probative=(\d+) behavior_proving=(\d+)",
        captured,
    )
    assert verdict is not None, "the capture no longer carries the verdict counters"
    status, total, verified, failed, skipped, _sup, non_prob, behaviour = (
        verdict.groups()
    )
    assert status == "skipped"
    assert behaviour == "0"
    assert failed == "0", "nothing FAILED — the work is fine, it was never executed"
    # `verified + non_probative` falls exactly one short of the denominator, and the
    # missing unit is the refused behaviour proof.
    assert int(verified) + int(non_prob) == int(total) - 1
    assert int(skipped) >= 1


def test_the_reconstructed_tree_reproduces_the_captured_git_error(
    captured: str, tmp_path: Path
) -> None:
    """Prove the shape under test is the failing one, against live git.

    The capture names the condition (`upstream <none>`, detached HEAD). This
    rebuilds that condition and requires live git to emit the SAME error string
    the runner recorded — so the replay below is driving the real failure and not
    a plausible-looking neighbour of it.
    """
    assert "upstream <none>" in captured

    clone = tmp_path / "detached"
    clone.mkdir()
    _git(clone, "init", "--quiet", "--initial-branch", "dev")
    _git(clone, "config", "user.email", "test@example.invalid")
    _git(clone, "config", "user.name", "Test")
    (clone / "a.txt").write_text("one\n", encoding="utf-8")
    _git(clone, "add", "a.txt")
    _git(clone, "commit", "--quiet", "-m", "one")
    _git(clone, "checkout", "--quiet", "--detach", "HEAD")

    probe = _git(
        clone, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
    )

    assert probe.returncode != 0
    assert "HEAD does not point to a branch" in probe.stderr, (
        "live git no longer produces the error the capture recorded; the "
        "reconstruction has drifted from the incident and the replay below would "
        "be proving something else"
    )


def test_the_real_normaliser_makes_the_captured_condition_go_away(
    captured: str, tmp_path: Path
) -> None:
    """The replay proper: drive the REAL script over the real failing shape.

    `accept` is the verdict this case pins, because the direction of the
    regression is `false_red` — good work was refused. What must become true is
    exactly the three-step predicate `EvidenceCollector` reads before it will
    execute a behaviour check.
    """
    assert "PRODUCT_CLONE_NOT_FRESH" in captured  # R4: the case consumes the artifact

    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "--quiet", "--initial-branch", "dev")
    _git(seed, "config", "user.email", "test@example.invalid")
    _git(seed, "config", "user.name", "Test")
    (seed / "a.txt").write_text("one\n", encoding="utf-8")
    _git(seed, "add", "a.txt")
    _git(seed, "commit", "--quiet", "-m", "one")
    _git(seed, "clone", "--quiet", "--bare", str(seed), str(origin))

    clone = tmp_path / "materialised"
    subprocess.run(
        ["git", "clone", "--quiet", str(origin), str(clone)],
        check=True,
        capture_output=True,
    )
    # The exact shape the sweep produced: a copy of an `actions/checkout` tree,
    # pinned to a resolved SHA, therefore detached.
    head = _git(clone, "rev-parse", "HEAD").stdout.strip()
    _git(clone, "checkout", "--quiet", "--detach", head)
    assert (
        _git(
            clone, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
        ).returncode
        != 0
    )

    proc = subprocess.run(
        [str(SCRIPT), "--no-move", str(clone), str(origin), "dev"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "FRESH" in proc.stdout

    # Step 1 of the collector's predicate: @{upstream} must resolve.
    upstream = _git(
        clone, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"
    )
    assert upstream.returncode == 0
    assert "/" in upstream.stdout.strip()
    # Step 3: HEAD must not be behind it.
    behind = _git(clone, "rev-list", "--count", f"HEAD..{upstream.stdout.strip()}")
    assert behind.stdout.strip() == "0"
    # And the tree itself was not moved — the property the drift guard depends on.
    assert _git(clone, "rev-parse", "HEAD").stdout.strip() == head
