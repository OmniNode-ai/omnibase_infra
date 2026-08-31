# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17320 incident replay (OMN-15547 convention).

The incident: OMN-17288 scrubbed a live tenant slug out of five files in this repo
(omnibase_infra#3062, merge ``3f10ee5e``). Three hours later omnimarket#2239
reintroduced the same slug there, and the rebase carried it onto omnimarket#2241 --
the PR whose acceptance criterion was "zero grep hits" -- with **every enforced gate
green**. There was no guard. That absence is the false-green being replaced.

The artifact is ``docker/migrations/forward/_ledger/migration-supersessions.tsv`` as
it stood on this repo's ``dev`` at ``3f10ee5e^``, immediately before the scrub -- one
of the five files the OMN-17288 census enumerated.

ONE DOCUMENTED REDACTION, and why it is not a hand-typed approximation
---------------------------------------------------------------------
The 11-byte slug on line 7 is replaced by an 11-byte stand-in. Every other byte of
the 8455 is verbatim, and the length is preserved so every offset in the file is
unchanged. The pre-redaction sha256 is recorded in the registry entry, so the claim
is checkable by anyone who re-fetches the git object.

The redaction is unavoidable: this repo is PUBLIC, and committing the real value as
a test fixture is precisely the disclosure the guard under test exists to prevent.
An unredacted fixture would fail the very gate it is proving. This is not a novel
liberty -- OMN-17288 set the same precedent in this same registry, redacting a byte
range out of ``tests/fixtures/omn16906/commit-7090f386f.gh-api.json.captured`` and
moving its sha256 pin with the original hash documented.

What is genuinely replayed is what matters: the guard is driven over the REAL
surrounding bytes -- the real TSV field structure, the real prose, the real token
boundaries at the real byte offset (line 7, column 379) -- not over a toy line an
author imagined. A synthetic fixture could only ever exhibit failures its author had
already thought of, which is the whole complaint OMN-15547 was filed about.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
GATE = REPO_ROOT / "scripts" / "validation" / "check_exposed_identifiers.py"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "omn17320"
ARTIFACT = FIXTURE_DIR / "migration-supersessions.tsv.captured"
REPLAY_DENYLIST = FIXTURE_DIR / "replay-denylist.json"
SHIPPED_DENYLIST = (
    REPO_ROOT / "scripts" / "validation" / "exposed_identifiers_denylist.json"
)

# Byte coordinates of the slug in the captured artifact, preserved by the
# length-neutral redaction. If either moves, the fixture is no longer the artifact.
SLUG_LINE = 7
SLUG_COLUMN = 379

pytestmark = pytest.mark.unit


def _run(denylist: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(GATE),
            "--denylist",
            str(denylist),
            "--root",
            str(FIXTURE_DIR),
            str(ARTIFACT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_real_guard_rejects_the_captured_artifact() -> None:
    """R5, false_green direction: the guard must say REJECT on the real bad input.

    Before this ticket there was no guard at all, so this content -- and the
    omnimarket reintroduction three hours later -- passed every gate in the org.
    """
    result = _run(REPLAY_DENYLIST)
    assert result.returncode == 1, (
        "the guard ACCEPTED the pre-scrub artifact; the replay is vacuous\n"
        + result.stdout
        + result.stderr
    )
    assert f"{ARTIFACT.name}:{SLUG_LINE}:{SLUG_COLUMN}:" in result.stdout, (
        "the finding did not land at the slug's real byte offset -- the fixture is "
        f"no longer positioned like the artifact\n{result.stdout}"
    )
    assert "match_len=11" in result.stdout, result.stdout


def test_the_guard_does_not_print_the_value_it_found() -> None:
    """A gate that echoes the identifier into CI logs has moved the leak, not closed it."""
    result = _run(REPLAY_DENYLIST)
    standin = "t-replayfix"
    assert standin not in result.stdout + result.stderr, result.stdout


def test_the_shipped_denylist_does_not_fire_on_this_artifact() -> None:
    """Accept control: a guard that rejects everything proves nothing by rejecting."""
    result = _run(SHIPPED_DENYLIST)
    assert result.returncode == 0, (
        "the shipped denylist flags the redacted artifact -- either the redaction "
        "failed or an entry is over-broad\n" + result.stdout
    )


def test_the_artifact_carries_no_real_denylisted_identifier() -> None:
    """The fixture itself must satisfy the rule the guard enforces."""
    result = subprocess.run(
        [sys.executable, str(GATE), "--mode", "blocking", "--scope", "all"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
