# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17139 — "comment-only" is a claim, and a bad stripper would launder it.

``prove_migration_revision_equivalence.py`` admits a rewritten migration onto a
lane on one basis: the recorded revision and the file on disk are byte-identical
once SQL comments are removed, so they are the same program. That claim is only
as strong as the comment stripper, and the ways a naive stripper gets it wrong
are all constructs migrations really contain -- a ``--`` inside a string literal,
a ``/*`` inside a dollar-quoted body, an apostrophe inside ``$$``.

Every case below is one of those. The ones that matter most are the negatives:
a stripper that removed a ``--`` inside a string literal would call two genuinely
different programs equivalent and hand a lane a false proof.

Ticket: OMN-17139
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "migrations" / "prove_migration_revision_equivalence.py"
)


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("prove_equivalence", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["prove_equivalence"] = module
    spec.loader.exec_module(module)
    return module


proof = _load()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("SELECT 1; -- trailing\n", "SELECT 1;"),
        ("/* leading */ SELECT 1;\n", "SELECT 1;"),
        ("/* outer /* inner */ still outer */ SELECT 1;\n", "SELECT 1;"),
        ("SELECT 1;\n\n\n   SELECT 2;\n", "SELECT 1; SELECT 2;"),
    ],
)
def test_comments_and_whitespace_are_removed(source: str, expected: str) -> None:
    assert proof.executable_text(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        # A -- inside a string literal is DATA. Removing it changes the program.
        "SELECT 'a -- b';",
        # So is a /* inside one.
        "SELECT 'a /* b */ c';",
        # And anything inside a dollar-quoted body.
        "SELECT $$ -- not a comment $$;",
        "SELECT $tag$ /* nor this */ $tag$;",
        # An escaped quote must not end the literal early.
        "SELECT 'it''s -- fine';",
        # A quoted identifier is not a string, and is equally untouchable.
        'SELECT "col -- name" FROM t;',
    ],
)
def test_string_and_quoted_content_is_never_treated_as_a_comment(source: str) -> None:
    assert proof.executable_text(source) == source


def test_a_comment_only_difference_normalises_identically() -> None:
    before = "-- revision 1\nCREATE TABLE t (id INT);\n"
    after = (
        "-- revision 2, rewritten header\n/* and a block */\nCREATE TABLE t (id INT);\n"
    )
    assert proof.executable_text(before) == proof.executable_text(after)


def test_a_difference_hidden_inside_a_string_is_not_comment_only() -> None:
    """The failure a naive stripper produces, asserted as a difference."""
    before = "INSERT INTO t (note) VALUES ('keep -- this');"
    after = "INSERT INTO t (note) VALUES ('keep');"
    assert proof.executable_text(before) != proof.executable_text(after)


@pytest.mark.parametrize(
    "source",
    [
        "SELECT 1; /* never closed",
        "SELECT 'never closed",
        "SELECT $$ never closed",
        'SELECT "never closed',
    ],
)
def test_unresolvable_input_refuses_rather_than_guessing(source: str) -> None:
    with pytest.raises(proof.ProofError):
        proof.executable_text(source)


def test_the_shipped_declaration_is_reproducible() -> None:
    """Re-run the live proof this ticket committed, from the repo as it stands.

    The adoption row in ``_ledger/verified-canonical-adoptions.tsv`` is only as
    good as the claim behind it. This re-derives that claim from git history and
    the working tree, so a later edit to the migration silently invalidating the
    declaration fails here instead of on a lane.
    """
    rows = [
        line.split("\t")
        for line in (
            REPO_ROOT
            / "docker"
            / "migrations"
            / "forward"
            / "_ledger"
            / "verified-canonical-adoptions.tsv"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert rows, "the canonical adoption relation is empty; this proof is vacuous"

    # Half of this check needs nothing but the working tree, so it runs
    # everywhere: the declared manifest_checksum must still be what the manifest
    # declares AND what the file on disk hashes to. That is exactly the staleness
    # the validator and bootstrap.sql both fail closed on.
    forward = proof.REPO_ROOT / "docker" / "migrations" / "forward"
    for row in rows:
        version, manifest_checksum = row[0], row[2]
        artifact_path, declared = proof.resolve_declaration(version)
        on_disk = proof.sha256_bytes((forward / artifact_path).read_bytes())
        assert manifest_checksum == declared == on_disk, (
            f"{version}: the adoption declares {manifest_checksum}, the manifest "
            f"declares {declared}, the file hashes to {on_disk}. The migration "
            "changed after the proof was earned; re-prove it."
        )

    # The other half re-derives the equivalence from git history, and a checkout
    # can be genuinely unable to answer it: this repository is a shallow clone,
    # and an exported tree may carry no git at all, so the commit that held the
    # recorded bytes need not be present. That is a property of the checkout, not
    # of the declaration.
    #
    # The skip is therefore narrow: ONLY a failure to LOCATE the recorded
    # revision. A located revision that no longer proves equivalent returns a
    # verdict rather than raising, so a broken declaration still fails here.
    for row in rows:
        version, source_checksum = row[0], row[1]
        try:
            result = proof.prove(version, source_checksum)
        except proof.ProofError as exc:
            if "history hashes to" in str(exc) or "git " in str(exc):
                pytest.skip(
                    "this checkout cannot resolve the recorded revision "
                    f"({exc}); the manifest/on-disk half above ran"
                )
            raise
        assert result.verdict == proof.VERDICT_COMMENT_ONLY_EQUIVALENT, (
            f"{version} is declared in verified-canonical-adoptions.tsv but its "
            f"proof no longer holds: {result.reason}"
        )
