#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Prove that the revision a lane recorded and the file on disk are the same program.

OMN-17139.

The question
------------
``run-forward-migrations.sh`` records a ``content_sha256`` for every node
migration it applies, into ``platform_catalog.schema_migrations``. When an
already-applied migration is later edited in place, the recorded hash and the
file's hash disagree and ``migration_is_applied()`` exits 1 -- permanently, on
every lane that applied the old bytes. The only honest way out is a per-version
declaration in ``_ledger/verified-canonical-adoptions.tsv`` asserting that the
recorded revision and the current file produce the same schema.

Why this tool exists alongside the replay prover
------------------------------------------------
``verify_migration_checksum_adoption.py`` answers that question by *execution*:
replay the file into a scratch database, derive the object surface it owns, and
diff that surface against the live database. That is the right proof when the
two revisions are genuinely different SQL.

It cannot reach every migration. A node migration whose schema is created by a
flat migration in a *different* database -- ``omninode_internal``, created by
``098_create_omninode_internal_schema.sql`` via ``\\connect`` -- fails its own
first precondition on a scratch server, and the tool correctly returns
``unreachable`` rather than guessing. ``node_projection_work_events:0001`` is
exactly that shape.

For one class of rewrite there is a stronger and cheaper proof available, and it
does not need a database at all: if the two revisions are byte-identical once SQL
comments are removed, they are the *same program*. Not "similar", not
"equivalent in the cases we thought to check" -- identical token-for-token, so
there is no execution in which they could differ. That is a deterministic
property of the two artifacts, re-checkable by anyone, forever.

What it does
------------
1. Resolve ``--version`` to its artifact path and declared checksum in
   ``_ledger/application-migrations.tsv``, and confirm the file on disk still
   hashes to that declared checksum. A stale manifest is refused, not worked
   around.
2. Find the recorded revision **in git history**: every commit that touched the
   path is searched for a blob whose sha256 equals ``--recorded-checksum``. If no
   revision of this file ever had those bytes, there is nothing to compare and
   the run refuses -- the recorded hash is then a fact about some other artifact,
   or about bytes that were never committed.
3. Strip SQL comments from both revisions with a PostgreSQL-correct lexer
   (``--`` to end of line, nested ``/* */``, single-quoted strings with ``''``
   and ``E'\\'`` escapes, double-quoted identifiers, dollar-quoted bodies), then
   collapse whitespace runs, and compare.
4. Write a receipt naming both revisions, both raw hashes and both normalised
   hashes. ``--emit-adoption`` writes the declaration row, carrying the receipt's
   own sha256 so the row is always traceable to the run that earned it.

Everything that could be ambiguous fails closed: an unterminated string, an
unterminated block comment, an empty executable body, or any normalised
difference at all yields a refusal and exit 1. A comment-only rewrite is a narrow
claim and this tool will only ever make that narrow claim.

Usage::

    python scripts/migrations/prove_migration_revision_equivalence.py \\
        --version node:node_projection_work_events:0001_create_work_events.sql \\
        --recorded-checksum cba8013e...6664 \\
        --lane dev \\
        --receipt-out docker/migrations/forward/_ledger/receipts/omn17139.json \\
        --emit-adoption --ticket OMN-17139
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

TOOL_VERSION = "1"
TICKET = "OMN-17139"

REPO_ROOT = Path(__file__).resolve().parents[2]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
LEDGER_DIR = FORWARD_DIR / "_ledger"
APPLICATION_MANIFEST = LEDGER_DIR / "application-migrations.tsv"
VERIFIED_CANONICAL_ADOPTIONS = LEDGER_DIR / "verified-canonical-adoptions.tsv"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TICKET_RE = re.compile(r"^OMN-[0-9]+$")
# A dollar-quote tag is an identifier: a letter, underscore or non-ASCII letter
# first, then letters, digits, underscores or non-ASCII. `$$` (empty tag) is the
# common spelling and is matched explicitly.
_DOLLAR_TAG_RE = re.compile(
    r"\$(?:[A-Za-z_\u0080-\uffff][A-Za-z0-9_\u0080-\uffff]*)?\$"
)

VERDICT_COMMENT_ONLY_EQUIVALENT = "comment_only_equivalent"
VERDICT_DIVERGENT = "divergent"
VERDICT_UNREACHABLE = "unreachable"

ADOPTION_COLUMNS = (
    "version",
    "source_checksum",
    "manifest_checksum",
    "ticket",
    "receipt_sha256",
    "verified_at",
)


class ProofError(RuntimeError):
    """A condition under which no equivalence claim may be made."""


# ---------------------------------------------------------------------------
# SQL comment stripping
# ---------------------------------------------------------------------------


def strip_sql_comments(text: str) -> str:
    """Return ``text`` with every SQL comment removed, or raise.

    Hand-written rather than delegated because the delegation targets available
    here (a regex, a generic C-family lexer) are exactly the ones that get this
    wrong on the constructs migrations actually contain: a ``--`` inside a string
    literal, a ``/*`` inside a dollar-quoted body, an apostrophe inside a
    ``$$``-quoted body, a nested block comment.

    Every one of those is a state in the scanner below, and anything the scanner
    cannot resolve -- an unterminated literal, an unterminated comment -- raises
    instead of returning a best guess. A stripper that guesses would launder a
    real SQL difference into a "comment-only" claim, which is the one failure
    this whole proof exists to make impossible.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        # Line comment: -- ... EOL (the newline is kept as a separator).
        if ch == "-" and text.startswith("--", i):
            end = text.find("\n", i)
            if end == -1:
                break
            out.append("\n")
            i = end + 1
            continue

        # Block comment: /* ... */, nestable per the SQL standard and Postgres.
        if ch == "/" and text.startswith("/*", i):
            depth = 1
            i += 2
            while i < n and depth > 0:
                if text.startswith("/*", i):
                    depth += 1
                    i += 2
                elif text.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth != 0:
                raise ProofError("unterminated block comment")
            out.append(" ")
            continue

        # Dollar-quoted string: $tag$ ... $tag$ (body is opaque, comments inside
        # it are DATA, not comments).
        if ch == "$":
            match = _DOLLAR_TAG_RE.match(text, i)
            if match is not None:
                tag = match.group(0)
                end = text.find(tag, match.end())
                if end == -1:
                    raise ProofError(f"unterminated dollar-quoted string {tag!r}")
                out.append(text[i : end + len(tag)])
                i = end + len(tag)
                continue

        # Single-quoted string: '' is an embedded quote; E'' also honours
        # backslash escapes, so a preceding E/e is consumed with the literal.
        if ch == "'":
            j = i + 1
            escaped = i > 0 and text[i - 1] in "Ee"
            while j < n:
                if escaped and text[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if text[j] == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                raise ProofError("unterminated single-quoted string")
            out.append(text[i : j + 1])
            i = j + 1
            continue

        # Double-quoted identifier: "" is an embedded quote.
        if ch == '"':
            j = i + 1
            while j < n:
                if text[j] == '"':
                    if j + 1 < n and text[j + 1] == '"':
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                raise ProofError("unterminated quoted identifier")
            out.append(text[i : j + 1])
            i = j + 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def executable_text(text: str) -> str:
    """Comment-free SQL with whitespace runs collapsed, for comparison."""
    stripped = strip_sql_comments(text)
    return re.sub(r"\s+", " ", stripped).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# git archaeology
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ProofError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _git_bytes(*args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ProofError(f"git {' '.join(args)} failed")
    return result.stdout


@dataclass(frozen=True)
class RecordedRevision:
    commit: str
    blob: str
    text: str


def find_recorded_revision(repo_path: str, recorded_checksum: str) -> RecordedRevision:
    """The commit whose version of ``repo_path`` hashes to ``recorded_checksum``.

    Searched across every ref rather than the current branch: the revision a lane
    applied may live on a merged branch whose commits are reachable only through
    a merge, and "I could not find it on dev" is not evidence that it never
    existed.
    """
    commits = [
        line
        for line in _git("rev-list", "--all", "--", repo_path).splitlines()
        if line.strip()
    ]
    if not commits:
        raise ProofError(f"no commit in this repository has ever touched {repo_path}")
    for commit in commits:
        try:
            blob = _git_bytes("show", f"{commit}:{repo_path}")
        except ProofError:
            continue
        if sha256_bytes(blob) == recorded_checksum:
            blob_id = _git("rev-parse", f"{commit}:{repo_path}").strip()
            return RecordedRevision(
                commit=commit, blob=blob_id, text=blob.decode("utf-8")
            )
    raise ProofError(
        f"no revision of {repo_path} in this repository's history hashes to "
        f"{recorded_checksum}. The recorded bytes were never committed at this "
        "path, so there is no revision to compare against and no equivalence "
        "claim is available -- this is not a comment-only rewrite."
    )


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def _read_tsv(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []
    return [
        line.split("\t")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() != ""
    ]


def resolve_declaration(version: str) -> tuple[str, str]:
    """Return ``(artifact_path, declared_checksum)`` for a manifest version."""
    for row in _read_tsv(APPLICATION_MANIFEST):
        if len(row) >= 6 and row[4] == version:
            return row[0], row[5]
    raise ProofError(
        f"{version} is not declared in {APPLICATION_MANIFEST.relative_to(REPO_ROOT)}"
    )


def load_canonical_adoptions() -> dict[str, dict[str, str]]:
    return {
        row[0]: dict(zip(ADOPTION_COLUMNS, row, strict=True))
        for row in _read_tsv(VERIFIED_CANONICAL_ADOPTIONS)
    }


def write_canonical_adoptions(adoptions: dict[str, dict[str, str]]) -> None:
    lines = [
        "\t".join(adoptions[version][column] for column in ADOPTION_COLUMNS)
        for version in sorted(adoptions)
    ]
    VERIFIED_CANONICAL_ADOPTIONS.write_text(
        "\n".join(lines) + "\n" if lines else "", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# proof
# ---------------------------------------------------------------------------


@dataclass
class Proof:
    version: str
    artifact_path: str
    recorded_checksum: str
    manifest_checksum: str
    recorded_commit: str = ""
    recorded_blob: str = ""
    recorded_normalized_sha256: str = ""
    current_normalized_sha256: str = ""
    verdict: str = VERDICT_UNREACHABLE
    reason: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "version": self.version,
            "artifact_path": self.artifact_path,
            "recorded_checksum": self.recorded_checksum,
            "manifest_checksum": self.manifest_checksum,
            "recorded_commit": self.recorded_commit,
            "recorded_blob": self.recorded_blob,
            "recorded_normalized_sha256": self.recorded_normalized_sha256,
            "current_normalized_sha256": self.current_normalized_sha256,
            "verdict": self.verdict,
            "reason": self.reason,
        }


def prove(version: str, recorded_checksum: str) -> Proof:
    artifact_path, declared_checksum = resolve_declaration(version)
    repo_path = f"docker/migrations/forward/{artifact_path}"
    on_disk = REPO_ROOT / repo_path
    if not on_disk.is_file():
        raise ProofError(f"{repo_path} does not exist in the working tree")

    current_bytes = on_disk.read_bytes()
    current_checksum = sha256_bytes(current_bytes)
    if current_checksum != declared_checksum:
        raise ProofError(
            f"{repo_path} hashes to {current_checksum} but the manifest declares "
            f"{declared_checksum}; the manifest is stale, so any proof against it "
            "would be a proof about bytes nobody ships"
        )
    if recorded_checksum == declared_checksum:
        raise ProofError(
            f"{version}: the recorded checksum already equals the manifest "
            "checksum, so nothing diverges and no adoption is needed"
        )

    proof = Proof(
        version=version,
        artifact_path=artifact_path,
        recorded_checksum=recorded_checksum,
        manifest_checksum=declared_checksum,
    )

    revision = find_recorded_revision(repo_path, recorded_checksum)
    proof.recorded_commit = revision.commit
    proof.recorded_blob = revision.blob

    recorded_executable = executable_text(revision.text)
    current_executable = executable_text(current_bytes.decode("utf-8"))
    if not recorded_executable or not current_executable:
        raise ProofError(
            f"{version}: one of the two revisions has no executable text once "
            "comments are removed; a comparison of nothing proves nothing"
        )
    proof.recorded_normalized_sha256 = sha256_text(recorded_executable)
    proof.current_normalized_sha256 = sha256_text(current_executable)

    if recorded_executable == current_executable:
        proof.verdict = VERDICT_COMMENT_ONLY_EQUIVALENT
        proof.reason = (
            f"revision {revision.commit[:9]} and the working tree differ only in "
            f"SQL comments and whitespace: both normalise to "
            f"{proof.current_normalized_sha256}, so the two revisions are the same "
            "program and cannot produce different schemas"
        )
        return proof

    proof.verdict = VERDICT_DIVERGENT
    proof.reason = (
        f"revision {revision.commit[:9]} and the working tree differ in executable "
        f"SQL (normalised {proof.recorded_normalized_sha256[:12]}... vs "
        f"{proof.current_normalized_sha256[:12]}...); this is not a comment-only "
        "rewrite and no equivalence claim is available from this tool"
    )
    return proof


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="node:<node>:<file>.sql")
    parser.add_argument(
        "--recorded-checksum",
        required=True,
        help="The content_sha256 the lane recorded, from "
        "platform_catalog.schema_migrations.",
    )
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument(
        "--lane",
        default="unspecified",
        help="Lane attribution recorded in the receipt.",
    )
    parser.add_argument(
        "--emit-adoption",
        action="store_true",
        help="Write the declaration row into "
        "_ledger/verified-canonical-adoptions.tsv. Only a "
        "comment_only_equivalent verdict is ever written.",
    )
    parser.add_argument(
        "--ticket",
        default=TICKET,
        help="Ticket recorded in the adoption row (defaults to this tool's own).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not SHA256_RE.match(args.recorded_checksum):
        print(
            "[prove-equivalence] FATAL: --recorded-checksum must be 64 hex chars",
            file=sys.stderr,
        )
        return 2
    if not TICKET_RE.match(args.ticket):
        print(
            f"[prove-equivalence] FATAL: --ticket must be OMN-NNNN, got {args.ticket!r}",
            file=sys.stderr,
        )
        return 2

    try:
        proof = prove(args.version, args.recorded_checksum)
    except ProofError as exc:
        print(f"[prove-equivalence] FATAL: {exc}", file=sys.stderr)
        return 2

    receipt = {
        "tool": "prove_migration_revision_equivalence.py",
        "tool_version": TOOL_VERSION,
        "ticket": TICKET,
        "lane": args.lane,
        "generated_at": datetime.now(UTC).isoformat(),
        "proof": proof.as_dict(),
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_out.write_text(payload, encoding="utf-8")
    receipt_sha = sha256_text(payload)
    print(
        f"[prove-equivalence] {proof.verdict.upper()} {proof.version}\n"
        f"[prove-equivalence]   {proof.reason}\n"
        f"[prove-equivalence] receipt {args.receipt_out} sha256={receipt_sha}",
        file=sys.stderr,
    )

    if proof.verdict != VERDICT_COMMENT_ONLY_EQUIVALENT:
        return 1

    if args.emit_adoption:
        adoptions = load_canonical_adoptions()
        adoptions[proof.version] = {
            "version": proof.version,
            "source_checksum": proof.recorded_checksum,
            "manifest_checksum": proof.manifest_checksum,
            "ticket": args.ticket,
            "receipt_sha256": receipt_sha,
            "verified_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        }
        write_canonical_adoptions(adoptions)
        print(
            "[prove-equivalence] wrote 1 canonical adoption declaration to "
            f"{VERIFIED_CANONICAL_ADOPTIONS.relative_to(REPO_ROOT)}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
