# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex occ`` — local stamp/validate of canonical OCC PR metadata (OMN-14190).

Piece 4/5 of the canonical OCC stamp-model (parent epic OMN-14180). Shift-left
enforcement of the ``Evidence-Source`` / ``Evidence-Ticket`` metadata that the
CI receipt-gate (``occ-preflight.yml`` / ``validator_receipt_gate``) parses,
run locally as a pre-commit hook BEFORE CI so a malformed or incomplete stamp
fails at ``git commit`` time instead of on the PR.

This command owns **zero** stamp logic. Parsing and rendering are delegated
verbatim to the canonical OCC stamp schema in
:mod:`omnibase_compat.contracts.pr_occ_stamp` (relocated there from
``omnibase_core`` under OMN-14223); this module is only the click plumbing, the
fail-closed assertions, and the idempotent stamp-insert glue. The accepted token
shapes (``OCC#<n>`` / commit sha / ``OMN-<n>``) therefore have a single source of
truth — the compat parser and ``ModelPrOccMetadataStamp`` validators — and this
CLI never re-derives them.

Two subcommands:

``onex occ validate [FILES...] [--stdin]``
    Parse each PR body (file args, or stdin when no files are given) and assert
    an ``Evidence-Source`` line **and** at least one ``Evidence-Ticket`` line
    are present and well-formed. Any problem prints an actionable message to
    stderr and exits non-zero — fail-closed. A missing/malformed stamp is a
    failure, never a silent pass.

``onex occ stamp [FILE] [--ticket OMN-N ...] [--evidence-source OCC#N|sha] [--in-place]``
    Idempotently insert the canonical Evidence block. Re-running with the same
    inputs does **not** create a duplicate block: the canonical renderer strips
    the renderer-owned Evidence section and re-appends exactly one canonical
    block, so ``stamp(stamp(x)) == stamp(x)`` is a fixpoint.

.. versionadded:: OMN-14190
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import click

# Canonical OCC stamp schema — single import block. The models + parser/renderer
# were relocated from omnibase_core to omnibase_compat (the lowest shared layer)
# under OMN-14223 so every repo consumes one definition. This CLI owns none of it.
from omnibase_compat.contracts.pr_occ_stamp import (
    ModelPrEvidenceSource,
    ModelPrOccMetadataStamp,
    parse_pr_occ_metadata_stamp,
    render_pr_occ_metadata_stamp,
)

__all__ = [
    "occ",
    "occ_validate",
    "occ_stamp",
    "parse_evidence_source_token",
    "validate_pr_body",
    "stamp_pr_body",
]


def parse_evidence_source_token(value: str) -> ModelPrEvidenceSource:
    """Validate and parse an ``--evidence-source`` token.

    Delegates format validation to the Piece-2 parser by round-tripping the
    value through a synthetic ``Evidence-Source:`` line, so the accepted shapes
    (``OCC#<pr-number>`` or a 7-40 char commit sha) have exactly one source of
    truth. Raises :class:`ValueError` on anything the parser rejects.
    """
    source: ModelPrEvidenceSource | None = parse_pr_occ_metadata_stamp(
        f"Evidence-Source: {value}"
    ).evidence_source
    if source is None:
        raise ValueError(
            f"invalid --evidence-source {value!r}: expected 'OCC#<pr-number>' "
            "(e.g. OCC#1408) or a 7-40 char commit sha (e.g. 7420667f)"
        )
    return source


def validate_pr_body(body: str, *, label: str = "PR body") -> list[str]:
    """Return a list of fail-closed problems for ``body`` (empty tuple == valid).

    A body is valid only when the Piece-2 parser resolves a well-formed
    ``Evidence-Source`` **and** at least one well-formed ``Evidence-Ticket``.
    The parser yields ``None``/empty for both missing and malformed lines, so a
    single presence check covers both failure modes.
    """
    stamp = parse_pr_occ_metadata_stamp(body)
    problems: list[str] = []
    if stamp.evidence_source is None:
        problems.append(
            f"{label}: missing or malformed 'Evidence-Source:' line — expected "
            "'Evidence-Source: OCC#<pr-number>' (e.g. OCC#1408) or a commit sha "
            "(e.g. 7420667f)."
        )
    if not stamp.evidence_tickets:
        problems.append(
            f"{label}: missing or malformed 'Evidence-Ticket:' line — expected "
            "'Evidence-Ticket: OMN-<n>' (e.g. OMN-14190)."
        )
    return problems


def stamp_pr_body(
    body: str,
    *,
    tickets: Sequence[str] = (),
    evidence_source: str | None = None,
) -> str:
    """Return ``body`` with the canonical Evidence block idempotently inserted.

    ``tickets`` are merged with any already present in the body and de-duplicated
    (first-seen order preserved) by the ``ModelPrOccMetadataStamp`` validator.
    ``evidence_source`` overrides an existing source when supplied; otherwise the
    parsed source is preserved. Idempotency is inherited from the Piece-2
    renderer, which emits exactly one Evidence block regardless of how many were
    present in the input.
    """
    base = parse_pr_occ_metadata_stamp(body)
    merged_tickets = list(base.evidence_tickets) + list(tickets)
    source = (
        parse_evidence_source_token(evidence_source)
        if evidence_source is not None
        else base.evidence_source
    )
    stamped = ModelPrOccMetadataStamp(
        repo=base.repo,
        pr_number=base.pr_number,
        head_sha=base.head_sha,
        evidence_source=source,
        evidence_tickets=merged_tickets,
        skip_tokens=base.skip_tokens,
        body_sections=base.body_sections,
    )
    rendered: str = render_pr_occ_metadata_stamp(stamped)
    return rendered


@click.group("occ")
def occ() -> None:  # stub-ok: click group, subcommands via @occ.command()
    """Local stamp/validate for canonical OCC PR metadata (OMN-14190)."""


def _read_body(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@occ.command("validate")
@click.argument(
    "files",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--stdin",
    "use_stdin",
    is_flag=True,
    default=False,
    help="Read a single PR body from stdin instead of file arguments.",
)
def occ_validate(files: tuple[Path, ...], use_stdin: bool) -> None:
    """Fail-closed check that each PR body carries a well-formed OCC stamp.

    With FILES, every file is validated and all problems are reported at once.
    With --stdin (or no FILES), a single body is read from stdin. Exits non-zero
    with an actionable message when any Evidence-Source / Evidence-Ticket line is
    missing or malformed.
    """
    if use_stdin and files:
        raise click.UsageError("--stdin cannot be combined with FILE arguments.")

    problems: list[str] = []
    if files:
        for path in files:
            problems.extend(validate_pr_body(_read_body(path), label=str(path)))
    else:
        problems.extend(validate_pr_body(sys.stdin.read(), label="<stdin>"))

    if problems:
        for problem in problems:
            click.echo(f"ERROR: {problem}", err=True)
        click.echo(
            "OCC stamp validation failed (OMN-14190). Add the missing lines or "
            "run 'onex occ stamp' to insert them, then re-commit.",
            err=True,
        )
        sys.exit(1)


@occ.command("stamp")
@click.argument(
    "file",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--ticket",
    "tickets",
    multiple=True,
    metavar="OMN-N",
    help="Evidence ticket to insert (repeatable). Merged and de-duplicated.",
)
@click.option(
    "--evidence-source",
    "evidence_source",
    default=None,
    metavar="OCC#N|SHA",
    help="Evidence source token (OCC#<pr-number> or a commit sha). Overrides "
    "an existing source when supplied.",
)
@click.option(
    "--in-place",
    "in_place",
    is_flag=True,
    default=False,
    help="Rewrite FILE with the stamped body instead of printing to stdout.",
)
def occ_stamp(
    file: Path | None,
    tickets: tuple[str, ...],
    evidence_source: str | None,
    in_place: bool,
) -> None:
    """Idempotently insert the canonical OCC Evidence block into a PR body.

    Reads FILE (or stdin when omitted), merges in --ticket / --evidence-source,
    and writes exactly one canonical Evidence block. Re-running with the same
    inputs is a no-op on the rendered output — no duplicate blocks.
    """
    if in_place and file is None:
        raise click.UsageError("--in-place requires a FILE argument.")

    body = _read_body(file) if file is not None else sys.stdin.read()
    try:
        stamped = stamp_pr_body(body, tickets=tickets, evidence_source=evidence_source)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    if in_place and file is not None:
        file.write_text(stamped, encoding="utf-8")
    else:
        click.echo(stamped, nl=False)
