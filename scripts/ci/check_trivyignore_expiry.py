#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Expiring-ignore policy for fix-unavailable CVEs in the Trivy image gate (OMN-16229).

Companion to OMN-16228. Trivy's image-build Docker scan
(``.github/workflows/build-and-push-runtime.yml`` /
``build-workspace-candidate-runtime.yml``) is a hard block with
``exit-code: 1`` and ``ignore-unfixed: true``. ``ignore-unfixed`` already
means a CVE with NO published fix never blocks -- that half of the
2026-08-18 sqlparse/Trivy incident (OMN-16170) is handled by Trivy itself.
What ``ignore-unfixed`` cannot do is stay auditable: it silently exempts
every unfixed finding forever, with no ticket, no reviewer, no forced
re-triage date. This module is the other half -- a committed ``.trivyignore``
with **mandatory, time-bounded metadata per entry**, so a fix-unavailable
exemption is reviewed, dated, and eventually forced back in front of a human
rather than becoming permanent by default.

Format
------
Each ignore entry is a 4-line metadata comment block immediately followed by
the bare CVE/GHSA id line Trivy itself reads (Trivy's own ``.trivyignore``
format: one vulnerability id per line, optionally with an inline ``#``
comment -- this module only cares about the block ABOVE that line)::

    # CVE: CVE-2024-12345
    # reason: no-upstream-fix
    # ticket: OMN-12345
    # expires: 2026-12-31
    CVE-2024-12345

Rules, each independently checked and reported:

* ``# CVE:`` value MUST equal the vulnerability id line immediately below the
  block (catches copy-paste drift between the metadata and what Trivy
  actually ignores).
* ``# reason:`` MUST be exactly ``no-upstream-fix`` -- the only reason this
  file exists to record. A fix-available CVE must never be silenced here;
  Trivy's own ``ignore-unfixed: true`` already refuses to suppress those
  regardless of what this file contains, so a malformed/absent reason field
  is a metadata defect, not a bypass.
* ``# ticket:`` MUST match ``OMN-<digits>`` -- every exemption traces to a
  tracking ticket, never a bare justification in prose.
* ``# expires:`` MUST be an ISO-8601 date (``YYYY-MM-DD``) STRICTLY IN THE
  FUTURE relative to the run. An expired entry FAILS THE BUILD -- that is the
  forcing function: the exemption lapses and someone must re-triage rather
  than the ignore silently living forever.
* A bare vulnerability-id line with NO preceding metadata block is malformed
  (no reason/ticket/expiry to check) and fails.

This module does not talk to Trivy or interpret scan output -- it only
validates the committed ``.trivyignore`` file's shape and expiry. Whether a
CVE has a fix is Trivy's job (``ignore-unfixed: true``); whether a
fix-unavailable ignore is still valid is this module's job.

Exit codes: ``0`` every entry valid (or the file is empty/header-only) |
``1`` at least one malformed or expired entry.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

_CVE_FIELD_RE = re.compile(r"^#\s*CVE:\s*(\S+)\s*$")
_REASON_FIELD_RE = re.compile(r"^#\s*reason:\s*(\S+)\s*$")
_TICKET_FIELD_RE = re.compile(r"^#\s*ticket:\s*(\S+)\s*$")
_EXPIRES_FIELD_RE = re.compile(r"^#\s*expires:\s*(\S+)\s*$")
_TICKET_VALUE_RE = re.compile(r"^OMN-\d+$")
_DATE_VALUE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_REQUIRED_REASON = "no-upstream-fix"
# A bare Trivy ignore-id line: CVE-YYYY-NNNN..., GHSA-xxxx-xxxx-xxxx, or a
# PYSEC/GO/RUSTSEC-style advisory id. Comment lines (starting with `#`) and
# blank lines are never id lines.
_ID_LINE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*-[A-Za-z0-9-]+$")


@dataclass(frozen=True)
class ModelTrivyIgnoreViolation:
    line_number: int
    reason: str


@dataclass(frozen=True)
class ModelTrivyIgnoreEntry:
    vuln_id: str
    cve_field: str | None
    reason_field: str | None
    ticket_field: str | None
    expires_field: str | None
    line_number: int


@dataclass(frozen=True)
class ModelTrivyIgnoreVerdict:
    entries: tuple[ModelTrivyIgnoreEntry, ...]
    violations: tuple[ModelTrivyIgnoreViolation, ...]

    @property
    def passed(self) -> bool:
        return not self.violations


def _parse_entries(lines: list[str]) -> list[ModelTrivyIgnoreEntry]:
    """Group ``.trivyignore`` lines into (metadata block, id line) entries.

    A metadata block is a contiguous run of ``# CVE:``/``# reason:``/
    ``# ticket:``/``# expires:`` comment lines. Any other comment line, or a
    blank line, resets the in-progress block (it cannot attach to a later,
    non-adjacent id line). An id line with no immediately-preceding block
    still produces an entry -- with every field ``None`` -- so it is reported
    as malformed rather than silently skipped.
    """
    entries: list[ModelTrivyIgnoreEntry] = []
    pending: dict[str, str] = {}

    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if not stripped:
            pending = {}
            continue

        cve_match = _CVE_FIELD_RE.match(stripped)
        reason_match = _REASON_FIELD_RE.match(stripped)
        ticket_match = _TICKET_FIELD_RE.match(stripped)
        expires_match = _EXPIRES_FIELD_RE.match(stripped)

        if cve_match:
            pending["cve"] = cve_match.group(1)
            continue
        if reason_match:
            pending["reason"] = reason_match.group(1)
            continue
        if ticket_match:
            pending["ticket"] = ticket_match.group(1)
            continue
        if expires_match:
            pending["expires"] = expires_match.group(1)
            continue
        if stripped.startswith("#"):
            # Any other comment (a header, a blank explainer line) breaks a
            # block -- metadata must be immediately adjacent to its id line.
            pending = {}
            continue

        if _ID_LINE_RE.match(stripped):
            entries.append(
                ModelTrivyIgnoreEntry(
                    vuln_id=stripped,
                    cve_field=pending.get("cve"),
                    reason_field=pending.get("reason"),
                    ticket_field=pending.get("ticket"),
                    expires_field=pending.get("expires"),
                    line_number=idx,
                )
            )
            pending = {}
            continue

        # A non-comment, non-id, non-blank line: leave pending as-is (Trivy
        # ignores unrecognized lines too) but do not attach it to anything.
        pending = {}

    return entries


def _validate_entry(
    entry: ModelTrivyIgnoreEntry, today: date
) -> ModelTrivyIgnoreViolation | None:
    if entry.cve_field is None:
        return ModelTrivyIgnoreViolation(
            entry.line_number,
            f"{entry.vuln_id}: no preceding `# CVE:` metadata block -- every "
            "ignore entry MUST carry CVE id, reason, ticket, and expires.",
        )
    if entry.cve_field != entry.vuln_id:
        return ModelTrivyIgnoreViolation(
            entry.line_number,
            f"{entry.vuln_id}: `# CVE: {entry.cve_field}` does not match the "
            f"id line `{entry.vuln_id}` -- metadata/id drift.",
        )
    if entry.reason_field != _REQUIRED_REASON:
        return ModelTrivyIgnoreViolation(
            entry.line_number,
            f"{entry.vuln_id}: `# reason:` must be exactly "
            f"`{_REQUIRED_REASON}`, got {entry.reason_field!r}.",
        )
    if entry.ticket_field is None or not _TICKET_VALUE_RE.match(entry.ticket_field):
        return ModelTrivyIgnoreViolation(
            entry.line_number,
            f"{entry.vuln_id}: `# ticket:` must match `OMN-<digits>`, got "
            f"{entry.ticket_field!r}.",
        )
    if entry.expires_field is None or not _DATE_VALUE_RE.match(entry.expires_field):
        return ModelTrivyIgnoreViolation(
            entry.line_number,
            f"{entry.vuln_id}: `# expires:` must be an ISO-8601 date "
            f"(YYYY-MM-DD), got {entry.expires_field!r}.",
        )
    try:
        expires_date = date.fromisoformat(entry.expires_field)
    except ValueError:
        return ModelTrivyIgnoreViolation(
            entry.line_number,
            f"{entry.vuln_id}: `# expires: {entry.expires_field}` is not a "
            "valid calendar date.",
        )
    if expires_date <= today:
        return ModelTrivyIgnoreViolation(
            entry.line_number,
            f"{entry.vuln_id}: expired on {entry.expires_field} (today is "
            f"{today.isoformat()}). Re-triage: confirm no fix has shipped, "
            "then either remove the entry or bump `# expires:` under the "
            f"same ticket ({entry.ticket_field}).",
        )
    return None


def evaluate_trivyignore(
    text: str, today: date | None = None
) -> ModelTrivyIgnoreVerdict:
    """Parse and validate a ``.trivyignore`` file's text."""
    if today is None:
        today = datetime.now(tz=None).date()
    entries = tuple(_parse_entries(text.splitlines(keepends=True)))
    violations = tuple(
        v for e in entries if (v := _validate_entry(e, today)) is not None
    )
    return ModelTrivyIgnoreVerdict(entries=entries, violations=violations)


def _format_report(verdict: ModelTrivyIgnoreVerdict) -> str:
    if not verdict.entries:
        return "No .trivyignore entries (header-only or empty file)."
    lines = [f"{len(verdict.entries)} .trivyignore entrie(s) checked."]
    if verdict.violations:
        lines.append(f"FAILED: {len(verdict.violations)} violation(s):")
        for v in verdict.violations:
            lines.append(f"  - line {v.line_number}: {v.reason}")
    else:
        lines.append("All entries carry valid, unexpired metadata.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Expiring-ignore policy check for .trivyignore (OMN-16229)"
    )
    parser.add_argument("path", nargs="?", default=".trivyignore")
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        # No .trivyignore is a valid state (nothing is being ignored).
        print(f"{path} does not exist -- nothing to validate.")
        return 0

    verdict = evaluate_trivyignore(path.read_text())
    print(_format_report(verdict))
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
