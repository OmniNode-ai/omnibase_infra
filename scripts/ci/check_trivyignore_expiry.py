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
  file exists to record. A fix-available CVE must never be silenced here; when
  ``--trivy-json`` is supplied, this module proves every ignored ID appears in
  the unignored image scan with no fixed version.
* ``# ticket:`` MUST match ``OMN-<digits>`` -- every exemption traces to a
  tracking ticket, never a bare justification in prose.
* ``# expires:`` MUST be an ISO-8601 date (``YYYY-MM-DD``) STRICTLY IN THE
  FUTURE relative to the run. An expired entry FAILS THE BUILD -- that is the
  forcing function: the exemption lapses and someone must re-triage rather
  than the ignore silently living forever.
* A bare vulnerability-id line with NO preceding metadata block is malformed
  (no reason/ticket/expiry to check) and fails.

Without ``--trivy-json``, this module validates the committed
``.trivyignore`` file's shape and expiry. With ``--trivy-json``, it also
interprets an unignored Trivy report for the exact image being scanned and
fails any stale or fix-available ignore before the blocking scan can apply
``.trivyignore``.

Exit codes: ``0`` every entry valid (or the file is empty/header-only) |
``1`` at least one malformed or expired entry.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timezone
from pathlib import Path
from typing import Any, cast

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


def _parse_entries(
    lines: list[str],
) -> tuple[list[ModelTrivyIgnoreEntry], list[ModelTrivyIgnoreViolation]]:
    """Group ``.trivyignore`` lines into (metadata block, id line) entries.

    A metadata block is a contiguous run of ``# CVE:``/``# reason:``/
    ``# ticket:``/``# expires:`` comment lines. Any other comment line, or a
    blank line, resets the in-progress block (it cannot attach to a later,
    non-adjacent id line). An id line with no immediately-preceding block
    still produces an entry -- with every field ``None`` -- so it is reported
    as malformed rather than silently skipped.
    """
    entries: list[ModelTrivyIgnoreEntry] = []
    violations: list[ModelTrivyIgnoreViolation] = []
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

        # A non-comment, non-id, non-blank line may still be meaningful to
        # Trivy, for example legacy inline expiry syntax. Fail closed so every
        # active ignore line is covered by the repository policy block above it.
        violations.append(
            ModelTrivyIgnoreViolation(
                idx,
                f"{stripped!r}: unsupported .trivyignore line -- every "
                "non-comment entry MUST be a bare vulnerability id with the "
                "required metadata block immediately above it.",
            )
        )
        pending = {}

    return entries, violations


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


def _vulnerabilities_by_id(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    results = report.get("Results", [])
    if not isinstance(results, list):
        raise ValueError("Trivy JSON report `Results` must be a list")

    by_id: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        vulnerabilities = result.get("Vulnerabilities", [])
        if vulnerabilities is None:
            continue
        if not isinstance(vulnerabilities, list):
            raise ValueError("Trivy JSON report `Vulnerabilities` must be a list")
        for vulnerability in vulnerabilities:
            if not isinstance(vulnerability, dict):
                continue
            vuln_id = vulnerability.get("VulnerabilityID")
            if not isinstance(vuln_id, str):
                continue
            by_id.setdefault(vuln_id, []).append(cast("dict[str, Any]", vulnerability))
    return by_id


def _validate_unignored_trivy_report(
    entries: tuple[ModelTrivyIgnoreEntry, ...],
    report: dict[str, Any],
) -> tuple[ModelTrivyIgnoreViolation, ...]:
    """Require ignored IDs to be present and unfixed in the unignored image scan."""
    vulnerabilities = _vulnerabilities_by_id(report)
    violations: list[ModelTrivyIgnoreViolation] = []
    for entry in entries:
        matches = vulnerabilities.get(entry.vuln_id, [])
        if not matches:
            violations.append(
                ModelTrivyIgnoreViolation(
                    entry.line_number,
                    f"{entry.vuln_id}: not present in the unignored Trivy JSON "
                    "report for this image; remove the stale .trivyignore entry.",
                )
            )
            continue
        fixed_versions = sorted(
            {
                str(match.get("FixedVersion", "")).strip()
                for match in matches
                if str(match.get("FixedVersion", "")).strip()
            }
        )
        if fixed_versions:
            violations.append(
                ModelTrivyIgnoreViolation(
                    entry.line_number,
                    f"{entry.vuln_id}: unignored Trivy JSON reports fixed "
                    f"version(s) {', '.join(fixed_versions)}; .trivyignore may "
                    "only suppress fix-unavailable findings.",
                )
            )
    return tuple(violations)


def evaluate_trivyignore(
    text: str, today: date | None = None, trivy_report: dict[str, Any] | None = None
) -> ModelTrivyIgnoreVerdict:
    """Parse and validate a ``.trivyignore`` file's text."""
    if today is None:
        today = datetime.now(UTC).date()
    parsed_entries, parse_violations = _parse_entries(text.splitlines(keepends=True))
    entries = tuple(parsed_entries)
    violations = tuple(
        [*parse_violations]
        + [v for e in entries if (v := _validate_entry(e, today)) is not None]
    )
    if trivy_report is not None:
        violations = violations + _validate_unignored_trivy_report(
            entries,
            trivy_report,
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
    parser.add_argument(
        "--trivy-json",
        type=Path,
        help=(
            "Unignored Trivy JSON report for the exact image being scanned. "
            "When supplied, every .trivyignore entry must appear in this report "
            "with no FixedVersion."
        ),
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if not path.exists():
        # No .trivyignore is a valid state (nothing is being ignored).
        print(f"{path} does not exist -- nothing to validate.")
        return 0

    trivy_report = None
    if args.trivy_json is not None:
        trivy_report = json.loads(args.trivy_json.read_text(encoding="utf-8"))
        if not isinstance(trivy_report, dict):
            raise ValueError("Trivy JSON report must be a mapping")

    verdict = evaluate_trivyignore(path.read_text(), trivy_report=trivy_report)
    print(_format_report(verdict))
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
