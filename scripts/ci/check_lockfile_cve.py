#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lockfile CVE gate: fail PR CI on a HIGH/CRITICAL, fix-available CVE in ``uv.lock`` (OMN-16228).

Born from the 2026-08-18 sqlparse/Trivy incident (OMN-16170): three HIGH
sqlparse CVEs (fixed in 0.6.0) were only caught by the Trivy scan on the
runtime **image**, deep in the deploy pipeline, with a staging rollout batch
queued behind the one-line fix. Detection needs to happen at
**dependency-pin time**, on the PR that introduces or keeps the vulnerable
pin -- not at image-build time.

Tool choice: ``osv-scanner``, not ``pip-audit``
------------------------------------------------
``osv-scanner`` parses ``uv.lock`` natively (confirmed against the upstream
supported-lockfiles list: Pipfile.lock, poetry.lock, requirements.txt,
pdm.lock, pylock.toml, uv.lock). ``pip-audit`` has no ``uv.lock`` parser --
using it would require an extra ``uv export --format requirements-txt`` step
that itself needs to run before every scan and adds a translation surface
(and requirements-txt export loses the exact resolution uv.lock pins). One
tool, one direct read of the real lockfile, no export step.

STRICT_GATE_JOBS, not job-level ``if:`` (fail-closed-on-skip)
---------------------------------------------------------------
Per the CI Summary doctrine (``scripts/ci/ci_summary_gate.py``), a STRICT gate
job in ``ci.yml`` never carries a job-level ``if:`` -- a job-level skip
reports ``skipped``, which is invisible to a poller expecting ``success``,
and an *unregistered* skip/absence reads as SUCCESS (the OMN-16225-class
fail-open-by-skip trap this ticket was explicitly warned about). This gate
follows the same pattern already used by ``fingerprint-check`` /
``migration-required-check`` / ``node-migration-declaration-check``: the JOB
always runs and always completes ``success`` or ``failure``; path relevance
is decided by *this script* (the ``relevant`` subcommand, fed a
``--changed-files-from`` file the same way ``detect_test_paths.py`` is), and
communicated to later STEPS (not the job) via ``$GITHUB_OUTPUT`` so the
osv-scanner install/run/evaluate steps are step-level-skipped when
irrelevant. The job itself never reports ``skipped``.

Severity classification
------------------------
Each OSV vulnerability entry is classified in priority order:

1. ``database_specific.severity`` (GHSA-sourced entries -- the overwhelming
   majority of PyPI advisories in OSV) -- a bucket string, normalized
   (``MODERATE`` -> ``MEDIUM``).
2. A ``CVSS_V3``/``CVSS_V3.1`` vector under ``severity[]`` -- scored with a
   full CVSS v3.1 base-score implementation (FIRST.org formula, reproduced
   below) and bucketed by the standard qualitative thresholds
   (0=NONE, 0.1-3.9=LOW, 4.0-6.9=MEDIUM, 7.0-8.9=HIGH, 9.0-10.0=CRITICAL).
3. Otherwise ``UNKNOWN`` -- see the blocking-decision table below for how an
   indeterminate severity is handled; it is never silently dropped.

Fix availability
------------------
A vulnerability HAS an upstream fix if any ``affected[].ranges[].events``
entry carries a ``fixed`` key (the canonical OSV "the vulnerability is fixed
in this version" marker).

Blocking decision
-------------------
| severity        | fix available | verdict                              |
|------------------|----------------|--------------------------------------|
| HIGH / CRITICAL  | yes            | BLOCK                                |
| HIGH / CRITICAL  | no             | report only (OMN-16229's lane)       |
| MEDIUM / LOW     | either         | report only                          |
| UNKNOWN          | yes            | BLOCK (can't apply the report-only   |
|                  |                | exception without knowing severity;  |
|                  |                | a fix exists, so blocking is cheap)  |
| UNKNOWN          | no             | report only (nothing actionable)     |

Exit codes: ``0`` no blocking findings | ``1`` at least one blocking finding |
``2`` misuse (bad args / malformed osv-scanner JSON).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Path prefixes/files whose change makes the scan substantive. Kept in sync
# with the two files osv-scanner actually reads.
_RELEVANT_PATHS: tuple[str, ...] = ("pyproject.toml", "uv.lock")

# Events that always warrant a full scan regardless of the changed-file set:
# these are the promotion-boundary events (push to main, merge_group), where
# rule 4's "full suite into main" doctrine applies -- a lockfile scan is not
# an expensive suite, but the same "never narrow at the boundary" logic holds.
_ALWAYS_RELEVANT_EVENTS: frozenset[str] = frozenset({"push", "merge_group"})


class EnumSeverityBucket(str, Enum):
    """Qualitative CVSS v3.1 severity bucket, plus UNKNOWN for indeterminate entries."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


_BLOCKING_BUCKETS = frozenset({EnumSeverityBucket.CRITICAL, EnumSeverityBucket.HIGH})

# Normalizes database-specific severity spellings onto the CVSS v3.1 bucket
# vocabulary. GHSA (the dominant PyPI advisory source in OSV) uses MODERATE
# where CVSS calls the same range MEDIUM.
_DATABASE_SEVERITY_ALIASES: dict[str, EnumSeverityBucket] = {
    "CRITICAL": EnumSeverityBucket.CRITICAL,
    "HIGH": EnumSeverityBucket.HIGH,
    "MODERATE": EnumSeverityBucket.MEDIUM,
    "MEDIUM": EnumSeverityBucket.MEDIUM,
    "LOW": EnumSeverityBucket.LOW,
    "NONE": EnumSeverityBucket.NONE,
}

# CVSS v3.1 base-metric weights (FIRST.org CVSS v3.1 specification section 7.4).
_AV_WEIGHTS = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
_AC_WEIGHTS = {"L": 0.77, "H": 0.44}
_PR_WEIGHTS_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}
_PR_WEIGHTS_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.5}
_UI_WEIGHTS = {"N": 0.85, "R": 0.62}
_CIA_WEIGHTS = {"H": 0.56, "L": 0.22, "N": 0.0}


@dataclass(frozen=True)
class ModelLockfileFinding:
    """One (package, vulnerability) pair from an osv-scanner run, classified."""

    package_name: str
    package_version: str
    vuln_id: str
    severity: EnumSeverityBucket
    fix_available: bool
    summary: str
    blocking: bool = field(init=False)

    def __post_init__(self) -> None:
        blocking = self.severity in _BLOCKING_BUCKETS and self.fix_available
        # UNKNOWN + fix-available also blocks (see module docstring table).
        if self.severity is EnumSeverityBucket.UNKNOWN and self.fix_available:
            blocking = True
        object.__setattr__(self, "blocking", blocking)


@dataclass(frozen=True)
class ModelLockfileCveVerdict:
    """Aggregate verdict over every finding in one osv-scanner run."""

    findings: tuple[ModelLockfileFinding, ...]

    @property
    def blocking_findings(self) -> tuple[ModelLockfileFinding, ...]:
        return tuple(f for f in self.findings if f.blocking)

    @property
    def report_only_findings(self) -> tuple[ModelLockfileFinding, ...]:
        return tuple(f for f in self.findings if not f.blocking)

    @property
    def passed(self) -> bool:
        return not self.blocking_findings


def _cvss_v3_base_score(vector: str) -> float | None:
    """Compute the CVSS v3.1 base score from a full vector string.

    Implements the FIRST.org CVSS v3.1 base-score formula exactly (section
    7.4 of the spec: ISCBase -> Impact -> Exploitability -> BaseScore, with
    the integer-arithmetic Roundup to avoid float drift). Returns ``None`` if
    the vector is missing a required base metric.
    """
    metrics: dict[str, str] = {}
    for part in vector.split("/"):
        if ":" not in part or part.startswith("CVSS"):
            continue
        key, _, value = part.partition(":")
        metrics[key] = value

    try:
        av = _AV_WEIGHTS[metrics["AV"]]
        ac = _AC_WEIGHTS[metrics["AC"]]
        ui = _UI_WEIGHTS[metrics["UI"]]
        scope_changed = metrics["S"] == "C"
        pr_table = _PR_WEIGHTS_CHANGED if scope_changed else _PR_WEIGHTS_UNCHANGED
        pr = pr_table[metrics["PR"]]
        c = _CIA_WEIGHTS[metrics["C"]]
        i = _CIA_WEIGHTS[metrics["I"]]
        a = _CIA_WEIGHTS[metrics["A"]]
    except KeyError:
        return None

    isc_base = 1 - ((1 - c) * (1 - i) * (1 - a))
    if scope_changed:
        impact = 7.52 * (isc_base - 0.029) - 3.25 * (isc_base - 0.02) ** 15
    else:
        impact = 6.42 * isc_base

    exploitability = 8.22 * av * ac * pr * ui

    if impact <= 0:
        return 0.0

    raw = (impact + exploitability) * (1.08 if scope_changed else 1.0)
    raw = min(raw, 10.0)
    return _roundup(raw)


def _roundup(score: float) -> float:
    """CVSS spec Roundup(): round UP to one decimal place via integer arithmetic."""
    int_score = round(score * 100000)
    if int_score % 10000 == 0:
        return int_score / 100000.0
    return (math.floor(int_score / 10000) + 1) / 10.0


def _bucket_from_score(score: float) -> EnumSeverityBucket:
    if score <= 0.0:
        return EnumSeverityBucket.NONE
    if score < 4.0:
        return EnumSeverityBucket.LOW
    if score < 7.0:
        return EnumSeverityBucket.MEDIUM
    if score < 9.0:
        return EnumSeverityBucket.HIGH
    return EnumSeverityBucket.CRITICAL


def _classify_severity(vuln: dict[str, Any]) -> EnumSeverityBucket:
    database_specific = vuln.get("database_specific") or {}
    raw = database_specific.get("severity")
    if isinstance(raw, str) and raw.upper() in _DATABASE_SEVERITY_ALIASES:
        return _DATABASE_SEVERITY_ALIASES[raw.upper()]

    for entry in vuln.get("severity") or []:
        entry_type = str(entry.get("type", ""))
        if entry_type.startswith("CVSS_V3"):
            score = _cvss_v3_base_score(str(entry.get("score", "")))
            if score is not None:
                return _bucket_from_score(score)

    return EnumSeverityBucket.UNKNOWN


def _has_upstream_fix(vuln: dict[str, Any]) -> bool:
    for affected in vuln.get("affected") or []:
        for rng in affected.get("ranges") or []:
            for event in rng.get("events") or []:
                if "fixed" in event:
                    return True
    return False


def evaluate_osv_results(osv_json: dict[str, Any]) -> ModelLockfileCveVerdict:
    """Classify every vulnerability in an osv-scanner ``--format=json`` payload."""
    findings: list[ModelLockfileFinding] = []
    for result in osv_json.get("results") or []:
        for pkg_entry in result.get("packages") or []:
            package = pkg_entry.get("package") or {}
            pkg_name = str(package.get("name", "<unknown>"))
            pkg_version = str(package.get("version", "<unknown>"))
            for vuln in pkg_entry.get("vulnerabilities") or []:
                severity = _classify_severity(vuln)
                findings.append(
                    ModelLockfileFinding(
                        package_name=pkg_name,
                        package_version=pkg_version,
                        vuln_id=str(vuln.get("id", "<unknown>")),
                        severity=severity,
                        fix_available=_has_upstream_fix(vuln),
                        summary=str(vuln.get("summary") or vuln.get("details") or ""),
                    )
                )
    return ModelLockfileCveVerdict(findings=tuple(findings))


def is_scan_relevant(changed_files: Iterable[str], event_name: str) -> bool:
    """Decide whether a substantive osv-scanner run is warranted."""
    if event_name in _ALWAYS_RELEVANT_EVENTS:
        return True
    normalized = {f.strip() for f in changed_files if f.strip()}
    return any(path in normalized for path in _RELEVANT_PATHS)


def _read_changed_files(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text().splitlines() if line.strip()]


def _format_report(verdict: ModelLockfileCveVerdict) -> str:
    lines: list[str] = []
    if not verdict.findings:
        lines.append("No known vulnerabilities found in uv.lock.")
        return "\n".join(lines)

    if verdict.blocking_findings:
        lines.append(
            f"BLOCKING: {len(verdict.blocking_findings)} HIGH/CRITICAL "
            "finding(s) with an upstream fix available:"
        )
        for f in verdict.blocking_findings:
            lines.append(
                f"  - [{f.severity.value}] {f.vuln_id} in "
                f"{f.package_name}=={f.package_version}: {f.summary}"
            )

    if verdict.report_only_findings:
        lines.append(
            f"Report-only: {len(verdict.report_only_findings)} finding(s) "
            "(no upstream fix, or below the HIGH threshold):"
        )
        for f in verdict.report_only_findings:
            lines.append(
                f"  - [{f.severity.value}] {f.vuln_id} in "
                f"{f.package_name}=={f.package_version} "
                f"(fix_available={f.fix_available}): {f.summary}"
            )

    return "\n".join(lines)


def _cmd_relevant(args: argparse.Namespace) -> int:
    changed = _read_changed_files(Path(args.changed_files_from))
    relevant = is_scan_relevant(changed, args.event_name)
    print("true" if relevant else "false")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    raw_path = Path(args.osv_json)
    try:
        payload = json.loads(raw_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not read/parse {raw_path}: {exc}", file=sys.stderr)
        return 2

    verdict = evaluate_osv_results(payload)
    print(_format_report(verdict))
    return 0 if verdict.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lockfile CVE gate (OMN-16228)")
    sub = parser.add_subparsers(dest="command", required=True)

    relevant_parser = sub.add_parser(
        "relevant", help="Decide whether a substantive scan is warranted"
    )
    relevant_parser.add_argument("--changed-files-from", required=True)
    relevant_parser.add_argument("--event-name", required=True)
    relevant_parser.set_defaults(func=_cmd_relevant)

    evaluate_parser = sub.add_parser(
        "evaluate", help="Classify an osv-scanner --format=json payload"
    )
    evaluate_parser.add_argument("--osv-json", required=True)
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
