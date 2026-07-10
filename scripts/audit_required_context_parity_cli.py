#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Required-context parity ratchet CLI (OMN-14288) — REPORT-ONLY.

Reads the declarative parity manifest (scripts/enforcement_parity_manifest.yaml),
fetches live branch-protection + aggregator-workflow state via `gh api`, and
reports where a CLAUDE.md/doctrine-CLAIMED load-bearing gate is NOT actually
enforced on live branch protection:

  * MISSING       — a manifest-declared `direct` gate absent from required_status_checks
  * NEEDS_CLOSURE — a `needs_child` gate not in its aggregator's transitive needs-closure
  * UNPROTECTED   — a declared branch with no protection object at all

This closes the exact hole the enforcement-parity audit found: the existing
branch-protection auditor only checks the ORPHANED direction (required -> seen),
never MISSING (manifest -> required) nor `needs:`-closure coverage.

REPORT-ONLY (report-then-enforce rollout discipline):
  * It NEVER mutates branch protection.
  * It ALWAYS exits 0, even with findings, unless `--fail-on-findings` is passed
    (reserved for the future enforcing follow-up; default OFF).

The pure assertion logic lives in audit_branch_protection_lib.py; this module is
the thin I/O shell (manifest + `gh api` + workflow YAML parse) so that logic
stays unit-testable without network.

Usage:
  uv run python scripts/audit_required_context_parity_cli.py report --owner OmniNode-ai
  uv run python scripts/audit_required_context_parity_cli.py report --json
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from audit_branch_protection_lib import (
    PARITY_MISSING,
    PARITY_NEEDS_CLOSURE,
    PARITY_UNPROTECTED,
    evaluate_manifest_parity,
)

_DEFAULT_MANIFEST = Path(__file__).resolve().parent / "enforcement_parity_manifest.yaml"


def real_gh(args: list[str]) -> tuple[int, str]:
    """Invoke the real `gh` CLI. Returns (returncode, stdout)."""
    proc = subprocess.run(
        ["gh", *args], capture_output=True, text=True, timeout=30, check=False
    )
    return proc.returncode, proc.stdout


def fetch_required_contexts(
    owner: str, repo: str, branch: str, gh: Any
) -> list[str] | None:
    """Fetch live required_status_checks.contexts for a repo+branch.

    Returns the contexts list, or None when the branch has no protection object
    (HTTP 404 "Branch not protected") — which the evaluator maps to UNPROTECTED.
    """
    rc, out = gh(
        [
            "api",
            f"repos/{owner}/{repo}/branches/{branch}/protection/required_status_checks",
        ]
    )
    if rc != 0:
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    contexts = data.get("contexts", [])
    return [str(c) for c in contexts] if isinstance(contexts, list) else []


def fetch_workflow_jobs(
    owner: str, repo: str, workflow_file: str, gh: Any
) -> dict[str, Any] | None:
    """Fetch + parse the `.jobs` mapping of a repo's workflow file.

    Uses the Contents API (base64) so it works cross-repo with the same PAT the
    existing auditor uses. Returns None on any fetch/parse failure (the
    evaluator maps that to INDETERMINATE, never a false MISSING).
    """
    rc, out = gh(
        [
            "api",
            f"repos/{owner}/{repo}/contents/.github/workflows/{workflow_file}",
            "--jq",
            ".content",
        ]
    )
    if rc != 0 or not out.strip():
        return None
    try:
        # binascii.Error (bad base64) and UnicodeDecodeError both subclass
        # ValueError; yaml.YAMLError covers a malformed workflow. On any of
        # these the gate is reported INDETERMINATE, never a false MISSING.
        decoded = base64.b64decode(out).decode("utf-8")
        data = yaml.safe_load(decoded)
    except (ValueError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    jobs = data.get("jobs")
    return jobs if isinstance(jobs, dict) else None


def collect_live_state(manifest: dict[str, Any], owner: str, gh: Any) -> dict[str, Any]:
    """Fetch all live state the manifest references: required contexts per
    repo+branch, plus aggregator workflow-job graphs for needs_child gates."""
    live: dict[str, Any] = {}
    for repo, branches in (manifest.get("repos", {}) or {}).items():
        for branch, spec in (branches or {}).items():
            key = f"{repo}:{branch}"
            required = fetch_required_contexts(owner, repo, branch, gh)
            aggregator_jobs: dict[str, Any] = {}
            for gate in (spec or {}).get("load_bearing_gates", []) or []:
                if str(gate.get("coverage", "direct")) != "needs_child":
                    continue
                aggregator = str(gate.get("aggregator", ""))
                workflow_file = str(gate.get("aggregator_workflow", ""))
                if aggregator in aggregator_jobs or not workflow_file:
                    continue
                jobs = fetch_workflow_jobs(owner, repo, workflow_file, gh)
                if jobs is not None:
                    aggregator_jobs[aggregator] = jobs
            live[key] = {"required": required, "aggregator_jobs": aggregator_jobs}
    return live


def _render_human(manifest: dict[str, Any], findings: list[dict[str, Any]]) -> None:
    """Print an org-wide, per-repo human-readable report."""
    by_key: dict[str, list[dict[str, Any]]] = {}
    for f in findings:
        by_key.setdefault(f"{f['repo']}:{f['branch']}", []).append(f)

    print(
        "=== required-context parity report (REPORT-ONLY — no enforcement, no mutation) ==="
    )
    print(
        f"manifest gates declared across {len(manifest.get('repos', {}) or {})} repo(s)"
    )
    print("")
    for repo, branches in (manifest.get("repos", {}) or {}).items():
        for branch, spec in (branches or {}).items():
            key = f"{repo}:{branch}"
            gate_count = len((spec or {}).get("load_bearing_gates", []) or [])
            repo_findings = by_key.get(key, [])
            print(f"--- {key}  ({gate_count} declared gate(s))")
            if not repo_findings:
                print(f"  [OK]   all {gate_count} declared gate(s) enforced")
            for f in repo_findings:
                print(f"  [{f['class']}]  {f['gate']}  (rule: {f['rule']})")
                print(f"      {f['detail']}")
    print("")

    counts: dict[str, int] = {}
    for f in findings:
        counts[f["class"]] = counts.get(f["class"], 0) + 1
    print("=== SUMMARY ===")
    print(f"total findings: {len(findings)}")
    if findings:
        parts = ", ".join(f"{cls}={n}" for cls, n in sorted(counts.items()))
        print(f"by class: {parts}")
    missing = [f for f in findings if f["class"] == PARITY_MISSING]
    if missing:
        print("")
        print("MISSING gates (claimed-enforced but NOT in required_status_checks):")
        for f in missing:
            print(f"  - {f['repo']}:{f['branch']}  {f['gate']}  ({f['rule']})")
    print("")
    print(
        "REPORT-ONLY: exit 0 regardless of findings "
        "(report-then-enforce rollout — the enforcing per-PR gate lands separately)."
    )


def cmd_report(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    owner = args.owner or manifest.get("owner", "OmniNode-ai")

    live = collect_live_state(manifest, owner, real_gh)
    findings = evaluate_manifest_parity(manifest, live)

    if args.json:
        print(
            json.dumps({"owner": owner, "findings": findings}, indent=2, sort_keys=True)
        )
    else:
        _render_human(manifest, findings)

    # REPORT-ONLY by default. --fail-on-findings is reserved for the future
    # enforcing follow-up and is OFF unless explicitly requested.
    blocking = [
        f
        for f in findings
        if f["class"] in {PARITY_MISSING, PARITY_NEEDS_CLOSURE, PARITY_UNPROTECTED}
    ]
    if args.fail_on_findings and blocking:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="Report required-context parity findings")
    p_report.add_argument(
        "--owner", default=None, help="GitHub org (default: manifest owner)"
    )
    p_report.add_argument(
        "--manifest",
        default=str(_DEFAULT_MANIFEST),
        help="Path to the parity manifest YAML",
    )
    p_report.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    p_report.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Exit non-zero on findings (RESERVED for the future enforcing gate; "
        "default OFF — this ratchet is report-only)",
    )
    p_report.set_defaults(func=cmd_report)

    ns = parser.parse_args()
    return int(ns.func(ns))


if __name__ == "__main__":
    sys.exit(main())
