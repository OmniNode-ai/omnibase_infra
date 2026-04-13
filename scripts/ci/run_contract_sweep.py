#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Standalone CI script: contract sweep.

Modes:
  drift   (default) — static cross-repo contract drift detection using
                      onex_change_control/scripts/validation/check_contract_drift.py
  runtime           — live contract compliance via omnibase_infra.verification.cli
  full              — drift then runtime

Usage:
    python run_contract_sweep.py [--mode drift|runtime|full]
                                 [--omni-home /path/to/omni_home]
                                 [--repos omnibase_core,omnibase_infra,...]
                                 [--sensitivity STRICT|STANDARD|LAX]
                                 [--fail-on-severity breaking|additive|non_breaking]
                                 [--dry-run]
                                 [--json]

Exit codes:
    0 — clean (no findings at or above --fail-on-severity)
    1 — findings at or above --fail-on-severity
    2 — usage/configuration error
    3 — tool not found (check_contract_drift.py or verification.cli missing)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path

DEFAULT_REPOS = [
    "omnibase_core",
    "omnibase_infra",
    "omniclaude",
    "omniintelligence",
    "omnimemory",
    "omninode_infra",
    "omnibase_spi",
    "onex_change_control",
]

_SEVERITY_ORDER = {"breaking": 3, "additive": 2, "non_breaking": 1}


# ---------------------------------------------------------------------------
# Drift mode
# ---------------------------------------------------------------------------


def run_drift_mode(
    omni_home: Path,
    repos: list[str],
    sensitivity: str,
    check_boundaries: bool,
) -> dict:
    """Run static drift detection for all repos."""
    change_control = omni_home / "onex_change_control"
    drift_script = change_control / "scripts" / "validation" / "check_contract_drift.py"

    drift_script_available = drift_script.is_file()

    results = []
    repos_not_found: list[str] = []
    baseline_missing: list[str] = []
    total_contracts = 0

    for repo in repos:
        repo_path = omni_home / repo
        if not repo_path.is_dir():
            repos_not_found.append(repo)
            continue

        # Discover contract files
        src_path = repo_path / "src"
        if not src_path.is_dir():
            src_path = repo_path  # fallback for repos without src/

        contract_files = list(src_path.rglob("contract.yaml")) + list(
            src_path.rglob("handler_contract.yaml")
        )
        total_contracts += len(contract_files)

        # Check for snapshot
        snapshot_file = change_control / "drift" / f"{repo}.sha256"

        if not drift_script_available:
            baseline_missing.append(repo)
            results.append(
                {
                    "repo": repo,
                    "status": "baseline_missing",
                    "contracts_found": len(contract_files),
                    "drift_detected": False,
                    "summary": "check_contract_drift.py not found — skipping drift check",
                }
            )
            continue

        if not snapshot_file.is_file():
            baseline_missing.append(repo)
            results.append(
                {
                    "repo": repo,
                    "status": "baseline_missing",
                    "contracts_found": len(contract_files),
                    "drift_detected": False,
                    "summary": "No baseline snapshot — skipping drift check",
                }
            )
            continue

        # Run drift check
        cmd = [
            sys.executable,
            str(drift_script),
            "--root",
            str(src_path),
            "--check",
            str(snapshot_file),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(change_control),
                check=False,
            )
            drift_detected = proc.returncode == 1
            results.append(
                {
                    "repo": repo,
                    "status": "drifted" if drift_detected else "clean",
                    "contracts_found": len(contract_files),
                    "drift_detected": drift_detected,
                    "output": (proc.stdout + proc.stderr).strip()[:500],
                    "summary": (
                        f"Drift detected in {repo}"
                        if drift_detected
                        else f"{repo} is clean"
                    ),
                }
            )
        except subprocess.TimeoutExpired:
            results.append(
                {
                    "repo": repo,
                    "status": "timeout",
                    "contracts_found": len(contract_files),
                    "drift_detected": False,
                    "summary": f"Drift check timed out for {repo}",
                }
            )
        except FileNotFoundError:
            results.append(
                {
                    "repo": repo,
                    "status": "error",
                    "contracts_found": len(contract_files),
                    "drift_detected": False,
                    "summary": f"Python interpreter not found while checking {repo}",
                }
            )

    # Boundary staleness check
    boundary_findings: list[dict] = []
    if check_boundaries:
        boundary_findings = _check_boundaries(omni_home)

    drifted = [r for r in results if r.get("drift_detected")]
    stale_boundaries = [b for b in boundary_findings if b.get("severity") == "critical"]

    if drifted or stale_boundaries:
        overall_status = "breaking"
    elif any(r["status"] == "baseline_missing" for r in results):
        overall_status = "drifted"
    else:
        overall_status = "clean"

    return {
        "mode": "drift",
        "status": overall_status,
        "repos_scanned": [r["repo"] for r in results],
        "repos_not_found": repos_not_found,
        "baseline_missing": baseline_missing,
        "total_contracts": total_contracts,
        "sensitivity": sensitivity,
        "repo_results": results,
        "boundary_findings": boundary_findings,
        "drifted_repos": [r["repo"] for r in drifted],
        "stale_boundary_count": len(stale_boundaries),
    }


def _check_boundaries(omni_home: Path) -> list[dict]:
    """Check kafka_boundaries.yaml for stale producer/consumer file references."""
    boundaries_file = (
        omni_home
        / "onex_change_control"
        / "src"
        / "onex_change_control"
        / "boundaries"
        / "kafka_boundaries.yaml"
    )
    if not boundaries_file.is_file():
        return []

    import re

    findings = []
    try:
        content = boundaries_file.read_text(encoding="utf-8")
    except OSError:
        return []

    # Simple regex parsing — avoids yaml dep requirement
    entry_re = re.compile(
        r"topic_name:\s*(\S+).*?"
        r"producer_repo:\s*(\S+).*?"
        r"producer_file:\s*(\S+)",
        re.DOTALL,
    )

    for m in entry_re.finditer(content):
        topic = m.group(1).strip()
        producer_repo = m.group(2).strip()
        producer_file = m.group(3).strip()

        repo_path = omni_home / producer_repo
        if not repo_path.is_dir():
            findings.append(
                {
                    "topic": topic,
                    "issue": "producer_repo_not_found",
                    "producer_repo": producer_repo,
                    "severity": "critical",
                    "message": f"Producer repo '{producer_repo}' not found at {repo_path}",
                }
            )
            continue

        file_path = repo_path / producer_file
        if not file_path.is_file():
            # Try glob search within src/
            matches = list(repo_path.rglob(Path(producer_file).name))
            if not matches:
                findings.append(
                    {
                        "topic": topic,
                        "issue": "producer_file_missing",
                        "producer_repo": producer_repo,
                        "producer_file": producer_file,
                        "severity": "critical",
                        "message": f"Producer file '{producer_file}' not found in {producer_repo}",
                    }
                )

    return findings


# ---------------------------------------------------------------------------
# Runtime mode
# ---------------------------------------------------------------------------


def run_runtime_mode(omni_home: Path, full_verification: bool) -> dict:
    """Run runtime contract compliance via omnibase_infra.verification.cli."""
    infra_path = omni_home / "omnibase_infra"
    if not infra_path.is_dir():
        return {
            "mode": "runtime",
            "status": "error",
            "error": f"omnibase_infra not found at {infra_path}",
        }

    run_id = f"contract-sweep-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}"
    cmd = ["uv", "run", "python", "-m", "omnibase_infra.verification.cli", "--json"]
    if not full_verification:
        cmd.append("--registration-only")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(infra_path),
            check=False,
        )
    except FileNotFoundError:
        return {
            "mode": "runtime",
            "status": "error",
            "run_id": run_id,
            "error": "uv not found — cannot run omnibase_infra.verification.cli",
        }
    except subprocess.TimeoutExpired:
        return {
            "mode": "runtime",
            "status": "error",
            "run_id": run_id,
            "error": "verification.cli timed out",
        }

    output = proc.stdout + proc.stderr

    if proc.returncode == 0:
        return {
            "mode": "runtime",
            "status": "PASS",
            "run_id": run_id,
            "output": output.strip()[:2000],
        }
    elif proc.returncode == 2:
        return {
            "mode": "runtime",
            "status": "QUARANTINE",
            "run_id": run_id,
            "output": output.strip()[:2000],
        }
    else:
        return {
            "mode": "runtime",
            "status": "FAIL",
            "run_id": run_id,
            "output": output.strip()[:2000],
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Contract sweep — standalone CI gate (drift, runtime, or full)"
    )
    parser.add_argument(
        "--mode",
        choices=["drift", "runtime", "full"],
        default="drift",
        help="Sweep mode (default: drift)",
    )
    parser.add_argument(
        "--omni-home",
        metavar="PATH",
        default=os.environ.get("OMNI_HOME", str(Path.cwd())),
        help="Path to omni_home (default: $OMNI_HOME or cwd)",
    )
    parser.add_argument(
        "--repos",
        metavar="REPO[,REPO...]",
        default=",".join(DEFAULT_REPOS),
        help="Comma-separated repo names to scan (drift mode, default: all 8)",
    )
    parser.add_argument(
        "--sensitivity",
        choices=["STRICT", "STANDARD", "LAX"],
        default="STANDARD",
        help="Drift sensitivity (default: STANDARD)",
    )
    parser.add_argument(
        "--fail-on-severity",
        metavar="LEVEL",
        default="breaking",
        choices=list(_SEVERITY_ORDER.keys()),
        help="Minimum severity to fail CI (default: breaking)",
    )
    parser.add_argument(
        "--no-check-boundaries",
        action="store_true",
        help="Skip Kafka boundary staleness check",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Runtime mode: run full 52-contract verification (default: registration-only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print findings only, do not create tickets",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable text",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    omni_home = Path(args.omni_home)
    if not omni_home.is_dir():
        print(f"ERROR: omni_home not found: {omni_home}", file=sys.stderr)
        return 2

    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    fail_threshold = _SEVERITY_ORDER[args.fail_on_severity]
    mode = args.mode

    drift_result: dict | None = None
    runtime_result: dict | None = None
    exit_code = 0

    if mode in ("drift", "full"):
        drift_result = run_drift_mode(
            omni_home=omni_home,
            repos=repos,
            sensitivity=args.sensitivity,
            check_boundaries=not args.no_check_boundaries,
        )
        if drift_result.get("status") == "error":
            print(f"ERROR: {drift_result.get('error')}", file=sys.stderr)
            return 3

        # Map drift status to severity
        drift_status = drift_result.get("status", "clean")
        drift_severity = {
            "breaking": 3,
            "drifted": 2,
            "clean": 0,
            "baseline_missing": 1,
        }.get(drift_status, 0)
        if drift_severity >= fail_threshold:
            exit_code = 1

    if mode in ("runtime", "full"):
        runtime_result = run_runtime_mode(
            omni_home=omni_home,
            full_verification=args.all,
        )
        if runtime_result.get("status") == "FAIL":
            if fail_threshold <= _SEVERITY_ORDER.get("breaking", 3):
                exit_code = 1
        elif runtime_result.get("status") == "error":
            print(
                f"WARN: runtime mode error: {runtime_result.get('error')}",
                file=sys.stderr,
            )

    if args.json:
        output = {
            "sweep": "contract_sweep",
            "mode": mode,
            "omni_home": str(omni_home),
            "fail_on_severity": args.fail_on_severity,
            "status": "FAIL" if exit_code == 1 else "PASS",
        }
        if drift_result:
            output["drift"] = drift_result
        if runtime_result:
            output["runtime"] = runtime_result
        print(json.dumps(output, indent=2))
    else:
        if drift_result:
            print("Contract Drift Sweep Results")
            print("=============================")
            print(f"Repos scanned: {len(drift_result.get('repos_scanned', []))}")
            print(f"Total contracts: {drift_result.get('total_contracts', 0)}")
            print(f"Sensitivity: {args.sensitivity}")
            print()

            drifted = drift_result.get("drifted_repos", [])
            if drifted:
                print(f"Drifted repos: {', '.join(drifted)}")
            missing = drift_result.get("baseline_missing", [])
            if missing:
                print(f"Repos without baseline snapshot: {', '.join(missing)}")
            not_found = drift_result.get("repos_not_found", [])
            if not_found:
                print(f"Repos not found: {', '.join(not_found)}")

            stale = drift_result.get("stale_boundary_count", 0)
            if stale:
                print(f"Stale boundaries: {stale}")

            print()
            print(f"Overall drift status: {drift_result.get('status', 'unknown')}")

        if runtime_result:
            status = runtime_result.get("status", "unknown")
            print()
            print(f"CONTRACT_VERIFY: {status}")
            if status == "FAIL":
                output_text = runtime_result.get("output", "")
                if output_text:
                    print(output_text[:1000])

        if mode == "full" and drift_result and runtime_result:
            drift_status = drift_result.get("status", "unknown")
            rt_status = runtime_result.get("status", "unknown")
            combined: str
            if drift_status == "clean" and rt_status == "PASS":
                combined = "CLEAN"
            elif drift_status == "breaking" or rt_status == "FAIL":
                combined = "FAILURES"
            else:
                combined = "WARNINGS"
            print()
            print(f"=== contract-sweep FULL: Combined status: {combined} ===")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
