#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""RT-7 fail-loud release/deploy drift monitor -- I/O shell (OMN-14466).

Gathers, for every monitored OmniNode repo, the facts the pure evaluator
(``release_drift_monitor_lib.py``) needs -- PyPI-published version, latest ``v*``
tag, ``main``/``dev`` pyproject versions + ``[tool.uv.sources]`` git overrides,
dev-ahead-of-main commit count, and the last ``release.yml`` /
``runtime-rebuild-trigger.yml`` run -- via the GitHub REST API (``gh api``) and
the PyPI JSON API. It then evaluates drift and, on divergence, writes a friction
record into the existing friction registry (``.onex_state/friction/``, consumed
by ``node_friction_triage_orchestrator`` -> Linear) and exits non-zero so a
scheduled run goes RED.

This is a **standalone monitor** (RT-7 seam): it does not modify
``stage_workspace.sh``, ``deploy-runtime.sh``, ``release.yml``, or the trigger
scripts. It reuses the existing friction substrate -- it does NOT build a new
store (design §4 RT-7). The friction-emit shape mirrors
``node_env_sync_alert_effect``, the platform's existing scheduled drift ->
friction -> Linear precedent.

SCOPE NOTE (per-lane running-signature): the design also calls for a per-lane
"running-signature vs intended-ref" check. That requires reading a live ``.201``
container signature (SSH / introspection endpoint) and is delivered by the RT-6
deploy-readback lane; it is out of band for a CI-schedulable monitor. RT-7 here
covers the repo/PyPI/tag/workflow axes that ARE observable from CI, which is
what catches the six-week PyPI-stuck class. No fake lane check is emitted.

Usage::

    uv run python scripts/release_drift_monitor.py               # human report, exit 1 on drift
    uv run python scripts/release_drift_monitor.py --json        # machine-readable report
    uv run python scripts/release_drift_monitor.py --no-emit     # do not write friction files
    uv run python scripts/release_drift_monitor.py --warn-only   # always exit 0 (report only)

Exit codes:
    0 -- clean: every fact gathered, nothing diverged
    1 -- divergence detected (RED, the fail-loud signal)
    2 -- the monitor could not gather some facts and found no divergence
         (a blind run, surfaced explicitly -- never silently green)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from release_drift_monitor_lib import (
    DriftReport,
    DriftThresholds,
    RepoFacts,
    WorkflowRun,
    evaluate_all,
    exit_code_for,
)

_ORG = "OmniNode-ai"
_PYPI_TIMEOUT = 15
_RELEASE_WORKFLOW = "release.yml"
_TRIGGER_WORKFLOW = "runtime-rebuild-trigger.yml"


@dataclass(frozen=True)
class RepoConfig:
    """A repo to monitor and the PyPI package it publishes (None if not published)."""

    repo: str
    pypi_package: str | None


# Default monitor set. onex_change_control publishes nothing to PyPI, so its
# PyPI-based checks are skipped (pypi_package=None) -- only override-drift and
# workflow health apply there.
DEFAULT_REPOS: tuple[RepoConfig, ...] = (
    RepoConfig("omnibase_core", "omnibase-core"),
    RepoConfig("omnibase_infra", "omnibase-infra"),
    RepoConfig("omnibase_spi", "omnibase-spi"),
    RepoConfig("omnibase_compat", "omnibase-compat"),
    RepoConfig("omnimarket", "omnimarket"),
    RepoConfig("onex_change_control", None),
)


# --------------------------------------------------------------------------- #
# GitHub / PyPI probes. Every probe returns (value, error_message | None) so a  #
# gather failure is carried as a probe_error rather than silently dropped.      #
# --------------------------------------------------------------------------- #
def _gh_api(path: str, jq: str | None = None) -> tuple[Any, str | None]:
    """Call ``gh api <path>`` (optionally with a --jq filter). Returns (data, error)."""
    args = ["gh", "api", path]
    if jq is not None:
        args += ["--jq", jq]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None, "gh CLI not found"
    if proc.returncode != 0:
        stderr = proc.stderr.strip().splitlines()
        detail = stderr[-1] if stderr else f"exit {proc.returncode}"
        return None, detail
    out = proc.stdout.strip()
    if not out:
        return None, None
    if jq is not None:
        # --jq output is raw text/lines, not necessarily JSON.
        return out, None
    try:
        return json.loads(out), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON from gh api {path}: {exc}"


def _pypi_latest(package: str) -> tuple[str | None, str | None]:
    """Latest published version on PyPI. (None, None) if the package is absent."""
    url = f"https://pypi.org/pypi/{package}/json"  # url-authority-ok: PyPI public JSON API, fixed canonical third-party endpoint (no ONEX routing authority)
    try:
        with urllib.request.urlopen(url, timeout=_PYPI_TIMEOUT) as resp:  # noqa: S310 (https literal)
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None, f"PyPI package {package} not found (404)"
        return None, f"PyPI HTTP {exc.code} for {package}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return None, f"PyPI probe failed for {package}: {exc}"
    return data.get("info", {}).get("version"), None


def _latest_tag(repo: str) -> tuple[str | None, str | None]:
    """Highest semver ``v*`` tag for a repo (semver-sorted in Python)."""
    from release_drift_monitor_lib import parse_version

    data, err = _gh_api(f"repos/{_ORG}/{repo}/tags?per_page=100", jq=".[].name")
    if err is not None:
        return None, f"tags: {err}"
    if not data:
        return None, None
    best_raw: str | None = None
    best_ver = None
    for name in str(data).splitlines():
        name = name.strip()
        ver = parse_version(name)
        if ver is None:
            continue
        if best_ver is None or ver > best_ver:
            best_ver = ver
            best_raw = name
    return best_raw, None


def _pyproject_at(
    repo: str, ref: str
) -> tuple[str | None, tuple[str, ...], str | None]:
    """Return (project.version, git-override package names, error) for pyproject at ref."""
    data, err = _gh_api(
        f"repos/{_ORG}/{repo}/contents/pyproject.toml?ref={ref}", jq=".content"
    )
    if err is not None:
        return None, (), f"pyproject@{ref}: {err}"
    if not data:
        return None, (), f"pyproject@{ref}: empty response"
    try:
        raw = base64.b64decode(str(data)).decode("utf-8")
        parsed = tomllib.loads(raw)
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        return None, (), f"pyproject@{ref}: decode failed: {exc}"
    version = parsed.get("project", {}).get("version")
    overrides = _git_overrides(parsed)
    return (str(version) if version is not None else None), overrides, None


def _git_overrides(pyproject: dict[str, Any]) -> tuple[str, ...]:
    """Package names in [tool.uv.sources] that pin a git source (git= or rev=)."""
    sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    names: list[str] = []
    for pkg, spec in sources.items():
        if isinstance(spec, dict) and ("git" in spec or "rev" in spec):
            names.append(pkg)
    return tuple(sorted(names))


def _dev_ahead(repo: str) -> tuple[int | None, str | None]:
    """Commits ``dev`` is ahead of ``main`` (GitHub compare ahead_by)."""
    data, err = _gh_api(f"repos/{_ORG}/{repo}/compare/main...dev", jq=".ahead_by")
    if err is not None:
        return None, f"compare main...dev: {err}"
    if data is None or str(data).strip() == "":
        return None, None
    try:
        return int(str(data).strip()), None
    except ValueError:
        return None, f"compare main...dev: non-int ahead_by {data!r}"


def _workflow_last_run(
    repo: str, workflow: str
) -> tuple[WorkflowRun | None, str | None]:
    """Last run of a workflow (any branch). exists=False when the workflow is absent."""
    data, err = _gh_api(
        f"repos/{_ORG}/{repo}/actions/workflows/{workflow}/runs?per_page=1",
        jq=".workflow_runs[0] | {conclusion, created_at, status}",
    )
    if err is not None:
        if "Not Found" in err or "404" in err:
            return WorkflowRun(name=workflow, exists=False), None
        return None, f"{workflow} runs: {err}"
    if not data:
        # Workflow exists but has no runs.
        return WorkflowRun(name=workflow, exists=True, conclusion=None), None
    try:
        parsed = json.loads(str(data))
    except json.JSONDecodeError as exc:
        return None, f"{workflow} runs: invalid JSON: {exc}"
    return (
        WorkflowRun(
            name=workflow,
            exists=True,
            conclusion=parsed.get("conclusion"),
            created_at=parsed.get("created_at"),
        ),
        None,
    )


def gather_repo_facts(cfg: RepoConfig) -> RepoFacts:
    """Gather all monitored facts for one repo, accumulating any probe errors."""
    errors: list[str] = []

    pypi_version: str | None = None
    if cfg.pypi_package is not None:
        pypi_version, err = _pypi_latest(cfg.pypi_package)
        if err is not None:
            errors.append(err)

    latest_tag, err = _latest_tag(cfg.repo)
    if err is not None:
        errors.append(err)

    main_version, main_overrides, err = _pyproject_at(cfg.repo, "main")
    if err is not None:
        errors.append(err)
    dev_version, dev_overrides, err = _pyproject_at(cfg.repo, "dev")
    if err is not None:
        errors.append(err)

    dev_ahead, err = _dev_ahead(cfg.repo)
    if err is not None:
        errors.append(err)

    release_run, err = _workflow_last_run(cfg.repo, _RELEASE_WORKFLOW)
    if err is not None:
        errors.append(err)
    trigger_run, err = _workflow_last_run(cfg.repo, _TRIGGER_WORKFLOW)
    if err is not None:
        errors.append(err)

    return RepoFacts(
        repo=cfg.repo,
        pypi_package=cfg.pypi_package,
        pypi_version=pypi_version,
        latest_tag=latest_tag,
        main_version=main_version,
        dev_version=dev_version,
        dev_ahead_commits=dev_ahead,
        main_git_overrides=main_overrides,
        dev_git_overrides=dev_overrides,
        release_run=release_run,
        trigger_run=trigger_run,
        probe_errors=tuple(errors),
    )


# --------------------------------------------------------------------------- #
# Friction emission -- reuse the existing registry (no new store).             #
# --------------------------------------------------------------------------- #
def _resolve_friction_dir(explicit: str | None) -> Path:
    """Resolve the friction registry directory (mirrors node_env_sync_alert_effect)."""
    if explicit:
        return Path(explicit)
    state_dir = os.environ.get("ONEX_STATE_DIR")
    if state_dir:
        return Path(state_dir) / "friction"
    omni_home = os.environ.get("OMNI_HOME")
    if omni_home:
        return Path(omni_home) / ".onex_state" / "friction"
    return Path(".onex_state") / "friction"


def emit_friction(report: DriftReport, friction_dir: Path) -> list[str]:
    """Write one friction YAML per finding signature; return the paths written."""
    if not report.findings:
        return []
    friction_dir.mkdir(parents=True, exist_ok=True)
    occurred_at = datetime.now(tz=UTC).isoformat()
    written: list[str] = []
    for finding in report.findings:
        event = {
            "event_type": "release_deploy_drift",
            "occurred_at": occurred_at,
            "severity": finding.severity,
            "drift_signature": finding.signature,
            "code": finding.code,
            "repo": finding.repo,
            "title": f"Release/deploy drift ({finding.code}): {finding.summary}",
            "summary": finding.summary,
            "detail": finding.detail,
            "source": "release_drift_monitor",
        }
        safe = finding.signature.replace("/", "-").replace(":", "-")
        path = friction_dir / f"release-drift-{safe}.yaml"
        path.write_text(yaml.safe_dump(event, sort_keys=True), encoding="utf-8")
        written.append(str(path))
    return written


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #
def _render_human(report: DriftReport, friction_paths: list[str]) -> str:
    lines: list[str] = []
    lines.append("=" * 74)
    lines.append("RT-7 release/deploy drift monitor")
    lines.append(f"generated_at: {report.generated_at}")
    lines.append(f"repos_checked: {report.repos_checked}")
    lines.append("=" * 74)
    if report.findings:
        lines.append(f"DRIFT DETECTED -- {len(report.findings)} finding(s):")
        for finding in report.findings:
            lines.append("")
            lines.append(f"  [{finding.severity}] {finding.code} -- {finding.repo}")
            lines.append(f"      {finding.summary}")
            lines.append(f"      {finding.detail}")
    else:
        lines.append(
            "No drift detected -- PyPI/tag/main/dev aligned, workflows healthy."
        )
    if report.probe_errors:
        lines.append("")
        lines.append(
            f"PROBE ERRORS ({len(report.probe_errors)}) -- monitor visibility gaps:"
        )
        for err in report.probe_errors:
            lines.append(f"  ! {err}")
    if friction_paths:
        lines.append("")
        lines.append("friction records written:")
        for path in friction_paths:
            lines.append(f"  - {path}")
    lines.append("=" * 74)
    return "\n".join(lines)


def _report_to_dict(report: DriftReport, friction_paths: list[str]) -> dict[str, Any]:
    return {
        "generated_at": report.generated_at,
        "repos_checked": report.repos_checked,
        "diverged": report.diverged,
        "blind": report.blind,
        "findings": [
            {
                "code": f.code,
                "severity": f.severity,
                "repo": f.repo,
                "signature": f.signature,
                "summary": f.summary,
                "detail": f.detail,
            }
            for f in report.findings
        ],
        "probe_errors": list(report.probe_errors),
        "friction_paths": friction_paths,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RT-7 release/deploy drift monitor")
    parser.add_argument(
        "--json", action="store_true", help="machine-readable JSON output"
    )
    parser.add_argument(
        "--no-emit", action="store_true", help="do not write friction records"
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="always exit 0 (report drift without failing the run)",
    )
    parser.add_argument(
        "--friction-dir", default=None, help="override the friction registry directory"
    )
    parser.add_argument(
        "--dev-ahead-warn",
        type=int,
        default=DriftThresholds.dev_ahead_warn,
        help="dev-ahead-of-main commit count that fires DEV_AHEAD_OF_MAIN",
    )
    parser.add_argument(
        "--release-stale-days",
        type=float,
        default=DriftThresholds.release_stale_days,
        help="release-run age (main ahead of tag) that fires RELEASE_TRAIN_STALLED",
    )
    args = parser.parse_args(argv)

    thresholds = DriftThresholds(
        dev_ahead_warn=args.dev_ahead_warn,
        release_stale_days=args.release_stale_days,
    )

    facts = [gather_repo_facts(cfg) for cfg in DEFAULT_REPOS]
    report = evaluate_all(facts, thresholds)

    friction_paths: list[str] = []
    if not args.no_emit:
        friction_paths = emit_friction(report, _resolve_friction_dir(args.friction_dir))

    if args.json:
        print(json.dumps(_report_to_dict(report, friction_paths), indent=2))
    else:
        print(_render_human(report, friction_paths))

    if args.warn_only:
        return 0
    exit_code: int = exit_code_for(report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
