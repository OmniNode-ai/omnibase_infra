#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Open or refresh one lockfile-only PR for base-inherited OSV blockers (OMN-20174).

Classification belongs to check_lockfile_cve; this remediation never changes
that gate. All external commands use the injected Runner, and writes require
an explicit writer-App GH_TOKEN. The scheduled workflow checks out only the
repository's default branch before scanning it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from scripts.ci.check_lockfile_cve import evaluate_osv_results

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner_image_identity

Runner = Callable[[Sequence[str], Path | None], subprocess.CompletedProcess[str]]
BRANCH = "bot/lockfile-cve-autobump"
RUNNER_LOCK = "docker/runners/runner-image.lock.json"
LABELS = (
    (
        "priority:landing",
        "B60205",
        "Priority landing for default-branch CVE remediation",
    ),
    ("drain:priority", "D93F0B", "Priority drain for default-branch CVE remediation"),
)


@dataclass(frozen=True)
class ModelBumpItem:
    """One package's minimum target satisfying every blocking advisory."""

    package: str
    current_version: str
    target_version: str
    vuln_ids: tuple[str, ...]


class ModelPlanError(ValueError):
    """A blocking package cannot be mapped to an actionable fixed version."""


@dataclass(frozen=True)
class ModelAutobumpConfig:
    """Inputs shared by the CLI and the testable remediation entrypoint."""

    osv_json: Path
    repo: str
    event_name: str
    ref: str
    default_branch: str
    repo_root: Path
    ticket: str = "OMN-20174"
    dry_run: bool = False
    allow_non_default_base: bool = False
    base_branch: str | None = None


class ModelCommandError(RuntimeError):
    """An external command failed; publication must stop."""


def compute_bump_plan(osv_json: dict[str, Any]) -> tuple[ModelBumpItem, ...]:
    """Select the smallest forward fix per blocker, then the maximum per package."""
    blocking = evaluate_osv_results(osv_json).blocking_findings
    raw_vulns: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for result in osv_json.get("results") or []:
        for entry in result.get("packages") or []:
            package = entry.get("package") or {}
            name = canonicalize_name(str(package.get("name", "")))
            version = str(package.get("version", ""))
            for vuln in entry.get("vulnerabilities") or []:
                key = (name, version, str(vuln.get("id", "")))
                raw_vulns.setdefault(key, []).append(vuln)

    items: dict[str, ModelBumpItem] = {}
    for finding in blocking:
        name = canonicalize_name(finding.package_name)
        try:
            current = Version(finding.package_version)
        except InvalidVersion as exc:
            raise ModelPlanError(
                f"{name}: invalid locked version {finding.package_version}"
            ) from exc
        fixes: set[Version] = set()
        for vuln in raw_vulns.get((name, finding.package_version, finding.vuln_id), []):
            for affected in vuln.get("affected") or []:
                package = affected.get("package") or {}
                if canonicalize_name(str(package.get("name", ""))) != name:
                    continue
                if package.get("ecosystem") not in (None, "PyPI"):
                    continue
                for affected_range in affected.get("ranges") or []:
                    for event in affected_range.get("events") or []:
                        if "fixed" not in event:
                            continue
                        try:
                            fixed = Version(str(event["fixed"]))
                        except InvalidVersion:
                            continue
                        if fixed > current:
                            fixes.add(fixed)
        if not fixes:
            raise ModelPlanError(
                f"{name}: no determinable fixed version > {current} for {finding.vuln_id}"
            )
        target = min(fixes)
        previous = items.get(name)
        ids = {finding.vuln_id}
        if previous:
            target = max(target, Version(previous.target_version))
            current = max(current, Version(previous.current_version))
            ids.update(previous.vuln_ids)
        items[name] = ModelBumpItem(name, str(current), str(target), tuple(sorted(ids)))
    return tuple(items[name] for name in sorted(items))


def is_base_inherited_run(event_name: str, ref: str, default_branch: str) -> bool:
    """Allow only trusted base events on the exact default-branch ref."""
    return (
        event_name in {"schedule", "push", "workflow_dispatch"}
        and ref == f"refs/heads/{default_branch}"
    )


def default_runner(
    cmd: Sequence[str], cwd: Path | None
) -> subprocess.CompletedProcess[str]:
    """Execute an argv without a shell, leaving failure handling to the caller."""
    return subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)


def _execute(
    runner: Runner, cfg: ModelAutobumpConfig, cmd: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    result = runner(cmd, cfg.repo_root)
    if result.returncode:
        raise ModelCommandError(
            f"{' '.join(cmd)} failed ({result.returncode}): {result.stderr or result.stdout}"
        )
    return result


def _open_prs(
    cfg: ModelAutobumpConfig, runner: Runner, branch: str
) -> list[dict[str, Any]]:
    result = _execute(
        runner,
        cfg,
        [
            "gh",
            "pr",
            "list",
            "--repo",
            cfg.repo,
            "--head",
            branch,
            "--state",
            "open",
            "--json",
            "number,url,headRefOid",
        ],
    )
    data = json.loads(result.stdout)
    if not isinstance(data, list) or len(data) > 1:
        raise ModelCommandError("Expected at most one open lockfile bump PR")
    return data


def _print_plan(plan: tuple[ModelBumpItem, ...]) -> None:
    for item in plan:
        print(
            f"PLAN: {item.package} {item.current_version} -> {item.target_version} ({', '.join(item.vuln_ids)})"
        )


def _body(cfg: ModelAutobumpConfig, plan: tuple[ModelBumpItem, ...]) -> str:
    lines = [
        f"Ticket: {cfg.ticket}.",
        "",
        "| Package | Locked version | Fixed version | Advisory ids |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {item.package} | {item.current_version} | {item.target_version} | {', '.join(item.vuln_ids)} |"
        for item in plan
    )
    lines.extend(
        [
            "",
            "Opened by the lockfile-cve-autobump workflow because the default branch itself carries these advisories (base-inherited). The diff is uv.lock plus the regenerated runner image lock only. The Lockfile CVE Scan is unchanged.",
        ]
    )
    return "\n".join(lines)


def _ensure_labels(cfg: ModelAutobumpConfig, runner: Runner) -> None:
    for name, color, description in LABELS:
        result = _execute(
            runner,
            cfg,
            [
                "gh",
                "label",
                "list",
                "--repo",
                cfg.repo,
                "--search",
                name,
                "--json",
                "name",
            ],
        )
        # `gh label list --json` prints nothing at all (not `[]`) on zero matches.
        labels = json.loads(result.stdout) if result.stdout.strip() else []
        if not any(label.get("name") == name for label in labels):
            _execute(
                runner,
                cfg,
                [
                    "gh",
                    "label",
                    "create",
                    name,
                    "--repo",
                    cfg.repo,
                    "--color",
                    color,
                    "--description",
                    description,
                ],
            )


def _verify_lock(cfg: ModelAutobumpConfig, plan: tuple[ModelBumpItem, ...]) -> None:
    with (cfg.repo_root / "uv.lock").open("rb") as stream:
        lock = tomllib.load(stream)
    for item in plan:
        versions = [
            Version(str(package["version"]))
            for package in lock.get("package", [])
            if canonicalize_name(str(package.get("name", ""))) == item.package
        ]
        if not versions or any(
            version < Version(item.target_version) for version in versions
        ):
            raise ModelPlanError(
                f"{item.package}: uv.lock did not reach target {item.target_version}"
            )


def run_autobump(cfg: ModelAutobumpConfig, runner: Runner = default_runner) -> int:
    """Remediate base blockers with one PR, or report a fail-closed error."""
    if not is_base_inherited_run(cfg.event_name, cfg.ref, cfg.default_branch):
        print("not a base-inherited run; no action")
        return 0
    try:
        payload = json.loads(cfg.osv_json.read_text(encoding="utf-8"))
        if not evaluate_osv_results(payload).blocking_findings:
            print("no blocking findings")
            return 0
        if not cfg.dry_run and not os.environ.get("GH_TOKEN", "").strip():
            print(
                "::error::GH_TOKEN writer-App installation token is required (OMN-18273)"
            )
            return 2
        base = cfg.base_branch or cfg.default_branch
        if base != cfg.default_branch and not cfg.allow_non_default_base:
            print("::error::base branch must equal the default branch")
            return 2
        branch = BRANCH
        if cfg.allow_non_default_base:
            branch += "-" + re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
        plan = compute_bump_plan(payload)
        packages = ", ".join(item.package for item in plan)
        title = f"fix({cfg.ticket}): bump {packages} to fixed versions for lockfile CVE advisories"
        if cfg.dry_run:
            _print_plan(plan)
            for item in plan:
                print(f"uv lock --upgrade-package {item.package}")
            print(f"WOULD use branch: {branch}")
            print(f"WOULD use PR title: {title}")
            prs = _open_prs(cfg, runner, branch)
            print(f"Open bump PR: {prs[0]['number'] if prs else 'none'}")
            return 0

        _ensure_labels(cfg, runner)
        _execute(runner, cfg, ["git", "fetch", "origin", base])
        _execute(runner, cfg, ["git", "checkout", "-B", branch, f"origin/{base}"])
        for item in plan:
            _execute(runner, cfg, ["uv", "lock", "--upgrade-package", item.package])
        _verify_lock(cfg, plan)
        paths = ["uv.lock"]
        lock_path = cfg.repo_root / RUNNER_LOCK
        if lock_path.exists():
            if runner_image_identity.generate_lock(cfg.repo_root, lock_path):
                raise ModelCommandError("runner image lock regeneration failed")
            paths.append(RUNNER_LOCK)
        _execute(
            runner,
            cfg,
            [
                "uv",
                "run",
                "python",
                "-m",
                "scripts.ci.check_lockfile_registry_allowlist",
                "uv.lock",
                "--min-packages",
                "1",
            ],
        )
        diff = runner(
            ["git", "diff", "--quiet", f"origin/{base}", "--", "uv.lock"], cfg.repo_root
        )
        if diff.returncode == 0:
            print("uv.lock unchanged; no action")
            return 0
        if diff.returncode != 1:
            raise ModelCommandError(f"git diff failed: {diff.stderr}")
        _execute(runner, cfg, ["git", "add", *paths])
        _execute(
            runner,
            cfg,
            [
                "git",
                "-c",
                "user.name=onexbot-occ-writer[bot]",
                "-c",
                "user.email=onexbot-occ-writer[bot]@users.noreply.github.com",
                "commit",
                "-m",
                f"fix({cfg.ticket}): bump {packages} for lockfile CVE advisories [bot]",
                "--only",
                "--",
                *paths,
            ],
        )
        prs = _open_prs(cfg, runner, branch)
        body = _body(cfg, plan)
        if prs:
            pr = prs[0]
            _execute(
                runner,
                cfg,
                [
                    "git",
                    "fetch",
                    "origin",
                    f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
                ],
            )
            new_tree = _execute(
                runner, cfg, ["git", "rev-parse", "HEAD^{tree}"]
            ).stdout.strip()
            old_tree = _execute(
                runner, cfg, ["git", "rev-parse", f"origin/{branch}^{{tree}}"]
            ).stdout.strip()
            if new_tree == old_tree:
                print("already current")
                return 0
            _execute(
                runner,
                cfg,
                [
                    "git",
                    "push",
                    f"--force-with-lease={branch}:{pr['headRefOid']}",
                    "origin",
                    f"HEAD:refs/heads/{branch}",
                ],
            )
            number = str(pr["number"])
            edit = ["--title", title, "--body", body]
        else:
            remote = _execute(
                runner,
                cfg,
                ["git", "ls-remote", "--heads", "origin", f"refs/heads/{branch}"],
            )
            lease: list[str] = []
            if remote.stdout.strip():
                _execute(
                    runner,
                    cfg,
                    [
                        "git",
                        "fetch",
                        "origin",
                        f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
                    ],
                )
                lease = ["--force-with-lease"]
            _execute(
                runner,
                cfg,
                ["git", "push", *lease, "origin", f"HEAD:refs/heads/{branch}"],
            )
            created = _execute(
                runner,
                cfg,
                [
                    "gh",
                    "pr",
                    "create",
                    "--repo",
                    cfg.repo,
                    "--base",
                    base,
                    "--head",
                    branch,
                    "--title",
                    title,
                    "--body",
                    body,
                ],
            )
            number = created.stdout.strip().rsplit("/", 1)[-1]
            if not number.isdigit():
                raise ModelCommandError(
                    f"Cannot determine created PR number: {created.stdout}"
                )
            edit = []
        _execute(
            runner,
            cfg,
            [
                "gh",
                "pr",
                "edit",
                number,
                "--repo",
                cfg.repo,
                *edit,
                "--add-label",
                "priority:landing",
                "--add-label",
                "drain:priority",
            ],
        )
        print(f"PR #{number}: published {branch}")
        return 0
    except (
        ModelPlanError,
        ModelCommandError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        print(f"::error::{exc}")
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Read an OSV payload for a side-effect-free plan or guarded remediation."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser(
        "plan", help="Print blocking advisory fixes without side effects"
    )
    plan_parser.add_argument("--osv-json", type=Path, required=True)
    run_parser = sub.add_parser(
        "run", help="Open or refresh the base-inherited bump PR"
    )
    run_parser.add_argument("--osv-json", type=Path, required=True)
    for flag in ("repo", "event-name", "ref", "default-branch"):
        run_parser.add_argument(f"--{flag}", required=True)
    run_parser.add_argument("--ticket", default="OMN-20174")
    run_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--allow-non-default-base", action="store_true")
    run_parser.add_argument(
        "--base-branch", help="Operator proof base; requires --allow-non-default-base"
    )
    args = parser.parse_args(argv)
    if args.command == "plan":
        try:
            _print_plan(
                compute_bump_plan(json.loads(args.osv_json.read_text(encoding="utf-8")))
            )
            return 0
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"::error::{exc}")
            return 1
    return run_autobump(
        ModelAutobumpConfig(
            osv_json=args.osv_json.resolve(),
            repo=args.repo,
            event_name=args.event_name,
            ref=args.ref,
            default_branch=args.default_branch,
            ticket=args.ticket,
            repo_root=args.repo_root.resolve(),
            dry_run=args.dry_run,
            allow_non_default_base=args.allow_non_default_base,
            base_branch=args.base_branch,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
