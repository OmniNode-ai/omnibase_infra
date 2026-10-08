#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Gated apply of a compose-config change to a governed lane (OMN-20260).

The ``stability-test`` and ``judge`` compose lanes are governed (CLAUDE.md rule
2a, OMN-15243): a mutation needs a CODEOWNERS-approved grant. Every grant kind
in ``prod_promotion_grants.yaml`` names an image digest or a kustomize overlay,
so a compose-config change (a loopback port bind, for one) had no grant that
could authorize it. The operator approved a new grant type on 2026-10-01
(RULING 2026-10-01T10:37:55Z lane=compose-grant-kind). Its anchor is
``onex_change_control`` ``grants/governed_lane_grants.yaml``, kind
``compose_config``; this script is the only path that consumes it.

``apply`` refuses (exit 1) unless ALL of these hold:

1. The anchor is read from ``onex_change_control@main`` (never a PR branch, and
   never a local file when ``--execute`` is given).
2. It holds an entry with ``--grant-id``, ``target_kind: compose_config``,
   ``runtime_lane`` equal to ``--lane`` (``stability-test`` or ``judge``), the
   lane's ``omnibase-infra-<lane>`` project, ``approved_by`` different from
   ``requested_by``, not consumed, and ``expires_at`` in the future.
3. This checkout's ``HEAD`` is the grant's ``compose_ref`` and none of the
   grant's compose files has an uncommitted change.
4. The digest of ``docker compose config`` rendered HERE, from exactly the
   grant's compose files, env files and profiles, equals the grant's
   ``rendered_digest``. The digest is recomputed in-process; no caller can
   assert it.

Only then, with ``--execute``, it runs the OMN-15218 lane-deploy attribution
preflight and recreates exactly the grant's services with ``up -d --no-deps
--no-build``. Without ``--execute`` it stops after the checks (exit 0 = would
apply).

``digest`` prints the canonical digest and ``HEAD`` for a set of inputs, which
is how a grant request is authored: run it on the lane host, at the commit the
grant will name, and copy the two values into the entry.

Canonical render: ``docker compose -p <project> [--env-file ...] -f ... [--profile
...] config --format json --no-path-resolution``, parsed as JSON and re-encoded
with sorted keys and compact separators; ``sha256:`` + hex of that. Path
resolution is off so the digest does not depend on where the checkout lives.
Interpolation is on, so the digest binds the env values the apply will use;
the rendered text is hashed and never printed. The env inputs are the same ones
``onex-runtime-deploy`` interpolates with: the grant's repo env files (normally
``docker/runtime-policy.env``), then the operator env file
(``$OMNIBASE_OPERATOR_ENV_FILE``, default ``~/.omnibase/.env``), which lives
outside the repo and is never named in a public grant. Run both ``digest`` and
``apply`` from a clean shell on the lane host: an exported variable outranks an
env file in compose interpolation, so a stray one changes the digest and the
apply refuses (fail closed).

Exit codes: 0 allowed / digest printed, 1 refused, 2 usage or environment error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

GOVERNED_COMPOSE_LANES: frozenset[str] = frozenset({"stability-test", "judge"})
TARGET_KIND = "compose_config"
ANCHOR_RELPATH = "grants/governed_lane_grants.yaml"
ANCHOR_REF = "main"
GIT_TIMEOUT_SECONDS = 60
COMPOSE_TIMEOUT_SECONDS = 600
REPO_ROOT = Path(__file__).resolve().parent.parent


class ApplyUsageError(RuntimeError):
    """Usage or environment error (exit 2), never a policy refusal."""


def canonical_digest(rendered: bytes) -> str:
    """sha256 over the canonical JSON re-encoding of a compose config render."""
    document = json.loads(rendered.decode("utf-8"))
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compose_argv(
    *,
    project: str,
    compose_files: Sequence[str],
    env_files: Sequence[str],
    profiles: Sequence[str],
    repo_root: Path,
    operator_env: Path,
) -> list[str]:
    """The ``docker compose`` prefix for exactly these inputs, in grant order."""
    argv = ["docker", "compose", "-p", project]
    for env_file in env_files:
        argv += ["--env-file", str(repo_root / env_file)]
    argv += ["--env-file", str(operator_env)]
    for compose_file in compose_files:
        argv += ["-f", str(repo_root / compose_file)]
    for profile in profiles:
        argv += ["--profile", profile]
    return argv


def render_digest(prefix: Sequence[str]) -> str:
    """Render the config with ``prefix`` and return its canonical digest."""
    completed = subprocess.run(
        [*prefix, "config", "--format", "json", "--no-path-resolution"],
        capture_output=True,
        timeout=COMPOSE_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        # stderr can name a missing variable; it never carries the rendered text.
        raise ApplyUsageError(
            "docker compose config failed: "
            + completed.stderr.decode(errors="replace").strip()
        )
    return canonical_digest(completed.stdout)


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    else:
        raise ValueError(f"not a timestamp: {value!r}")
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def match_grant(
    raw_anchor: bytes, *, grant_id: str, lane: str, now: datetime
) -> tuple[dict[str, Any] | None, list[str]]:
    """Find and check the grant. Pure: returns (entry, refusals)."""
    try:
        document = yaml.safe_load(raw_anchor.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        return None, [f"grant anchor does not parse: {exc}"]
    if not isinstance(document, dict) or not isinstance(document.get("entries"), list):
        return None, ["grant anchor has no 'entries' list"]
    matches = [
        e
        for e in document["entries"]
        if isinstance(e, dict) and e.get("grant_id") == grant_id
    ]
    if len(matches) != 1:
        return None, [
            f"grant {grant_id!r} is not in the anchor at @{ANCHOR_REF} "
            f"({len(matches)} matches). No grant, no apply."
        ]
    entry = matches[0]
    refusals: list[str] = []
    if entry.get("target_kind") != TARGET_KIND:
        refusals.append(
            f"target_kind is {entry.get('target_kind')!r}, not {TARGET_KIND!r}"
        )
    if lane not in GOVERNED_COMPOSE_LANES:
        refusals.append(f"--lane {lane!r} is not a governed compose lane")
    if entry.get("runtime_lane") != lane:
        refusals.append(
            f"grant is for lane {entry.get('runtime_lane')!r}, not {lane!r}"
        )
    if entry.get("compose_project") != f"omnibase-infra-{lane}":
        refusals.append(f"grant names project {entry.get('compose_project')!r}")
    approved, requested = entry.get("approved_by"), entry.get("requested_by")
    if (
        not isinstance(approved, str)
        or not isinstance(requested, str)
        or not approved.strip()
        or approved.casefold() == requested.casefold()
    ):
        refusals.append("approved_by is missing or equals requested_by (self_granted)")
    if entry.get("consumed") is True:
        refusals.append("grant is consumed; a spent grant is never replayed")
    try:
        if _parse_time(entry.get("expires_at")) <= now:
            refusals.append(f"grant expired at {entry.get('expires_at')}")
    except ValueError as exc:
        refusals.append(f"expires_at unreadable: {exc}")
    for field in ("compose_files", "services"):
        value = entry.get(field)
        if not isinstance(value, list) or not value:
            refusals.append(f"{field} must be a non-empty list")
    for field in ("env_files", "profiles"):
        if not isinstance(entry.get(field), list):
            refusals.append(f"{field} must be a list")
    for field in ("compose_ref", "rendered_digest"):
        if not isinstance(entry.get(field), str):
            refusals.append(f"{field} missing")
    return entry, refusals


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *args],
        env=scrub_git_location_env(os.environ),
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        raise ApplyUsageError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def fetch_anchor_from_main(occ_repo: Path) -> tuple[bytes, str]:
    """Read the anchor at ``onex_change_control@main`` after a fresh fetch."""
    if not (occ_repo / ".git").exists():
        raise ApplyUsageError(f"not a git clone: {occ_repo}")
    _git(occ_repo, "fetch", "origin", ANCHOR_REF, "--quiet")
    commit = _git(occ_repo, "rev-parse", f"origin/{ANCHOR_REF}").strip()
    raw = _git(occ_repo, "show", f"origin/{ANCHOR_REF}:{ANCHOR_RELPATH}")
    return raw.encode("utf-8"), commit


def checkout_refusals(repo_root: Path, entry: dict[str, Any]) -> list[str]:
    """The checkout must be the grant's commit, with its inputs unmodified."""
    refusals: list[str] = []
    head = _git(repo_root, "rev-parse", "HEAD").strip()
    if head != entry["compose_ref"]:
        refusals.append(
            f"checkout HEAD {head} is not the grant's compose_ref {entry['compose_ref']}"
        )
    dirty = _git(
        repo_root, "status", "--porcelain", "--", *entry["compose_files"]
    ).strip()
    if dirty:
        refusals.append(f"grant compose files are modified in this checkout: {dirty}")
    return refusals


def operator_env_file() -> Path:
    """The operator env onex-runtime-deploy sources; required, never defaulted away."""
    raw = os.environ.get("OMNIBASE_OPERATOR_ENV_FILE", "").strip()
    path = Path(raw) if raw else Path.home() / ".omnibase" / ".env"
    if not path.is_file():
        raise ApplyUsageError(f"operator env file not found: {path}")
    return path


def _occ_repo(arg: str) -> Path:
    if arg:
        return Path(arg)
    omni_home = os.environ.get("OMNI_HOME", "").strip()
    if not omni_home:
        raise ApplyUsageError("set --occ-repo or OMNI_HOME (no default path)")
    return Path(omni_home) / "onex_change_control"


def cmd_digest(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root)
    prefix = compose_argv(
        project=args.project,
        compose_files=args.file,
        env_files=args.env_file,
        profiles=args.profile,
        repo_root=repo_root,
        operator_env=operator_env_file(),
    )
    print(
        json.dumps(
            {
                "compose_project": args.project,
                "compose_files": args.file,
                "env_files": args.env_file,
                "profiles": args.profile,
                "compose_ref": _git(repo_root, "rev-parse", "HEAD").strip(),
                "rendered_digest": render_digest(prefix),
            },
            indent=2,
        )
    )
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root)
    now = _parse_time(args.now) if args.now else datetime.now(UTC)
    if args.grants_file:
        if args.execute:
            raise ApplyUsageError(
                "--grants-file is a test seam; --execute reads the anchor at "
                f"onex_change_control@{ANCHOR_REF} only"
            )
        raw, source = Path(args.grants_file).read_bytes(), f"file:{args.grants_file}"
    else:
        raw, commit = fetch_anchor_from_main(_occ_repo(args.occ_repo))
        source = f"onex_change_control@{ANCHOR_REF}={commit}"

    entry, refusals = match_grant(raw, grant_id=args.grant_id, lane=args.lane, now=now)
    if entry is not None and not refusals:
        refusals = checkout_refusals(repo_root, entry)
    prefix: list[str] = []
    if entry is not None and not refusals:
        prefix = compose_argv(
            project=entry["compose_project"],
            compose_files=entry["compose_files"],
            env_files=entry["env_files"],
            profiles=entry["profiles"],
            repo_root=repo_root,
            operator_env=operator_env_file(),
        )
        local = render_digest(prefix)
        if local != entry["rendered_digest"]:
            refusals.append(
                f"rendered digest {local} does not match the grant's "
                f"{entry['rendered_digest']}: this is not the config the approver read"
            )

    if refusals or entry is None:
        print(f"REFUSED grant={args.grant_id} lane={args.lane} anchor={source}")
        for refusal in refusals:
            print(f"  - {refusal}")
        return 1

    services = [str(s) for s in entry["services"]]
    up = [*prefix, "up", "-d", "--no-deps", "--no-build", *services]
    print(f"ALLOWED grant={args.grant_id} lane={args.lane} anchor={source}")
    print(f"  services: {' '.join(services)}")
    if not args.execute:
        print("  check only; pass --execute to apply")
        return 0

    preflight = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "preflight_lane_deploy_attribution.py"),
            "--compose-project",
            entry["compose_project"],
            "--source",
            "apply_governed_compose_config.py",
            "--invoking-command",
            shlex.join(up),
        ],
        check=False,
    )
    if preflight.returncode != 0:
        print("REFUSED by the lane-deploy attribution preflight (OMN-15218)")
        return 1
    return subprocess.run(up, timeout=COMPOSE_TIMEOUT_SECONDS, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    digest = sub.add_parser("digest", help="print the canonical digest and HEAD")
    digest.add_argument("--project", required=True)
    digest.add_argument("--file", action="append", required=True)
    digest.add_argument("--env-file", action="append", default=[])
    digest.add_argument("--profile", action="append", default=[])
    digest.set_defaults(func=cmd_digest)

    apply = sub.add_parser("apply", help="check a grant and, with --execute, apply")
    apply.add_argument("--lane", required=True, choices=sorted(GOVERNED_COMPOSE_LANES))
    apply.add_argument("--grant-id", required=True)
    apply.add_argument("--occ-repo", default="")
    apply.add_argument("--grants-file", default="")
    apply.add_argument("--now", default="")
    apply.add_argument("--execute", action="store_true")
    apply.set_defaults(func=cmd_apply)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result: int = args.func(args)
    except ApplyUsageError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return result


if __name__ == "__main__":
    sys.exit(main())
