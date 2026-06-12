#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hot-patch ledger rebuild preflight (OMN-13014, night-plan retro B-1).

Container-layer hot-patches (``.prepatch`` sibling discipline) silently revert
on any image rebuild or ``compose up --force-recreate``. This gate refuses to
rebuild a container when:

1. any hot-patch ledger row for the target container/lane has a source PR
   merge commit that is NOT an ancestor of the build ref for that repo
   (``git merge-base --is-ancestor``), or
2. the running container carries ``.prepatch`` files that are NOT recorded in
   the ledger (unledgered patches would be silently destroyed), or
3. with ``--post-rebuild``: any ``.prepatch`` file survives the rebuild.

The canonical ledger lives at ``/data/omninode/hotpatch-ledger/ledger.yaml``
on the runtime host (override with ``--ledger`` or ``HOTPATCH_LEDGER_PATH``).

Sole bypass: export ``HOTPATCH_PREFLIGHT_BYPASS`` containing a line of the
exact Rule-10 form ``# skip-token-allowed: <user-approval-receipt-id>`` where
the receipt id is a real user-issued approval handle. Any other value of the
variable is a hard failure.

Exit codes: 0 = pass (or authorized bypass), 1 = gate failure, 2 = usage or
configuration error (missing ledger, unknown commit, malformed bypass).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_LEDGER_PATH = "/data/omninode/hotpatch-ledger/ledger.yaml"
BYPASS_PATTERN = re.compile(r"^# skip-token-allowed: (\S+)$")
TRIPWIRE_SEARCH_PATHS = ("/app", "/usr/local/lib", "/usr/lib/python3", "/opt")
SUPPORTED_SCHEMA = 1


def fail(message: str, *, code: int = 1) -> int:
    print(f"HOTPATCH-PREFLIGHT FAIL: {message}", file=sys.stderr)
    return code


def load_ledger(ledger_path: Path) -> dict[str, Any]:
    if not ledger_path.is_file():
        raise FileNotFoundError(
            f"hot-patch ledger not found at {ledger_path}; refusing to guess. "
            "If this host has never recorded a hot-patch, create an empty "
            "ledger (schema: 1, rows: [])."
        )
    raw = yaml.safe_load(ledger_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"ledger at {ledger_path} is not a mapping")
    schema = raw.get("schema")
    if schema != SUPPORTED_SCHEMA:
        raise ValueError(
            f"unsupported ledger schema {schema!r} (expected {SUPPORTED_SCHEMA})"
        )
    rows = raw.get("rows")
    if rows is None:
        raise ValueError("ledger has no 'rows' key")
    if not isinstance(rows, list):
        raise ValueError("ledger 'rows' must be a list")
    return raw


def select_rows(
    rows: list[dict[str, Any]],
    container: str | None,
    lane: str | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if container is not None and row.get("container") != container:
            continue
        if lane is not None and row.get("lane") != lane:
            continue
        selected.append(row)
    return selected


def resolve_build_ref(
    repo: str,
    explicit_refs: dict[str, str],
    clones_root: Path,
) -> str:
    """Return the build ref for *repo*, defaulting to the clone's HEAD.

    Workspace-mode builds vendor sibling repos from their clone working tree,
    so the clone HEAD *is* the build ref when not explicitly overridden. The
    resolved SHA is always printed so the deploy ledger can record it.
    """
    if repo in explicit_refs:
        return explicit_refs[repo]
    clone = clones_root / repo
    head = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode != 0:
        raise ValueError(
            f"no --build-ref given for repo {repo!r} and could not resolve "
            f"HEAD of clone {clone}: {head.stderr.strip()}"
        )
    return head.stdout.strip()


def commit_known(clone: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(clone), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def is_ancestor(clone: Path, commit: str, build_ref: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(clone), "merge-base", "--is-ancestor", commit, build_ref],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise ValueError(
        f"git merge-base failed in {clone} for {commit}..{build_ref}: "
        f"{result.stderr.strip()}"
    )


def tripwire_prepatch_files(container: str, docker_cmd: str) -> list[str]:
    """Return every ``.prepatch`` path found inside the running container."""
    find_cmd = " ".join(
        f'find {path} -name "*.prepatch" 2>/dev/null;' for path in TRIPWIRE_SEARCH_PATHS
    )
    result = subprocess.run(
        [docker_cmd, "exec", container, "sh", "-c", find_cmd],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            f"tripwire probe could not exec into container {container!r}: "
            f"{result.stderr.strip()}"
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def check_bypass() -> str | None:
    """Return the receipt id when an authorized bypass is present.

    Raises ValueError when the bypass variable is set but malformed — a
    malformed bypass is a hard failure, never a silent pass-through.
    """
    raw = os.environ.get("HOTPATCH_PREFLIGHT_BYPASS")
    if raw is None:
        return None
    match = BYPASS_PATTERN.fullmatch(raw.strip())
    if match is None:
        raise ValueError(
            "HOTPATCH_PREFLIGHT_BYPASS is set but does not match the required "
            "form '# skip-token-allowed: <user-approval-receipt-id>'"
        )
    return match.group(1)


def parse_build_refs(pairs: list[str]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for pair in pairs:
        repo, sep, ref = pair.partition("=")
        if not sep or not repo or not ref:
            raise ValueError(f"--build-ref must be <repo>=<ref>, got {pair!r}")
        refs[repo] = ref
    return refs


def run_preflight(args: argparse.Namespace) -> int:
    try:
        receipt = check_bypass()
    except ValueError as exc:
        return fail(str(exc), code=2)
    if receipt is not None:
        print(
            "HOTPATCH-PREFLIGHT BYPASS: authorized by user approval receipt "
            f"{receipt!r} — gate checks skipped."
        )
        return 0

    ledger_path = Path(args.ledger)
    try:
        ledger = load_ledger(ledger_path)
        explicit_refs = parse_build_refs(args.build_ref)
    except (FileNotFoundError, ValueError) as exc:
        return fail(str(exc), code=2)

    rows = select_rows(ledger["rows"] or [], args.container, args.lane)
    scope = args.container or args.lane
    print(
        f"HOTPATCH-PREFLIGHT: ledger {ledger_path} — {len(rows)} row(s) "
        f"in scope {scope!r}"
    )

    failures: list[str] = []
    clones_root = Path(args.clones_root)
    resolved_refs: dict[str, str] = {}

    for row in rows:
        repo = row["source_repo"]
        commit = row["merge_commit"]
        clone = clones_root / repo
        if repo not in resolved_refs:
            try:
                resolved_refs[repo] = resolve_build_ref(
                    repo, explicit_refs, clones_root
                )
            except ValueError as exc:
                return fail(str(exc), code=2)
            print(f"HOTPATCH-PREFLIGHT: build ref {repo}={resolved_refs[repo]}")
        build_ref = resolved_refs[repo]
        if not commit_known(clone, commit):
            return fail(
                f"merge commit {commit} ({row['source_pr']}) is unknown in "
                f"clone {clone} — the clone is stale or wrong; "
                "run 'git fetch' on the build host first.",
                code=2,
            )
        try:
            merged = is_ancestor(clone, commit, build_ref)
        except ValueError as exc:
            return fail(str(exc), code=2)
        marker = "ancestor-of-build-ref" if merged else "NOT in build ref"
        print(
            f"HOTPATCH-PREFLIGHT: {row['container']} {row['file']} "
            f"<- {row['source_pr']} ({commit[:12]}): {marker}"
        )
        if not merged:
            failures.append(
                f"{row['file']} in {row['container']} is hot-patched from "
                f"{row['source_pr']} (merge commit {commit}) which is NOT "
                f"merged into build ref {build_ref} for repo {repo}; "
                "rebuilding would silently destroy this patch."
            )

    if not args.skip_tripwire:
        containers = sorted({str(row["container"]) for row in rows})
        if args.container is not None:
            containers = sorted(set(containers) | {args.container})
        ledgered = {str(row["prepatch_path"]) for row in rows}
        for container in containers:
            try:
                found = tripwire_prepatch_files(container, args.docker_cmd)
            except ValueError as exc:
                failures.append(str(exc))
                continue
            unledgered = [path for path in found if path not in ledgered]
            missing = sorted(
                path
                for path in ledgered
                if path not in found
                and any(row["container"] == container for row in rows)
            )
            print(
                f"HOTPATCH-PREFLIGHT: tripwire {container}: "
                f"{len(found)} .prepatch file(s) live"
            )
            if args.post_rebuild and found:
                failures.append(
                    f"post-rebuild tripwire: {container} still carries "
                    f".prepatch files {found}; the rebuild did not start from "
                    "a clean image."
                )
            if not args.post_rebuild and unledgered:
                failures.append(
                    f"tripwire: {container} carries UNLEDGERED .prepatch "
                    f"files {unledgered}; record them in the ledger before "
                    "any rebuild."
                )
            if not args.post_rebuild and missing:
                print(
                    f"HOTPATCH-PREFLIGHT WARN: ledgered .prepatch missing from "
                    f"{container}: {missing} (patch may already be reverted)"
                )

    if failures:
        for failure in failures:
            fail(failure)
        return 1
    print("HOTPATCH-PREFLIGHT PASS: all in-scope hot-patches merged into build ref.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--container", help="exact container name to gate")
    scope.add_argument("--lane", help="gate every ledger row in this lane")
    parser.add_argument(
        "--build-ref",
        action="append",
        default=[],
        metavar="REPO=REF",
        help="build ref per repo (repeatable); defaults to clone HEAD",
    )
    parser.add_argument(
        "--clones-root",
        required=True,
        help="directory containing the build-input git clones, one per repo",
    )
    parser.add_argument(
        "--ledger",
        default=os.environ.get("HOTPATCH_LEDGER_PATH", DEFAULT_LEDGER_PATH),
        help="hot-patch ledger path (env HOTPATCH_LEDGER_PATH overrides default)",
    )
    parser.add_argument("--docker-cmd", default="docker")
    parser.add_argument(
        "--skip-tripwire",
        action="store_true",
        help="skip the running-container .prepatch probe (offline analysis)",
    )
    parser.add_argument(
        "--post-rebuild",
        action="store_true",
        help="post-rebuild mode: any surviving .prepatch file is a failure",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_preflight(args)


if __name__ == "__main__":
    sys.exit(main())
