#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hot-patch ledger rebuild preflight (OMN-13014, night-plan retro B-1).

Container-layer hot-patches (``.prepatch`` sibling discipline) silently revert
on any image rebuild or ``compose up --force-recreate``. This gate refuses to
rebuild a container when:

1. any hot-patch ledger row for the target container/lane has NO source PR
   merge commit that is an ancestor of the build ref for that repo
   (``git merge-base --is-ancestor``), or
2. the running container carries ``.prepatch`` files that are NOT recorded in
   the ledger (unledgered patches would be silently destroyed), or
3. with ``--post-rebuild``: any ``.prepatch`` file survives the rebuild.

A row's ``merge_commit`` may be a single commit string (legacy single-lineage
format) or a list of commit strings. Under dev->main squash promotion the same
patch content lives in two non-linear lineages (a dev merge commit and a main
promotion commit that are NOT ancestors of one another), so a list lets one
ledger row satisfy either a dev-ref or a main-ref rebuild. A row passes when
ANY candidate commit is known in the clone AND is an ancestor of the build ref.

The canonical ledger lives at ``/data/omninode/hotpatch-ledger/ledger.yaml``
on the runtime host (override with ``--ledger`` or ``HOTPATCH_LEDGER_PATH``).

Row lifecycle (OMN-16803 AC5): a row carries an optional ``status`` of
``active`` (the default when the field is absent) or ``reconciled``. A patch
whose content has been durably superseded — rebuilt from merged source, its
``.prepatch`` sibling gone — is retired by setting ``status: reconciled``
together with ``reconciled_utc`` and ``reconciliation_note``; the preflight
then skips the row with a printed notice instead of gating on it. The retiring
is done by ``node_hotpatch_ledger_reconcile_effect`` (OMN-17427), never by a
hand edit: it checks the fix commit is an ancestor of the deployed ref and the
container carries no ``.prepatch``, backs the ledger up, and refuses otherwise.
A refusal over an unusable ``merge_commit`` prints the exact command. Retiring a
row by DELETING it is wrong: the ledger is the forensic record of what was
ever patched, and deletion destroys exactly the history it exists to hold.
Retirement is not an allowlist — a reconciled row's ``.prepatch`` path leaves
the ledgered set, so if that file ever reappears the tripwire reports it as a
new UNLEDGERED patch. Any other ``status`` value is a hard configuration
failure, so a typo can never silently drop a live row out of scope.

Sole bypass: export ``HOTPATCH_PREFLIGHT_BYPASS`` containing a line of the
exact Rule-10 form ``# skip-token-allowed: <user-approval-receipt-id>`` where
the receipt id is a real user-issued approval handle. Any other value of the
variable is a hard failure.

Exit codes: 0 = pass (or authorized bypass), 1 = gate failure, 2 = usage or
configuration error (missing ledger, unknown commit, malformed bypass).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_LEDGER_PATH = "/data/omninode/hotpatch-ledger/ledger.yaml"
BYPASS_PATTERN = re.compile(r"^# skip-token-allowed: (\S+)$")
TRIPWIRE_SEARCH_PATHS = ("/app", "/usr/local/lib", "/usr/lib/python3", "/opt")
SUPPORTED_SCHEMA = 1
# Docker's own daemon error text for `docker exec` against a container name/id
# that does not exist, e.g. "Error response from daemon: No such container:
# omninode-stability-test-runtime-effects". Matched case-insensitively against
# stderr to distinguish "this container was never created" (OMN-16111 --
# expected on a --cold-start bring-up) from every other exec failure
# (permission error, daemon hiccup, ...), which must stay a hard failure.
_NO_SUCH_CONTAINER_RE = re.compile(r"no such container", re.IGNORECASE)

# Row lifecycle statuses (OMN-16803 AC5). ``active`` is the default for any row
# with no ``status`` field, so every pre-OMN-16803 ledger keeps its exact
# behavior. ``reconciled`` retires a row from gating WITHOUT deleting it.
ROW_STATUS_ACTIVE = "active"
ROW_STATUS_RECONCILED = "reconciled"
VALID_ROW_STATUSES = (ROW_STATUS_ACTIVE, ROW_STATUS_RECONCILED)
# A retirement is a recorded decision, not a quiet edit: both fields are
# mandatory on a reconciled row so the ledger always answers "when, and on
# what evidence" for every row that stopped being gated.
RECONCILED_REQUIRED_FIELDS = ("reconciled_utc", "reconciliation_note")

# OMN-17427: retirement is the node's job, not a hand edit. The refusal below
# prints the exact command; the node re-verifies everything it is told.
RECONCILE_NODE = "node_hotpatch_ledger_reconcile_effect"
RECONCILE_PLACEHOLDER_COMMIT = "<FIX_MERGE_SHA>"
RECONCILE_LINE_PREFIX = "HOTPATCH-PREFLIGHT RECONCILE: "
REPO_ROOT = Path(__file__).resolve().parents[1]


class ContainerAbsentError(ValueError):
    """Raised by ``tripwire_prepatch_files`` when the target container does
    not exist at all (Docker's "No such container" daemon error).

    Deliberately a ``ValueError`` subclass so any caller that still catches
    the broad ``ValueError`` (pre-OMN-16111 behavior) keeps working exactly
    as before; ``run_preflight`` catches this narrower type first to apply
    the ``--cold-start`` skip-not-fail carve-out (OMN-16111).
    """


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


def partition_by_status(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split *rows* into ``(active, reconciled)`` (OMN-16803 AC5).

    A row with no ``status`` field is ``active`` — pre-OMN-16803 ledgers are
    gated byte-for-byte as before. A ``reconciled`` row is retired from gating
    but stays in the ledger as forensic history.

    Raises:
        ValueError: the row carries an unrecognized ``status``, or a
            ``reconciled`` row is missing ``reconciled_utc`` /
            ``reconciliation_note``. Both are hard configuration failures: an
            unvalidated status would let a typo silently remove a live row
            from the gate's scope, and an unannotated retirement would record
            no decision at all.
    """
    active: list[dict[str, Any]] = []
    reconciled: list[dict[str, Any]] = []
    for row in rows:
        status = row.get("status", ROW_STATUS_ACTIVE)
        if status not in VALID_ROW_STATUSES:
            raise ValueError(
                f"ledger row for {row.get('file')!r} in "
                f"{row.get('container')!r} has unknown status {status!r}; "
                f"expected one of {', '.join(VALID_ROW_STATUSES)}"
            )
        if status == ROW_STATUS_ACTIVE:
            active.append(row)
            continue
        missing = [field for field in RECONCILED_REQUIRED_FIELDS if not row.get(field)]
        if missing:
            raise ValueError(
                f"ledger row for {row.get('file')!r} in "
                f"{row.get('container')!r} is status "
                f"{ROW_STATUS_RECONCILED!r} but is missing "
                f"{', '.join(missing)}; a retired row must record when it was "
                "reconciled and on what evidence"
            )
        reconciled.append(row)
    return active, reconciled


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


def candidate_commits(value: Any) -> list[str]:
    """Normalize a ledger row's ``merge_commit`` to a list of commit strings.

    Accepts either a scalar string (legacy single-lineage format) or a list of
    strings. Under dev->main squash promotion the same patch content lives in
    two non-linear lineages, so a row may carry both candidates. Raises
    ValueError for any other shape so a malformed ledger fails loudly.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        if not value or not all(isinstance(item, str) for item in value):
            raise ValueError(
                "ledger row 'merge_commit' list must be a non-empty list of "
                f"commit strings, got {value!r}"
            )
        return list(value)
    raise ValueError(
        "ledger row 'merge_commit' must be a string or list of strings, got "
        f"{type(value).__name__}"
    )


def suggest_fix_commit(clone: Path, source_pr: str, build_ref: str) -> str | None:
    """Return the one commit on *build_ref* whose message cites ``(#N)`` of the
    row's source PR, or None when zero or several match.

    Only a suggestion for the printed command: the reconcile node verifies
    ancestry and the container itself before it writes anything.
    """
    match = re.search(r"#(\d+)$", source_pr)
    if match is None:
        return None
    result = subprocess.run(
        [
            "git",
            "-C",
            str(clone),
            "log",
            "--format=%H",
            "-n",
            "2",
            "--fixed-strings",
            f"--grep=(#{match.group(1)})",
            build_ref,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    commits = result.stdout.split()
    if result.returncode != 0 or len(commits) != 1:
        return None
    return commits[0]


def reconcile_command(
    row: dict[str, Any],
    *,
    ledger_path: Path,
    clones_root: Path,
    docker_cmd: str,
    build_ref: str | None,
    fix_commit: str | None,
) -> str:
    """The exact shell command that retires *row* through the reconcile node."""
    payload = {
        "ledger_path": str(Path(ledger_path).resolve()),
        "clones_root": str(Path(clones_root).resolve()),
        "container": str(row.get("container")),
        "file": str(row.get("file")),
        "merge_commit": fix_commit or RECONCILE_PLACEHOLDER_COMMIT,
    }
    if build_ref is not None:
        payload["deployed_ref"] = build_ref
    if docker_cmd != "docker":
        payload["docker_cmd"] = docker_cmd
    state_root = REPO_ROOT / ".onex_state" / "hotpatch-reconcile"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{payload['container']}{payload['file']}")
    input_file = state_root / f"{stem}.json"
    return (
        f"mkdir -p {shlex.quote(str(state_root))}"
        f" && printf '%s' {shlex.quote(json.dumps(payload))} > {shlex.quote(str(input_file))}"
        f" && uv run --frozen --project {shlex.quote(str(REPO_ROOT))}"
        f" onex node {RECONCILE_NODE} --input {shlex.quote(str(input_file))}"
        f" --backend event_bus=inmemory --state-root {shlex.quote(str(state_root))}"
    )


def report_unusable_merge_commits(
    unusable: list[tuple[dict[str, Any], ValueError]],
    args: argparse.Namespace,
    explicit_refs: dict[str, str],
) -> int:
    """Refuse every row whose ``merge_commit`` is unusable, each with the
    command that retires it (OMN-17427). Reporting all of them in one run
    keeps an operator from discovering a second stale row only after fixing
    the first."""
    clones_root = Path(args.clones_root)
    for row, exc in unusable:
        repo = row.get("source_repo")
        build_ref: str | None = None
        fix_commit: str | None = None
        if isinstance(repo, str):
            try:
                build_ref = resolve_build_ref(repo, explicit_refs, clones_root)
            except ValueError:
                build_ref = None
            if build_ref is not None:
                fix_commit = suggest_fix_commit(
                    clones_root / repo, str(row.get("source_pr")), build_ref
                )
        advice = (
            f"Replace {RECONCILE_PLACEHOLDER_COMMIT} in the command with the "
            f"merge commit of {row.get('source_pr')} first."
            if fix_commit is None
            else f"Merge commit {fix_commit[:12]} was found on the build ref "
            "from the PR number."
        )
        fail(
            f"ledger row {row.get('file')!r} in {row.get('container')!r} "
            f"({row.get('source_pr')}) has no usable merge_commit: {exc}. "
            "If its fix is merged into the build ref and the container no "
            "longer carries the .prepatch, retire the row with the command "
            "below. It re-checks both, refuses otherwise (the reason is in "
            "handler_result of workflow_result.json under its --state-root), "
            "and writes a timestamped backup beside the ledger. " + advice,
            code=2,
        )
        print(
            RECONCILE_LINE_PREFIX
            + reconcile_command(
                row,
                ledger_path=Path(args.ledger),
                clones_root=clones_root,
                docker_cmd=args.docker_cmd,
                build_ref=build_ref,
                fix_commit=fix_commit,
            ),
            file=sys.stderr,
        )
    return 2


def tripwire_prepatch_files(container: str, docker_cmd: str) -> list[str]:
    """Return every ``.prepatch`` path found inside the running container.

    Raises:
        ContainerAbsentError: ``docker exec`` reports the container does not
            exist at all. Distinct from any other exec failure so
            ``run_preflight`` can skip-not-fail this outcome under
            ``--cold-start`` (OMN-16111) while every other failure mode
            (permission error, daemon hiccup, ...) stays a hard failure.
        ValueError: any other exec failure.
    """
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
        stderr = result.stderr.strip()
        if _NO_SUCH_CONTAINER_RE.search(stderr):
            raise ContainerAbsentError(
                f"tripwire probe could not exec into container {container!r} "
                f"— it does not exist: {stderr}"
            )
        raise ValueError(
            f"tripwire probe could not exec into container {container!r}: {stderr}"
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

    selected = select_rows(ledger["rows"] or [], args.container, args.lane)
    try:
        rows, reconciled_rows = partition_by_status(selected)
    except ValueError as exc:
        return fail(str(exc), code=2)
    scope = args.container or args.lane
    print(
        f"HOTPATCH-PREFLIGHT: ledger {ledger_path} — {len(rows)} row(s) "
        f"in scope {scope!r} ({len(reconciled_rows)} reconciled, skipped)"
    )
    for row in reconciled_rows:
        # Skip-with-notice, never silent: a retired row still shows up in
        # every preflight log, so the decision stays auditable from the run
        # that relies on it.
        print(
            f"HOTPATCH-PREFLIGHT SKIP (reconciled {row['reconciled_utc']}): "
            f"{row['container']} {row['file']} — "
            f"{' '.join(str(row['reconciliation_note']).split())}"
        )

    failures: list[str] = []
    clones_root = Path(args.clones_root)
    resolved_refs: dict[str, str] = {}

    usable: list[tuple[dict[str, Any], list[str]]] = []
    unusable: list[tuple[dict[str, Any], ValueError]] = []
    for row in rows:
        try:
            usable.append((row, candidate_commits(row.get("merge_commit"))))
        except ValueError as exc:
            unusable.append((row, exc))
    if unusable:
        return report_unusable_merge_commits(unusable, args, explicit_refs)

    for row, candidates in usable:
        repo = row["source_repo"]
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

        # A row passes when ANY candidate commit is known in the clone AND is an
        # ancestor of the build ref. Under dev->main squash promotion the dev
        # and main lineage commits are not ancestors of one another, so only one
        # candidate is expected to satisfy a given build ref.
        known_candidates: list[str] = []
        matched: str | None = None
        for candidate in candidates:
            if not commit_known(clone, candidate):
                continue
            known_candidates.append(candidate)
            try:
                if is_ancestor(clone, candidate, build_ref):
                    matched = candidate
                    break
            except ValueError as exc:
                return fail(str(exc), code=2)

        joined = ", ".join(candidates)
        if not known_candidates:
            # No candidate exists in the clone at all — stale or wrong clone.
            return fail(
                f"merge commit(s) {joined} ({row['source_pr']}) unknown in "
                f"clone {clone} — the clone is stale or wrong; "
                "run 'git fetch' on the build host first.",
                code=2,
            )
        if matched is not None:
            print(
                f"HOTPATCH-PREFLIGHT: {row['container']} {row['file']} "
                f"<- {row['source_pr']} ({matched[:12]}): ancestor-of-build-ref"
            )
        else:
            print(
                f"HOTPATCH-PREFLIGHT: {row['container']} {row['file']} "
                f"<- {row['source_pr']} ({joined}): NOT in build ref"
            )
            failures.append(
                f"{row['file']} in {row['container']} is hot-patched from "
                f"{row['source_pr']} (candidate merge commit(s): {joined}), "
                f"none of which is merged into build ref {build_ref} for repo "
                f"{repo}; rebuilding would silently destroy this patch."
            )

    if not args.skip_tripwire:
        containers = sorted({str(row["container"]) for row in rows})
        if args.container is not None:
            containers = sorted(set(containers) | {args.container})
        ledgered = {str(row["prepatch_path"]) for row in rows}
        for container in containers:
            try:
                found = tripwire_prepatch_files(container, args.docker_cmd)
            except ContainerAbsentError as exc:
                if args.cold_start:
                    # OMN-16111: a from-scratch cold bring-up legitimately has
                    # not created this container yet. A container that does
                    # not exist cannot carry a live hot-patch — vacuously 0
                    # .prepatch files, nothing to check. Warm-lane behavior
                    # (no --cold-start) is byte-for-byte unchanged below.
                    print(
                        f"HOTPATCH-PREFLIGHT: tripwire {container}: container "
                        "does not exist yet (--cold-start bring-up) — 0 "
                        ".prepatch file(s), nothing to check."
                    )
                    continue
                failures.append(str(exc))
                continue
            except ValueError as exc:
                failures.append(str(exc))
                continue
            unledgered = [path for path in found if path not in ledgered]
            container_ledgered = {
                str(row["prepatch_path"])
                for row in rows
                if row["container"] == container
            }
            missing = sorted(path for path in container_ledgered if path not in found)
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
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help=(
            "from-scratch cold bring-up (OMN-16111): a ledgered container "
            "that does not exist yet is 0 .prepatch files, not a failure. "
            "Every existing warm-refresh call site must NOT pass this flag "
            "-- absence there means a container that should already be up "
            "unexpectedly vanished, which stays a hard failure exactly as "
            "before this flag existed."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_preflight(args)


if __name__ == "__main__":
    sys.exit(main())
