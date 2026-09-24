#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Return the compose-dev lane to its last known-good composition, under rule RB.

OMN-19322 (unified verification plan rows R1 and R2). One command:

  1. reads the lane's current composition from ``/app/build-provenance.json``
     (the "bad" composition unless ``--bad-sha`` names one);
  2. reads PASS ``lab-pass-receipt-compose-dev-<sha>`` artifacts newest first
     as rollback CANDIDATES -- only a PASS counts, only a strict ancestor of
     the bad sha is older than it;
  3. reads the live migration ledger of every database the lane binds, the
     database names resolved from the lane's topology catalog;
  4. keeps the newest candidate rule RB makes eligible (below) and prints why
     every newer one was refused, naming the migration and the database;
  5. with ``--execute`` only, redeploys it through the sanctioned path: it
     reruns the Runtime Rebuild Trigger run of the merge whose commit IS the
     candidate, so the same CI job, identity and bus credentials that published
     that merge's redeploy-start publish it again, pinned to the candidate's
     40-hex sha. It needs no local bus credential. It refuses while any trigger
     run is queued or running (one dev-lane deploy at a time), then reads the
     provenance of every runtime container back until it names the candidate.
     The rerun also re-runs that run's verify job, which emits a fresh
     compose-dev receipt for the candidate sha from the lane as it now runs.

Rule RB (plan section 4, workstream R):
  RB-1  a candidate is eligible when every migration applied on the lane that
        its own tree does not declare is expand-only, or has a PASS lab
        execution of its down-migration recorded (RB-2's exception).
  RB-2  a forward-only, contract or undeclared migration applied since the
        candidate refuses it, naming the migration and the database.
  RB-3  an unreadable ledger, or no ledger read at all, is INDETERMINATE and
        nothing is redeployed. No PASS receipt at all is a named gap.
  RB-4  the executed pair: tests/ci/test_rollback_lane_to_known_good_omn19322.py.

The classes and the down-execution records come from OMN-19344
(``scripts/validation/check_migration_class.py``). A selection that depends on a
lifted barrier needs that down-migration run on the lane BEFORE the redeploy;
this command does not run DDL on a lane, so it reports the selection and
refuses ``--execute`` for it.

What it cannot restore (named, not hidden): the redeploy command pins ONE
repository, omnibase_infra. The deploy agent resolves every sibling
(omnibase_core, omnibase_compat, omnimarket) at staging time from its own
fallback ref, so a rollback restores the infra half of a composition and the
readback reports the siblings as observed, not as restored. A receipt records
no sibling refs today (plan P9's composition block is not built).

There is no path to production or to a governed lane: ``--lane`` accepts
``compose-dev`` and nothing else, and only a trigger run of a merge into
``dev`` is ever rerun, which the rebuild trigger maps to the dev runtime lane
only. GitHub keeps a run rerunnable for 30 days, so an older candidate is
refused by name rather than redeployed some other way.

Usage::

    uv run python scripts/rollback_lane_to_known_good.py --lane compose-dev \\
        --lane-host <ssh target of the lane host>          # select, redeploy nothing
    ... --execute                                         # redeploy the selection
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
OWN_REPO: Final[str] = "OmniNode-ai/omnibase_infra"
_CHECKER_PATH: Final[Path] = (
    REPO_ROOT / "scripts" / "validation" / "check_migration_class.py"
)
_TOPOLOGY: Final[Path] = (
    REPO_ROOT / "docker" / "catalog" / "database-topology" / "local.yaml"
)
_RECEIPT_NAME: Final[re.Pattern[str]] = re.compile(
    r"^lab-pass-receipt-compose-dev-([0-9a-f]{40})$"
)
_SHA40: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ModelLaneSpec:
    """The one lane this command may act on."""

    runtime_lane: str
    base_branch: str
    compose_project: str
    postgres_container: str
    provenance_container: str


#: compose-dev ONLY. Adding a lane here is adding a rollback path to it, which
#: for any governed or production lane is exactly what rule 2a forbids.
LANES: Final[Mapping[str, ModelLaneSpec]] = {
    "compose-dev": ModelLaneSpec(
        runtime_lane="dev",
        base_branch="dev",
        compose_project="omnibase-infra",
        postgres_container="omnibase-infra-postgres",
        provenance_container="omninode-runtime",
    )
}


@dataclass(frozen=True)
class ModelLedgerSpec:
    """How one bound database records its applied migrations.

    The database's physical name comes from the topology catalog. The id
    column is the one its runner WRITES: the catalog declares ``version`` for
    the two service-owned ledgers, but ``run-forward-migrations.sh`` inserts
    ``migration_id`` and ``run-intelligence-migrations.sh`` inserts
    ``migration_name`` there (read on the dev lane 2026-09-24). Reading the
    declared column would make every selection INDETERMINATE; reading the
    written one is reading the ledger that exists.
    """

    catalog_key: str
    query: str


LEDGER_SPECS: Final[tuple[ModelLedgerSpec, ...]] = (
    ModelLedgerSpec(
        "omnibase_infra", "select migration_id from public.schema_migrations"
    ),
    ModelLedgerSpec(
        "application", "select version from platform_catalog.schema_migrations"
    ),
    ModelLedgerSpec(
        "omniintelligence", "select migration_name from public.schema_migrations"
    ),
)


def _load(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_checker = _load("check_migration_class", _CHECKER_PATH)


def load_classes() -> dict[str, str]:
    return dict(_checker.load_manifest(_checker.DEFAULT_MANIFEST))  # type: ignore[attr-defined]


def load_down_executions() -> list[object]:
    return list(_checker.load_executions(_checker.DEFAULT_EXECUTIONS))  # type: ignore[attr-defined]


def ledger_id_to_key(database: str, raw: str) -> str | None:
    """A ledger row's id as the class manifest keys it, or None if foreign.

    None marks a row this composition does not own (the application database
    also carries the cloud image's own migration stream); it is counted and
    reported, never evaluated.
    """
    raw = raw.strip()
    if database == "omnibase_infra":
        return f"forward/{raw[len('docker/') :]}" if raw.startswith("docker/") else None
    if database == "omnidash_analytics":
        parts = raw.split(":")
        if len(parts) == 3 and parts[0] == "node" and parts[1].startswith("node_"):
            return f"forward/nodes/{parts[1]}/{parts[2]}"
        return None
    if database == "omniintelligence":
        return f"intelligence/{raw}.sql" if raw and "/" not in raw else None
    return None


def declared_keys_from_listing(paths: Iterable[str]) -> frozenset[str]:
    """The forward migrations a tree declares, from ``git ls-tree`` paths.

    The same enumeration ``check_migration_class.enumerate_forward_migrations``
    applies to a working tree: flat ``forward/*.sql``, ``forward/nodes/<n>/*.sql``
    and ``intelligence/*.sql``.
    """
    keys: set[str] = set()
    prefix = "docker/migrations/"
    for path in paths:
        if not path.startswith(prefix) or not path.endswith(".sql"):
            continue
        parts = path[len(prefix) :].split("/")
        if (len(parts) == 2 and parts[0] in ("forward", "intelligence")) or (
            len(parts) == 4 and parts[:2] == ["forward", "nodes"]
        ):
            keys.add("/".join(parts))
    return frozenset(keys)


@dataclass(frozen=True)
class ModelCandidate:
    sha: str
    receipt_artifact_id: int
    finished_at: str
    declared: frozenset[str]


@dataclass(frozen=True)
class ModelLedgerRead:
    database: str
    applied: frozenset[str] | None
    error: str = ""
    foreign_rows: int = 0


@dataclass(frozen=True)
class ModelSelection:
    verdict: str  # SELECTED | REFUSED | INDETERMINATE | NO_CANDIDATE
    candidate: ModelCandidate | None = None
    refusals: tuple[str, ...] = ()
    requires_down: tuple[str, ...] = ()
    reason: str = ""
    evaluated: tuple[str, ...] = field(default=())


def select_candidate(
    candidates: Sequence[ModelCandidate],
    ledgers: Sequence[ModelLedgerRead],
    classes: Mapping[str, str],
    executions: Sequence[object],
    root: Path,
) -> ModelSelection:
    """Rule RB over ``candidates`` (newest first). Pure; every refusal is named."""
    if not candidates:
        return ModelSelection(
            verdict="NO_CANDIDATE",
            reason=(
                "no PASS compose-dev receipt for any strict ancestor of the bad "
                "composition in the receipts read; nothing to roll back to"
            ),
        )
    if not ledgers:
        return ModelSelection(
            verdict="INDETERMINATE",
            reason="no migration ledger was read, so no candidate can be judged (RB-3)",
        )
    unreadable = [ledger for ledger in ledgers if ledger.applied is None]
    if unreadable:
        return ModelSelection(
            verdict="INDETERMINATE",
            reason="; ".join(
                f"ledger of database {ledger.database} unreadable: {ledger.error}"
                for ledger in unreadable
            )
            + " (RB-3: never eligible, nothing redeployed)",
        )
    refusals: list[str] = []
    evaluated: list[str] = []
    for candidate in candidates:
        barriers: list[str] = []
        lifted: list[str] = []
        for ledger in ledgers:
            assert ledger.applied is not None
            for key in sorted(ledger.applied - candidate.declared):
                declared = classes.get(key)
                barrier = _checker.rb2_barrier(key, declared, executions, root)  # type: ignore[attr-defined]
                if barrier is not None:
                    barriers.append(f"{barrier} [database {ledger.database}]")
                elif declared != _checker.EXPAND_ONLY:  # type: ignore[attr-defined]
                    lifted.append(key)
        evaluated.append(candidate.sha[:12])
        if barriers:
            refusals.append(
                f"candidate {candidate.sha[:12]} refused: " + "; ".join(barriers)
            )
            continue
        return ModelSelection(
            verdict="SELECTED",
            candidate=candidate,
            refusals=tuple(refusals),
            requires_down=tuple(lifted),
            evaluated=tuple(evaluated),
        )
    return ModelSelection(
        verdict="REFUSED",
        refusals=tuple(refusals),
        reason=(
            "every candidate is older than a rollback barrier; revert by a new "
            "forward commit through the normal merge path instead (plan R1)"
        ),
        evaluated=tuple(evaluated),
    )


def act_on_selection(
    selection: ModelSelection,
    *,
    execute: bool,
    redeploy: Callable[[str], object],
) -> int:
    """Print the selection; redeploy only a clean SELECTED under ``--execute``."""
    for line in selection.refusals:
        print(f"  {line}")
    if selection.verdict != "SELECTED" or selection.candidate is None:
        print(f"VERDICT {selection.verdict}: {selection.reason}")
        return 1 if selection.verdict == "REFUSED" else 2
    sha = selection.candidate.sha
    if selection.requires_down:
        print(
            f"VERDICT SELECTED {sha} REQUIRES-DOWN: eligible only because a recorded "
            f"lab down-migration lifts {', '.join(selection.requires_down)}; running "
            "that down on the lane must precede the redeploy, and this command runs "
            "no DDL on a lane, so nothing is redeployed"
        )
        return 3
    print(
        f"VERDICT SELECTED {sha} (receipt artifact {selection.candidate.receipt_artifact_id})"
    )
    if not execute:
        print("dry run: nothing redeployed (pass --execute to redeploy it)")
        return 0
    redeploy(sha)
    return 0


# ---------------------------------------------------------------------------
# Live reads (lane host over ssh, GitHub artifacts, the local clone)
# ---------------------------------------------------------------------------


def _run(
    argv: Sequence[str], *, timeout: float = 120.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), capture_output=True, text=True, check=False, timeout=timeout
    )


def _on_lane(host: str, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            shlex.join(argv),
        ]
    )


def catalog_databases() -> dict[str, str]:
    data = yaml.safe_load(_TOPOLOGY.read_text(encoding="utf-8"))
    return {
        key: str(value["physical_name"]) for key, value in data["databases"].items()
    }


def read_ledgers(host: str, lane: ModelLaneSpec) -> list[ModelLedgerRead]:
    physical = catalog_databases()
    reads: list[ModelLedgerRead] = []
    for spec in LEDGER_SPECS:
        database = physical.get(spec.catalog_key)
        if database is None:
            reads.append(
                ModelLedgerRead(
                    spec.catalog_key, None, "the topology catalog does not resolve it"
                )
            )
            continue
        result = _on_lane(
            host,
            [
                "docker",
                "exec",
                lane.postgres_container,
                "psql",
                "-U",
                "postgres",
                "-d",
                database,
                "-tAc",
                spec.query,
            ],
        )
        rows = [row for row in result.stdout.splitlines() if row.strip()]
        if result.returncode != 0 or not rows:
            error = f"exit {result.returncode}: {result.stderr.strip()[:200] or 'zero rows'}"
            reads.append(ModelLedgerRead(database, None, error))
            continue
        keys = [ledger_id_to_key(database, row) for row in rows]
        reads.append(
            ModelLedgerRead(
                database,
                frozenset(k for k in keys if k is not None),
                foreign_rows=sum(1 for k in keys if k is None),
            )
        )
    return reads


def read_provenance(host: str, lane: ModelLaneSpec) -> dict[str, dict[str, object]]:
    """``{container: build-provenance.json}`` for every lane container carrying one."""
    listed = _on_lane(
        host,
        [
            "docker",
            "ps",
            "--filter",
            f"label=com.docker.compose.project={lane.compose_project}",
            "--format",
            "{{.Names}}",
        ],
    )
    out: dict[str, dict[str, object]] = {}
    for name in sorted(listed.stdout.split()):
        result = _on_lane(
            host, ["docker", "exec", name, "cat", "/app/build-provenance.json"]
        )
        if result.returncode == 0:
            try:
                out[name] = json.loads(result.stdout)
            except json.JSONDecodeError:
                continue
    return out


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", "-C", str(REPO_ROOT), *args])


def _is_strict_ancestor(sha: str, bad: str) -> bool:
    return sha != bad and _git("merge-base", "--is-ancestor", sha, bad).returncode == 0


def _declared_at(sha: str) -> frozenset[str] | None:
    result = _git(
        "ls-tree",
        "-r",
        "--name-only",
        sha,
        "--",
        "docker/migrations/forward",
        "docker/migrations/intelligence",
    )
    if result.returncode != 0:
        return None
    return declared_keys_from_listing(result.stdout.splitlines())


def _gh_json(path: str) -> object:
    result = _run(["gh", "api", path])
    if result.returncode != 0:
        msg = f"gh api {path} failed: {result.stderr.strip()[:200]}"
        raise RuntimeError(msg)
    return json.loads(result.stdout)


def read_candidates(
    bad: str, *, max_pages: int, max_receipts: int
) -> tuple[list[ModelCandidate], list[str]]:
    """PASS compose-dev receipts for strict ancestors of ``bad``, newest first."""
    notes: list[str] = []
    newest: dict[str, dict[str, object]] = {}
    for page in range(1, max_pages + 1):
        data = _gh_json(f"repos/{OWN_REPO}/actions/artifacts?per_page=100&page={page}")
        assert isinstance(data, dict)
        for artifact in data.get("artifacts", []):
            match = _RECEIPT_NAME.match(str(artifact.get("name", "")))
            if not match or artifact.get("expired"):
                continue
            sha = match.group(1)
            if sha not in newest or str(artifact["created_at"]) > str(
                newest[sha]["created_at"]
            ):
                newest[sha] = artifact
    notes.append(
        f"{len(newest)} compose-dev receipt artifact(s) in {max_pages} page(s)"
    )
    candidates: list[ModelCandidate] = []
    seen: set[str] = set()
    ordered = sorted(
        newest.items(), key=lambda kv: str(kv[1]["created_at"]), reverse=True
    )
    for sha, artifact in ordered[:max_receipts]:
        blob = subprocess.run(
            ["gh", "api", f"repos/{OWN_REPO}/actions/artifacts/{artifact['id']}/zip"],
            capture_output=True,
            check=False,
            timeout=120,
        )
        try:
            with zipfile.ZipFile(io.BytesIO(blob.stdout)) as archive:
                receipt = json.loads(archive.read("receipt.json"))
        except (KeyError, zipfile.BadZipFile, json.JSONDecodeError):
            notes.append(
                f"{sha[:12]}: artifact {artifact['id']} has no readable receipt.json"
            )
            continue
        if receipt.get("result") != "PASS" or receipt.get("lane") != "compose-dev":
            notes.append(
                f"{sha[:12]}: receipt {receipt.get('result')} (not a candidate)"
            )
            continue
        probed = probed_revision(receipt)
        resolved = (
            _git(
                "rev-parse", "--verify", "--quiet", f"{probed}^{{commit}}"
            ).stdout.strip()
            if probed
            else ""
        )
        if not _SHA40.match(resolved):
            notes.append(
                f"{sha[:12]}: PASS receipt names no resolvable probed revision "
                f"({probed!r}); the composition it proved is unknown, so not a candidate"
            )
            continue
        if resolved != sha:
            notes.append(
                f"{sha[:12]}: PASS receipt was proven on revision {resolved[:12]} "
                "(a coalesced build); the candidate is the composition that ran"
            )
        if resolved in seen:
            continue
        seen.add(resolved)
        if not _is_strict_ancestor(resolved, bad):
            notes.append(f"{resolved[:12]}: not a strict ancestor of the bad sha")
            continue
        declared = _declared_at(resolved)
        if declared is None:
            notes.append(
                f"{resolved[:12]}: tree unreadable in the local clone (fetch origin)"
            )
            continue
        candidates.append(
            ModelCandidate(
                resolved,
                int(str(artifact["id"])),
                str(receipt.get("finished_at", "")),
                declared,
            )
        )
    commit_time = {
        c.sha: _git("log", "-1", "--format=%ct", c.sha).stdout.strip()
        for c in candidates
    }
    candidates.sort(key=lambda c: int(commit_time[c.sha] or 0), reverse=True)
    return candidates, notes


_PROBED: Final[re.Pattern[str]] = re.compile(r"\brevision=([0-9a-f]{7,40})\b")


def probed_revision(receipt: Mapping[str, object]) -> str:
    """The omnibase_infra revision a receipt's probes actually ran against.

    A receipt is KEYED by the merge sha that asked for a build, but a coalesced
    or superseded build proves a different revision: on 2026-09-23 twenty PASS
    receipts keyed by twenty different shas all carried one verification run,
    whose probes read revision 842f4c122c2c. The composition that was good is
    the one that ran, so the candidate is read from ``probe_generation_bound``
    (the revision the probes were bound to) and never from the receipt's key.
    Empty when the receipt names none -- which makes it no candidate.
    """
    checks = receipt.get("checks")
    if not isinstance(checks, list):
        return ""
    for check in checks:
        if not isinstance(check, dict) or check.get("name") != "probe_generation_bound":
            continue
        if check.get("ok") is not True:
            return ""
        match = _PROBED.search(str(check.get("evidence", "")))
        return match.group(1) if match else ""
    return ""


TRIGGER_WORKFLOW: Final[str] = "runtime-rebuild-trigger.yml"
_TRIGGER_JOB: Final[str] = "Trigger node_redeploy Start"
#: The job that runs only when the trigger job PUBLISHED a redeploy-start
#: (``if: needs.trigger-rebuild.outputs.published == 'true'`` upstream of it).
_VERIFY_JOB: Final[str] = "Verify dev lane applied the redeploy"
_RAN: Final[frozenset[str]] = frozenset(
    {"success", "failure", "cancelled", "timed_out"}
)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _records(value: object) -> list[Mapping[str, object]]:
    return (
        [v for v in value if isinstance(v, Mapping)] if isinstance(value, list) else []
    )


def select_trigger_run(
    sha: str,
    pulls: Sequence[Mapping[str, object]],
    runs: Sequence[Mapping[str, object]],
) -> tuple[int | None, str]:
    """The Runtime Rebuild Trigger run that published the redeploy of ``sha``.

    ``sha`` is a candidate composition, which is a dev merge commit. Its merge's
    own trigger run published a redeploy-start pinned to exactly that sha, and a
    rerun of that run publishes it again, through the same CI job, identity and
    bus credentials a dev merge uses. So the redeploy path needs no local bus
    credential and adds no second publisher.

    ``pulls`` are the PRs GitHub associates with the commit; ``runs`` are that
    workflow's runs, each with its ``jobs``. Returns ``(run id, why)``, or
    ``(None, why not)`` naming the gap. A run qualifies when it is a
    ``pull_request`` run of the merged PR's head, created at or after the merge,
    whose trigger job succeeded and whose verify job ran (it runs only when a
    redeploy was published). The newest qualifying run wins.
    """
    merged = [
        pr
        for pr in pulls
        if pr.get("merge_commit_sha") == sha
        and pr.get("merged_at")
        and _mapping(pr.get("base")).get("ref") == "dev"
    ]
    if not merged:
        return None, (
            f"no merged dev PR has merge commit {sha[:12]}, so there is no rebuild-"
            "trigger run that published its redeploy"
        )
    pr = merged[0]
    head = _mapping(pr.get("head")).get("sha")
    merged_at = str(pr["merged_at"])
    published: list[Mapping[str, object]] = []
    for run in runs:
        if run.get("event") != "pull_request" or run.get("head_sha") != head:
            continue
        if str(run.get("created_at", "")) < merged_at:
            continue
        jobs = {
            str(j.get("name")): str(j.get("conclusion"))
            for j in _records(run.get("jobs"))
        }
        if jobs.get(_TRIGGER_JOB) == "success" and jobs.get(_VERIFY_JOB) in _RAN:
            published.append(run)
    if not published:
        return None, (
            f"PR #{pr.get('number')} (merge {sha[:12]}) has no trigger run that "
            "published a redeploy; rerunning one would publish nothing"
        )
    best = max(published, key=lambda r: str(r.get("created_at", "")))
    return int(str(best["id"])), f"PR #{pr.get('number')} trigger run {best['id']}"


_INFLIGHT: Final[frozenset[str]] = frozenset(
    {"queued", "in_progress", "waiting", "requested", "pending"}
)


def inflight_trigger_runs(runs: Sequence[Mapping[str, object]]) -> list[int]:
    """Trigger runs still queued or running: a deploy is already in flight."""
    return [int(str(r["id"])) for r in runs if r.get("status") in _INFLIGHT]


def _redeploy_via_rerun(sha: str) -> None:
    """Rerun the candidate merge's own trigger run, which republishes its redeploy.

    Any read that fails refuses: nothing is rerun on a partial read.
    """
    try:
        run_id, why = _resolve_rerun(sha)
    except RuntimeError as exc:
        raise SystemExit(f"refused: {exc}; nothing was redeployed") from exc
    if run_id is None:
        raise SystemExit(f"refused: {why}; nothing was redeployed")
    print(
        f"redeploy path: rerun of {why} (the merge's own redeploy-start, pinned to {sha})"
    )
    result = _run(["gh", "run", "rerun", str(run_id), "--repo", OWN_REPO])
    if result.returncode != 0:
        raise SystemExit(
            f"gh run rerun {run_id} failed (exit {result.returncode}): "
            f"{result.stderr.strip()[:300]}; lane untouched by this command"
        )


def _resolve_rerun(sha: str) -> tuple[int | None, str]:
    """Live reads for :func:`select_trigger_run`, refusing while a deploy is in flight."""
    base = f"repos/{OWN_REPO}/actions/workflows/{TRIGGER_WORKFLOW}/runs"
    busy: list[int] = []
    for status in ("queued", "in_progress", "waiting"):
        listing = _mapping(_gh_json(f"{base}?status={status}&per_page=100"))
        busy += inflight_trigger_runs(_records(listing.get("workflow_runs")))
    if busy:
        return None, (
            f"rebuild-trigger run(s) {busy} are in flight; one dev-lane deploy at a time"
        )
    pulls = _records(_gh_json(f"repos/{OWN_REPO}/commits/{sha}/pulls"))
    runs: list[Mapping[str, object]] = []
    for pr in pulls:
        head = _mapping(pr.get("head")).get("sha")
        if not head:
            continue
        listing = _mapping(
            _gh_json(f"{base}?head_sha={head}&event=pull_request&per_page=100")
        )
        for run in _records(listing.get("workflow_runs")):
            jobs = _mapping(
                _gh_json(f"repos/{OWN_REPO}/actions/runs/{run['id']}/jobs?per_page=100")
            )
            runs.append({**run, "jobs": jobs.get("jobs", [])})
    return select_trigger_run(sha, pulls, runs)


def _utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lane", required=True)
    parser.add_argument(
        "--lane-host", required=True, help="ssh target of the lane host"
    )
    parser.add_argument(
        "--bad-sha", default="", help="default: the lane's current infra ref"
    )
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--max-receipts", type=int, default=40)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--readback-timeout", type=int, default=5400)
    args = parser.parse_args(argv)

    lane = LANES.get(args.lane)
    if lane is None:
        print(
            f"refused: --lane {args.lane!r} is not a rollback lane; only "
            f"{', '.join(LANES)} is (no path to production or a governed lane)",
            file=sys.stderr,
        )
        raise SystemExit(2)

    decided_at = _utc()
    print(f"decision_at={decided_at} lane={args.lane} host={args.lane_host}")
    before = read_provenance(args.lane_host, lane)
    current = before.get(lane.provenance_container, {}).get("infra_vcs_ref")
    bad = args.bad_sha or (str(current) if current else "")
    if not _SHA40.match(bad):
        print(
            f"VERDICT INDETERMINATE: cannot read the lane's current composition ({current!r})"
        )
        return 2
    print(f"bad composition: omnibase_infra {bad}")
    _git("fetch", "-q", "origin", lane.base_branch)
    candidates, notes = read_candidates(
        bad, max_pages=args.max_pages, max_receipts=args.max_receipts
    )
    for note in notes:
        print(f"  receipts: {note}")
    ledgers = read_ledgers(args.lane_host, lane)
    for ledger in ledgers:
        count = (
            "UNREADABLE" if ledger.applied is None else f"{len(ledger.applied)} applied"
        )
        print(
            f"  ledger {ledger.database}: {count} ({ledger.foreign_rows} foreign-stream rows) {ledger.error}"
        )
    selection = select_candidate(
        candidates,
        ledgers,
        load_classes(),
        load_down_executions(),
        _checker.DEFAULT_MIGRATIONS_ROOT,  # type: ignore[attr-defined]
    )
    print(
        f"  candidates evaluated newest first: {', '.join(selection.evaluated) or 'none'}"
    )

    published: list[str] = []

    def _redeploy(sha: str) -> None:
        _redeploy_via_rerun(sha)
        published.append(_utc())

    rc = act_on_selection(selection, execute=args.execute, redeploy=_redeploy)
    if not published or selection.candidate is None:
        return rc
    target = selection.candidate.sha
    print(f"redeploy_published_at={published[0]} target={target}")
    deadline = time.monotonic() + args.readback_timeout
    refs: dict[str, str] = {}
    while time.monotonic() < deadline:
        observed = read_provenance(args.lane_host, lane)
        refs = {
            name: str(p["infra_vcs_ref"])
            for name, p in observed.items()
            if p.get("infra_vcs_ref")
        }
        if refs and all(ref == target for ref in refs.values()):
            print(
                f"serving_at={_utc()} every runtime container reads infra_vcs_ref={target}:"
            )
            for name, prov in sorted(observed.items()):
                siblings = prov.get("per_repo_vcs_provenance", {})
                sib = siblings.get("siblings", {}) if isinstance(siblings, dict) else {}
                pinned = {
                    k: v.get("vcs_ref") for k, v in sib.items() if isinstance(v, dict)
                }
                print(
                    f"  {name}: infra={refs[name][:12]} siblings(observed, not pinned)={pinned}"
                )
            return 0
        time.sleep(30)
    print(
        f"VERDICT NOT-CONVERGED at {_utc()}: containers still read {sorted(set(refs.values()))}"
    )
    return 4


if __name__ == "__main__":
    sys.exit(main())
