# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Retire a hot-patch ledger row on evidence the handler checks itself.

OMN-17427: two ``merge_commit: null`` rows blocked every dev redeploy, and the
only defined retirement was a hand edit of the ledger YAML. A row is retired
(``status: reconciled``, never deleted) only when its fix commit is an ancestor
of the deployed ref and the container carries no ``.prepatch``. Anything else
is a refusal that leaves the ledger byte-for-byte untouched.
"""

from __future__ import annotations

import copy
import fcntl
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_hotpatch_ledger_reconcile_request import (
    ModelHotpatchLedgerReconcileRequest,
)
from omnibase_infra.models.model_hotpatch_ledger_reconcile_result import (
    ModelHotpatchLedgerReconcileResult,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

_SUPPORTED_SCHEMA = 1
_PREPATCH_SEARCH_PATHS = ("/app", "/usr/local/lib", "/usr/lib/python3", "/opt")
_COMMIT_ID = re.compile(r"^[0-9a-fA-F]{7,64}$")
_REPO_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_SUBPROCESS_TIMEOUT_SECONDS = 120


class ReconcileRefusedError(Exception):
    """A reason the row must not be retired; carries the operator-facing text."""


class HandlerHotpatchLedgerReconcile:
    """Fail closed: every unverifiable claim is a refusal, never a retirement."""

    def __init__(
        self,
        container: ModelONEXContainer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._container = container
        self._clock = clock or (lambda: datetime.now(UTC))

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self, request: ModelHotpatchLedgerReconcileRequest
    ) -> ModelHotpatchLedgerReconcileResult:
        ledger = Path(request.ledger_path)
        try:
            if not ledger.is_file():
                raise ReconcileRefusedError(f"hot-patch ledger not found at {ledger}")
            with _ledger_lock(ledger):
                return self._reconcile(request, ledger)
        except ReconcileRefusedError as refusal:
            reason = str(refusal)
            return ModelHotpatchLedgerReconcileResult(
                success=False, reason=reason, error_message=reason
            )
        except (OSError, yaml.YAMLError, subprocess.SubprocessError) as exc:
            reason = f"cannot reconcile {ledger}: {exc}"
            return ModelHotpatchLedgerReconcileResult(
                success=False, reason=reason, error_message=reason
            )

    def _reconcile(
        self, request: ModelHotpatchLedgerReconcileRequest, ledger: Path
    ) -> ModelHotpatchLedgerReconcileResult:
        original = ledger.read_bytes()
        document = yaml.safe_load(original)
        if not isinstance(document, dict) or not isinstance(document.get("rows"), list):
            raise ReconcileRefusedError(
                f"ledger at {ledger} is not a mapping with a rows list"
            )
        if document.get("schema") != _SUPPORTED_SCHEMA:
            raise ReconcileRefusedError(
                f"unsupported ledger schema {document.get('schema')!r} "
                f"(expected {_SUPPORTED_SCHEMA})"
            )

        row = _select_row(document["rows"], request)
        candidates = _candidates(row, request.merge_commit)
        repo = row.get("source_repo")
        if not isinstance(repo, str) or not _REPO_NAME.fullmatch(repo):
            raise ReconcileRefusedError(f"row has no usable source_repo: {repo!r}")
        clone = Path(request.clones_root) / repo
        ref = _resolve_ref(clone, request.deployed_ref)
        matched = _ancestor_candidate(clone, candidates, ref)
        found = _live_prepatch_files(request.container, request.docker_cmd)
        if found:
            raise ReconcileRefusedError(
                f"container {request.container!r} still carries .prepatch "
                f"file(s) {found}; the patch is live, rebuild it from merged "
                "source first"
            )

        now = self._clock().astimezone(UTC)
        note = (
            f"verified by hotpatch reconcile: {matched[:12]} is an ancestor of "
            f"{ref[:12]} in {repo}; {request.container} carries 0 .prepatch "
            "file(s)."
        )
        if request.note.strip():
            note = f"{note} {' '.join(request.note.split())}"
        index = next(i for i, other in enumerate(document["rows"]) if other is row)
        before = copy.deepcopy(document["rows"])
        if not row.get("merge_commit"):
            row["merge_commit"] = matched
        row["merged"] = True
        row["status"] = "reconciled"
        row["reconciled_utc"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        row["reconciliation_note"] = note

        backup = _backup(ledger, now)
        _write_atomically(ledger, document)
        reloaded = yaml.safe_load(ledger.read_bytes())["rows"]
        if (
            len(reloaded) != len(before)
            or reloaded[index] != row
            or any(reloaded[i] != before[i] for i in range(len(before)) if i != index)
        ):
            ledger.write_bytes(original)
            raise ReconcileRefusedError(
                "rewritten ledger does not read back as the intended change; "
                f"restored from {backup}"
            )
        return ModelHotpatchLedgerReconcileResult(
            success=True,
            reason=f"retired {request.container} {request.file}: {note}",
            backup_path=str(backup),
            merge_commit=matched,
            deployed_ref=ref,
        )


@contextmanager
def _ledger_lock(ledger: Path) -> Iterator[None]:
    """Exclusive advisory lock beside the ledger for the whole read-modify-write."""
    with ledger.with_name(f"{ledger.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _select_row(
    rows: list[object], request: ModelHotpatchLedgerReconcileRequest
) -> dict[str, object]:
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("container") == request.container
        and row.get("file") == request.file
    ]
    where = f"container {request.container!r} file {request.file!r}"
    if not matches:
        raise ReconcileRefusedError(f"no ledger row for {where}")
    if len(matches) > 1:
        raise ReconcileRefusedError(
            f"more than one ledger row for {where}; refusing to guess"
        )
    row = matches[0]
    status = row.get("status", "active")
    if status == "reconciled":
        raise ReconcileRefusedError(f"ledger row for {where} is already reconciled")
    if status != "active":
        raise ReconcileRefusedError(
            f"ledger row for {where} has unknown status {status!r}"
        )
    return row


def _candidates(row: dict[str, object], supplied: str) -> list[str]:
    recorded_value = row.get("merge_commit")
    if recorded_value is None:
        recorded: list[str] = []
    elif isinstance(recorded_value, str):
        recorded = [recorded_value]
    elif isinstance(recorded_value, list) and all(
        isinstance(item, str) for item in recorded_value
    ):
        recorded = list(recorded_value)
    else:
        raise ReconcileRefusedError(
            f"row merge_commit is malformed: {recorded_value!r}"
        )
    if supplied:
        if recorded and supplied not in recorded:
            raise ReconcileRefusedError(
                f"supplied merge commit {supplied} differs from the row's "
                f"recorded merge commit(s) {', '.join(recorded)}"
            )
        return [supplied]
    if not recorded:
        raise ReconcileRefusedError(
            "the row records no merge commit and none was supplied; pass the "
            "merge commit of its source PR"
        )
    return recorded


def _run(args: list[str], *, git: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        env=scrub_git_location_env(os.environ) if git else None,
    )


def _git(clone: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", "-C", str(clone), *args], git=True)


def _resolve_ref(clone: Path, deployed_ref: str) -> str:
    if deployed_ref.startswith("-"):
        raise ReconcileRefusedError(f"deployed ref {deployed_ref!r} is not a ref")
    result = _git(
        clone,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{deployed_ref or 'HEAD'}^{{commit}}",
    )
    if result.returncode != 0:
        raise ReconcileRefusedError(
            f"cannot resolve deployed ref {deployed_ref or 'HEAD'!r} in clone "
            f"{clone}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _ancestor_candidate(clone: Path, candidates: list[str], ref: str) -> str:
    known: list[str] = []
    for candidate in candidates:
        if not _COMMIT_ID.fullmatch(candidate):
            continue
        resolved = _git(
            clone,
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{candidate}^{{commit}}",
        )
        if resolved.returncode != 0:
            continue
        sha = resolved.stdout.strip()
        known.append(sha)
        ancestry = _git(clone, "merge-base", "--is-ancestor", sha, ref)
        if ancestry.returncode == 0:
            return sha
        if ancestry.returncode != 1:
            raise ReconcileRefusedError(
                f"git merge-base failed in {clone} for {sha}..{ref}: "
                f"{ancestry.stderr.strip()}"
            )
    joined = ", ".join(candidates)
    if not known:
        raise ReconcileRefusedError(
            f"merge commit(s) {joined} unknown in clone {clone}; the commit id "
            "is wrong or the clone is stale (git fetch it first)"
        )
    raise ReconcileRefusedError(
        f"merge commit(s) {', '.join(known)} not an ancestor of deployed ref "
        f"{ref}; the fix is not in what is deployed"
    )


def _live_prepatch_files(container: str, docker_cmd: str) -> list[str]:
    find_cmd = " ".join(
        f'find {path} -name "*.prepatch" 2>/dev/null;'
        for path in _PREPATCH_SEARCH_PATHS
    )
    result = _run([docker_cmd, "exec", container, "sh", "-c", find_cmd])
    if result.returncode != 0:
        raise ReconcileRefusedError(
            f"cannot verify container {container!r} carries no .prepatch: "
            f"{result.stderr.strip()}"
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _backup(ledger: Path, now: datetime) -> Path:
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    for attempt in range(100):
        suffix = "" if attempt == 0 else f"-{attempt}"
        target = ledger.with_name(f"{ledger.name}.bak-reconcile-{stamp}{suffix}")
        if not target.exists():
            shutil.copy2(ledger, target)
            return target
    raise ReconcileRefusedError(f"cannot pick a free backup name beside {ledger}")


def _write_atomically(ledger: Path, document: dict[str, object]) -> None:
    handle, temp_name = tempfile.mkstemp(dir=ledger.parent, prefix=f".{ledger.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            yaml.safe_dump(
                document,
                stream,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )
        shutil.copymode(ledger, temp_name)
        Path(temp_name).replace(ledger)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
